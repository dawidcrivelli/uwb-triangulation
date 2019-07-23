#%%
from __future__ import print_function
import argparse
import logging

import numpy as np
import shelve
import serial
import json
from shapely.geometry import Point, Polygon
from datetime import date
from scipy.linalg import solve
from serial.tools import list_ports
from itertools import cycle
from triangulate import triangulate
from time import sleep, time
from influxdb import InfluxDBClient
from collections import defaultdict

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# plt.ion()
# plt.close()
USE_3d = False


def xy_to_latlng(matrix, zero, p):
    latlng_diff = np.dot(matrix, p)
    latlng_coords = zero + latlng_diff
    return latlng_coords


def latlng_to_xy(matrix, zero, latlng_coords):
    latlng_diff = latlng_coords - zero
    xy = solve(matrix, latlng_diff)
    return xy


def find_area(p, polygons):
    pp = Point(p)
    where = None
    for name, poly in polygons.iteritems():
        if pp.within(poly):
            where = name
            break
    return where

def color_area(p, polygons, coloredareas, label=""):
    where = find_area(p, polygons)

    for area in coloredareas.itervalues():
        area.set_facecolor([0., 0, 0.5, 0.1])

    if where:
        print(label, ",", where)
        coloredareas[where].set_facecolor([0.8, 0, 0, 0.2])
    else:
        print("Zone is unknown!!!")


def setup_logging(name, level='WARNING', logfile=False, logfile_suffix='_log'):
    logging.basicConfig(
        level=level, format='[%(asctime)s] %(levelname)s \t %(message)s', datefmt='%H:%M:%S')
    root_logger = logging.getLogger(name)
    root_logger.setLevel(logging.INFO)

    if logfile:
        fh = logging.FileHandler("{}_{}.log".format(
            name + logfile_suffix, date.today().isoformat()))
        fh.setFormatter(logging.Formatter(
            '[%(asctime)s] %(name)s  %(levelname)s  %(message)s'))
        log = logging.getLogger()
        fh.setLevel(logging.INFO)
        log.addHandler(fh)

    return root_logger

def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Parse lines and insert them into InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='rnd.local',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    parser.add_argument('--database', type=str, default='dcrivelli')
    parser.add_argument('--rp', type=str, default='stream_rp')
    parser.add_argument('--measurement', type=str, default='uwb')
    parser.add_argument('--sourceId', type=str, required=True,
                        help="Name of the tagging field, required")
    parser.add_argument('--extra', type=str, default=None,
                        help="JSON to add as fields")
    parser.add_argument('--points', action='store_true')
    parser.add_argument('--geo', action='store_true')
    parser.add_argument('--map', action='store_true')
    parser.add_argument('--areas', action='store_true')
    parser.add_argument('--no_serial', action='store_true')
    parser.add_argument('--fix-areas', action='store_true')
    parser.add_argument('--no-write', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    return parser.parse_args()

args = parse_args()
log = setup_logging('uwb', logfile='uwb_measurements')

np.set_printoptions(formatter={'float': lambda f: '%5.06f' % f})


""" Load gateways """
all_gateways = json.load(open("gateways.json"))
gateway_list = [g for g in all_gateways if g['active']]
fields = [g['column'] for g in gateway_list]
influxfields = ",".join(["median({}) as {}".format(f, f) for f in fields])

calibrated_p0 = np.array([g['rssi'] for g in gateway_list])
calibrated_dict = {g['name']: g['rssi'] for g in gateway_list}

""" Map data read """
map_data = json.load(open("map.json"))
points = np.array([[g['x'], g['y']] for g in gateway_list])
points_dict = {g['name']: np.array([g['x'], g['y']]) for g in gateway_list}
matrix = np.array(map_data['matrix'])
zerocoords = np.array(map_data['zerocoords'])
geopoints = np.array([xy_to_latlng(matrix, zerocoords, p) for p in points])
position = np.tile(points.mean(axis=0), [10, 1])
floormap = None
bounds = map_data['bounds']
print ('Points loaded.', len(gateway_list), 'gateways out of', len(all_gateways))


""" Load areas """
area_data = json.load(open("areas.json"))
areas = area_data['areas']

polygons = {}
for (name, coords) in areas.items():
    polygons[name] = Polygon(coords)
entityList = {}

gateway_zones = [find_area(p, polygons) for p in points]
zones = defaultdict(lambda: 'Unknown', area_data['mapping'])
assert (set(zones) == set(areas))


if args.no_serial:
    raise SystemExit(0)

try:
    port = (p.device for p in serial.tools.list_ports.comports()
            if 'STM32' in str(p)).next()
except StopIteration as e:
    log.fatal('No suitable serial port connected. TREK1000 not found')
    raise SystemExit(1)

print('Opening up serial connection to', port)
connection = serial.Serial(port=port, baudrate=115200, timeout=10)
np.set_printoptions(formatter={'float': lambda f: '%5.02f' % f})

position = np.tile(points.mean(axis=0), [10, 1])
dist_circles = []

if args.interactive:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(20, 15))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    floormap = plt.imread(map_data['mapfile'])
    ax.imshow(floormap, extent=map_data['extents'], cmap='binary_r')
    current_point, = plt.plot(position[-1, 0], position[-1, 1], 's')
    ellipse = patches.Ellipse(points.mean(axis=0), 1, 1, alpha=0.3, color='r')
    ax.add_patch(ellipse)

    for p in points:
        c = patches.Circle(p, 1, fill=None, alpha=0.4)
        dist_circles.append(c)
        ax.add_patch(c)

        coloredareas = {}

    for (name, coords) in areas.items():
        p = patches.Polygon(coords, edgecolor=[0.4, 0.4, 0.4], facecolor=[0, 0, 0.5, 0.03])
        ax.add_patch(p)
        coloredareas[name] = p
        polygons[name] = Polygon(coords)
        x, y = np.mean(coords, axis=0)

    for g in all_gateways:
        x, y = g['x'], g['y']
        if g['active']:
            ax.plot(x, y, 'o', ms = 14, alpha=0.5)
        else:
            ax.plot(x, y, 'o', color='grey', alpha = 0.5)
        ax.text(x, y + 0.1, g['column'])

    plt.pause(10)



client = InfluxDBClient(args.host, args.port)
# client.create_database(args.database)
client.switch_database(args.database)
# client.create_retention_policy('stream_rp', '52w', 1, default=True)
tags = {'sourceId': args.sourceId, 'trackingId': args.sourceId}

try:
    while True:
        line = connection.readline()
        if not line.startswith('mc'):
            continue
        try:
            fields = line.split()
            dists = np.array(map(lambda x: int(x, 16) / 1000.0, fields[2:5]))
        except ValueError:
            continue

        if any(dists == 0):
            print("Skipping because of zero: ", dists)
            # plt.pause(0.01)
            continue
        results = triangulate(dists, points, position[-1,:])
        X = results.x
        X = [np.clip(X[0], 1.0, 33.0), np.clip(X[1], 1.0, 15.0)]
        zone = find_area(X, polygons)
        
        loc_error = np.abs(results.fun).mean()        

        position[:-1,:] = position[1:,:]
        position[-1,:] = X
        print("{:<.02f}".format(time()), "position:", X, "from distance:", dists)

        mean_pos = position.mean(axis=0)
        stderr = 2 * position.std(axis=0) + 0.2

        try:
            contents = {'x': X[0], 'y': X[1], 'error': 0.1, 'dist1': dists[0],
                        'dist2': dists[1], 'dist3': dists[2], 'zone': zone}
            log.info(json.dumps(contents))
            point = [{'measurement': args.measurement, 'fields': contents, 'tags': tags}]
            if not args.no_write:
                client.write_points(point, retention_policy=args.rp)
        except Exception as e:
            print("Malformed line:", line, "Error:", e)

        if args.interactive:
            for i in range(len(dists)):
                if dists[i] == 0:
                    dist_circles[i].set_fill('r')
                else:
                    dist_circles[i].set_fill(None)
            color_area(X, polygons, coloredareas)
            for i in range(len(points)):
                dist_circles[i].set_radius(dists[i])

            current_point.set_data(position[-1, 0], position[-1, 1])

            ellipse.center = mean_pos
            ellipse.width = stderr[0]
            ellipse.height = stderr[1]
            color_area(X, polygons, coloredareas)
            print("Error circle", ellipse.center, stderr)

            plt.pause(0.01)
except KeyboardInterrupt:
    print("Finishing.")
    pass
