#%%
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import argparse
import numpy as np
import shelve
import easygui
import serial
import json
from shapely.geometry import Point, Polygon

from scipy.linalg import solve
from serial.tools import list_ports
from itertools import cycle
from triangulate import triangulate
from time import sleep, time
from influxdb import InfluxDBClient

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

plt.ion()
plt.close()
USE_3d = False


def xy_to_latlng(data, p):
    zero = data['zerocoords']
    matrix = data['matrix']

    latlng_diff = np.dot(matrix, p)
    latlng_coords = zero + latlng_diff
    return latlng_coords


def latlng_to_xy(data, latlng_coords):
    zero = data['zerocoords']
    matrix = data['matrix']

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


def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Parse lines and insert them into InfluxDB')
    parser.add_argument('--host', type=str, required=False,
                        default='localhost',
                        help='hostname of InfluxDB http API')
    parser.add_argument('--port', type=int, required=False, default=8086,
                        help='port of InfluxDB http API')
    parser.add_argument('--database', type=str, default='telemetry')
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
    return parser.parse_args()

args = parse_args()

data = shelve.open("storage.pkl", writeback=True)
np.set_printoptions(formatter={'float': lambda f: '%5.06f' % f})

if 'map-ok' not in data or args.map:
    filename = easygui.fileopenbox("Filename to map")
    image = mpimg.imread(filename).mean(axis=2)

    scale = float(easygui.enterbox("Enter scale, in pixels / meter"))

    data['map'] = image
    data['scale'] = scale
    x0, y0 = 0, 0
    extents = [-x0, image.shape[1] / scale -
               x0, -y0, image.shape[0] / scale - y0]

    plt.imshow(data['map'], extent=extents, cmap='binary_r')
    print('Click to set coordinate zero')
    # ((x0, y0),) = plt.ginput()
    (x0, y0) = (155.0 / scale, 192.0 / scale)

    data['zero'] = (x0, y0)
    plt.close()

    extents = [-x0, image.shape[1] / scale - x0, -y0, image.shape[0] / scale - y0]
    data['extents'] = extents
    data['map-ok'] = True

print('Map data read')
plt.imshow(data['map'], extent=data['extents'], cmap='binary_r')
plt.plot(0, 0, '+')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.show()
plt.pause(1)

if not data.has_key('geocoords') or args.geo:
    zero = np.array(map(float, easygui.enterbox(
        "Enter LAT,LNG coords for zero").split(",")))
    data['zerocoords'] = zero

    print("Select X coordinate point")
    # X = np.array(plt.ginput(1)).ravel()
    X = np.array([33.66, 0])
    X[np.abs(X) < 0.1] = 0
    Xvec = np.array(map(float, easygui.enterbox(
        "Enter LAT,LNG coords for point X").split(",")))

    print("Select XY coordinate point")
    # XY = np.array(plt.ginput(1)).ravel()
    XY = np.array([33.66, 15.66])
    # XY[np.abs(XY) < 0.1] = 0
    XYvec = np.array(map(float, easygui.enterbox(
        "Enter LAT,LNG coords for point XY").split(",")))

    Y = XY - X
    Yvec = XYvec - Xvec

    data['geocoords'] = (((0,0), zero), (X, Xvec), (XY, XYvec))

    x = (Xvec - zero)  / np.sqrt(X.dot(X))
    y = (Yvec) / np.sqrt(Y.dot(Y))

    matrix = np.vstack((x, y)).T

    data['matrix'] = matrix
    data.sync()


if 'points' not in data or args.points:
    N = easygui.integerbox("Number of points", default=3)
    points = []
    geopoints = []
    for i in range(N):
        print("Enter point", i+1)
        p = np.array(plt.ginput(n=1, timeout=-1)).ravel()
        points.append(p)
        plt.show()
        plt.pause(0.01)
    data['points'] = np.array(points)
    data.sync()

    if USE_3d:
        points_3d = np.zeros((N, 3))
        points_3d[:, :-1] = points
        points_3d[:, 2] = [2.05, 2.05, 2.25]
        points = points_3d


print('Points loaded')
points = data['points']
geopoints = []
for i, p in enumerate(points):
    x, y = p[0], p[1]
    plt.plot(x, y, 'o')
    plt.text(x, y + 0.1, 'A{}'.format(i+1))
    if data.has_key('matrix'):
        latlng_coords = xy_to_latlng(data, p)
        print("Point", i+1, "XY", p, "GEO", latlng_coords)
        geopoints.append(latlng_coords)
data['geopoints'] = np.array(geopoints)
for ((x,y), label) in data['geocoords']:
    plt.plot(x, y, 'x')
    plt.text(x, y+0.2, str(label))
plt.show()
plt.pause(1)


if 'areas' not in data or args.areas:
    areas = {}
    for item in json.loads(open("areas.json").read()):
        name, latlng_coords = item['name'], item['coordinates']
        xycoords = np.array([latlng_to_xy(data, [c[1], c[0]]) for c in latlng_coords])
        print(name, latlng_coords, xycoords)

        areas[name] = xycoords
    data['areas'] = areas

areas = data['areas']
polygons = {}
coloredareas = {}
ax  = plt.gca()
for (name, coords) in areas.iteritems():
    p = patches.Polygon(coords, edgecolor=[0.5, 0.5, 0.5], facecolor=[0, 0, 0.5, 0.1])
    ax.add_patch(p)
    coloredareas[name] = p
    polygons[name] = Polygon(coords)
    x, y = coords.mean(axis=0)
    plt.text(x, y, str(name))
    latlng_coords = np.array([xy_to_latlng(data, point) for point in coords])
    plt.show()
    plt.pause(0.1)

    change = easygui.ynbox("Want to change it?") if args.fix_areas else False
    if change:
        coords = np.array(plt.ginput(4))
        areas[name] = coords
        p.remove()
        p = patches.Polygon(coords, edgecolor=[0.5, 0.5, 0.5], facecolor=[0, 0, 0.5, 0.1])
        ax.add_patch(p)
        coloredareas[name] = p
        polygons[name] = Polygon(coords)
        plt.show()
        plt.pause(1.0)

    print("Area ", name, ", coords XY: ", (coords), ", latlng: ", (latlng_coords))
    data['areas'] = areas
    data.sync()


for i, p in enumerate(points):
    color_area(p, polygons, coloredareas, label="A{}".format(i + 1))
    plt.pause(0.2)
color_area([-10, -10], polygons, coloredareas)

data.close()

if args.no_serial:
    raise SystemExit(0)

port = (p.device for p in serial.tools.list_ports.comports()
        if 'STM32' in str(p)).next()

print('Opening up serial connection to', port)
connection = serial.Serial(port=port, baudrate=115200, timeout=10)
np.set_printoptions(formatter={'float': lambda f: '%5.02f' % f})

position = np.tile(points.mean(axis=0), [10, 1])
current_point, = plt.plot(position[-1, 0], position[-1, 1], 's')
ellipse = patches.Ellipse(points.mean(axis=0), 1, 1, alpha=0.3, color='r')
plt.gca().add_patch(ellipse)

dist_circles = []
for p in points:
    c = patches.Circle(p, 1, fill=None, alpha=0.4)
    dist_circles.append(c)
    ax.add_patch(c)


client = InfluxDBClient(args.host, args.port)
client.create_database(args.database)
client.switch_database(args.database)
# client.create_retention_policy('stream_rp', '52w', 1, default=True)
tags = {'sourceId': args.sourceId}

try:
    while True:
        line = connection.readline()
        if not line.startswith('mc'):
            continue

        try:
            fields = line.split()
            dists = np.array(map(lambda x: int(x, 16) / 1000.0, fields[2:5]))
            for i in range(len(dists)):
                if dists[i] == 0:
                    dist_circles[i].set_fill('r')
                else:
                    dist_circles[i].set_fill(None)
        except ValueError:
            continue

        if any(dists == 0):
            print("Skipping because of zero: ", dists)
            plt.pause(0.01)
            continue
        results = triangulate(dists, points, position[-1,:])
        X = results.x
        X = [np.clip(X[0], 1.0, 33.0), np.clip(X[1], 1.0, 15.0)]
        zone = find_area(X, polygons)
        color_area(X, polygons, coloredareas)
        loc_error = np.abs(results.fun).mean()
        # print(results)

        position[:-1,:] = position[1:,:]
        position[-1,:] = X
        print("{:<.02f}".format(time()), "position:", X, "from distance:", dists)

        mean_pos = position.mean(axis=0)
        stderr = 2 * position.std(axis=0) + 0.2
        ellipse.center = mean_pos
        ellipse.width = stderr[0]
        ellipse.height = stderr[1]
        color_area(X, polygons, coloredareas)
        # print("Error circle", ellipse.center, stderr)

        try:
            contents = {'x': X[0], 'y': X[1], 'dist1': dists[0],
                        'dist2': dists[1], 'dist3': dists[2], 'zone': zone}
            point = [{'measurement': 'uwb', 'fields': contents, 'tags': tags}]
            if not args.no_write:
                client.write_points(point)
        except Exception as e:
            print("Malformed line:", line, "Error:", e)

        for i in range(len(points)):
            dist_circles[i].set_radius(dists[i])

        current_point.set_data(position[-1, 0], position[-1, 1])

        plt.pause(0.01)
except KeyboardInterrupt:
    print("Finishing.")
    pass
