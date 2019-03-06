#%%
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import shelve
import easygui
import serial
from serial.tools import list_ports
from itertools import cycle
from triangulate import triangulate
from time import sleep

plt.ion()


#%%
data = shelve.open("storage.pkl", writeback=True)

if 'map-ok' not in data:
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
    ((x0, y0),) = plt.ginput()

    data['zero'] = (x0, y0)
    plt.close()

    extents = [-x0, image.shape[1] / scale - x0, -y0, image.shape[0] / scale - y0]
    data['extents'] = extents

    data['map-ok'] = True

print('Map data read')
plt.imshow(data['map'], extent=data['extents'], cmap='binary_r')
plt.plot(0, 0, '+')

if 'points' not in data:
    N = easygui.integerbox("Number of points", default=3)
    points = plt.ginput(N)

    data['points'] = np.array(points)

print('Points loaded')
points = data['points']
for i, p in enumerate(points):
    x, y = p[0], p[1]
    plt.plot(x, y, 'o')
    plt.text(x, y + 0.1, 'A{}'.format(i))

plt.show()
plt.pause(0.01)
data.close()

port = (p.device for p in serial.tools.list_ports.comports()
        if 'STM32' in str(p)).next()

print('Opening up serial connection to', port)
connection = serial.Serial(port=port, baudrate=115200, timeout=10)

position = np.tile(points.mean(axis=0), [10, 1])
current_point, = plt.plot(position[:, 0], position[:, 1], 's')
ellipse = patches.Ellipse(points.mean(axis=0), 1, 1, alpha=0.3, color='r')
plt.gca().add_patch(ellipse)

dist_circles = []
for p in points:
    c = patches.Circle(p, 1, fill=None, alpha=0.4)
    dist_circles.append(c)
    plt.gca().add_patch(c)

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
                    dist_circles[i].set_color('r')
                    print("Skipping because of zero: ", dists)
                    continue
                else:
                    dist_circles[i].set_color(None)
        except ValueError:
            continue

        results = triangulate(dists, points, position[-1, :])
        (xp, yp) = results.x
        loc_error = np.abs(results.fun).mean()
        # print(results)

        position[:-1,:] = position[1:,:]
        position[-1,:] = (xp, yp)
        print("New position", results.x, "from distances", dists)

        mean_pos = position.mean(axis=0)
        stderr = 2 * position.std(axis=0)
        ellipse.center = mean_pos
        ellipse.width = stderr[0]
        ellipse.height = stderr[1]
        print("Error circle", ellipse.center, stderr)

        for i in range(len(points)):
            dist_circles[i].set_radius(dists[i])

        current_point.set_data(position[:, 0], position[:, 1])
        plt.pause(0.01)
except KeyboardInterrupt:
    print("Finishing.")
    pass
