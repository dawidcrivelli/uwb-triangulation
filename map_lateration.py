import numpy as np
lat, lng = 50.050923, 19.983022
phi = lat / 180 * np.pi

K1 = (111.13209 - 0.56605 * np.cos(2 * phi) + 0.00120 * np.cos(4 * phi)) * 1000.0
K2 = (111.41513 * np.cos(phi) - 0.09455 * np.cos(3 * phi) + 0.00012 * np.cos(5 * phi)) * 1000.0


def inverse_matrix():
    return np.diag([1 / K1, 1 / K2])


def degrees_to_meters(x):
    xvec = x.ravel()
    return np.array([K1 * xvec[0], K2 * xvec[1]])


def distance_angles(x, y):
    X = degrees_to_meters(np.array([x[0] - y[0], x[1] - y[1]]))
    return np.sqrt(X.dot(X))


p0 = (50.051013, 19.982839)
pX = (50.050784, 19.983139)
pY = (50.051105, 19.983003)


pg0 = (50.0510108, 19.9828327)
pgX = (50.0507779, 19.9831330)
pgXY = (50.0508675, 19.9832992)

pGPS = (50.050810, 19.983340)
pCALC = (50.05087618, 19.98330276)
print distance_angles(p0, pX)
print distance_angles(p0, pY)
print distance_angles(p0, pg0)
print distance_angles(pX, pgX)
print distance_angles(pgXY, pGPS)
print distance_angles(pgXY, pCALC)

p19new = [50.05098508, 19.9830187]
p19old = [50.05099045, 19.98302338]
print distance_angles(p19old, p19new)
