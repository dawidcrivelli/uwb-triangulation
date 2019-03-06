#!/usr/bin/python
from __future__ import print_function
import numpy as np
from scipy.optimize import least_squares


def triangulate(dists, points, startpoint=None):
    weights = np.array([1.0, 1.0, 0.3])
    def res(xy):
        return (((xy - points)** 2).sum(axis=1) - dists ** 2) * weights

    if startpoint is None:
        startpoint = points.mean(axis=0)
    result = least_squares(res, startpoint)
    return result
