#!/usr/bin/python
from __future__ import print_function
import numpy as np
from scipy.optimize import least_squares


def triangulate(dists, points, startpoint=None):
    def res(xy):
        return ((xy - points)** 2).sum(axis=1) - dists ** 2

    if startpoint is None:
        startpoint = points.mean(axis=0)
    result = least_squares(res, startpoint)
    return result
