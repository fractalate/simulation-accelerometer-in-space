import numpy as np
from scipy.interpolate import CubicSpline

def rotate_about_x(theta):
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array([
        [one, zero, zero],
        [zero, np.cos(theta), -np.sin(theta)],
        [zero, np.sin(theta), np.cos(theta)],
    ])

def rotate_about_y(theta):
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array([
        [np.cos(theta), zero, np.sin(theta)],
        [zero, one, zero],
        [-np.sin(theta), zero, np.cos(theta)],
    ])

def rotate_about_z(theta):
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta), zero],
        [np.sin(theta), np.cos(theta), zero],
        [zero, zero, one],
    ])
