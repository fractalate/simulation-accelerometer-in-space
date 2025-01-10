import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def getOutputFilenamePrefix(run: int):
    return f'out_position_and_orientation_{run:04d}'

def getOutputFilename(run: int, kind: str):
    assert kind in ['reading', 'reference', 'metadata']
    if kind in ['reading', 'reference']:
        return f'{getOutputFilenamePrefix(run)}.{kind}.csv'
    elif kind in ['metadata']:
        return f'{getOutputFilenamePrefix(run)}.{kind}.json'
    assert False  # Womp, womp.

def rotate_about_x(vector, theta):
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ]) @ vector

def rotate_about_y(vector, theta):
    return np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ]) @ vector

def rotate_about_z(vector, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]) @ vector

class PositionAndOrientationData():
    # `prefix` is the input sample prefix from which the associated CSV and JSON metadata will be loaded.
    def __init__(self, prefix: str):
        file_csv_path = prefix + '.csv'
        file_metadata_path = prefix + '.metadata.json'

        self._prefix = prefix

        with open(file_metadata_path, 'r') as fin:
            self.metadata = json.loads(fin.read())

        # t,x,y,z,rx,ry,rz
        self.df = pd.read_csv(file_csv_path)


data = PositionAndOrientationData('sample_position_and_orientation_01')
cumsum_data = data.df.cumsum()

t = cumsum_data['time' ]
x = cumsum_data['x']
y = cumsum_data['y']
z = cumsum_data['z']
rx = cumsum_data['rx'].values * 2 * np.pi
ry = cumsum_data['ry'].values * 2 * np.pi
rz = cumsum_data['rz'].values * 2 * np.pi

cs_x = CubicSpline(t, x)
cs_y = CubicSpline(t, y)
cs_z = CubicSpline(t, z)
cs_rx = CubicSpline(t, rx)
cs_ry = CubicSpline(t, ry)
cs_rz = CubicSpline(t, rz)

t_new = np.linspace(t[0], t[len(t) - 1], 100)
x_new = cs_x(t_new)
y_new = cs_y(t_new)
z_new = cs_z(t_new)
rx_new = cs_rx(t_new)
ry_new = cs_ry(t_new)
rz_new = cs_rz(t_new)

def rotate_about_x(theta):
    return np.array([
        [1.0 + 0.0 * theta, 0.0 * theta, 0.0 * theta],
        [0.0 * theta, np.cos(theta), -np.sin(theta)],
        [0.0 * theta, np.sin(theta), np.cos(theta)],
    ])

def rotate_about_y(theta):
    return np.array([
        [np.cos(theta), 0.0 * theta, np.sin(theta)],
        [0.0 * theta, 1.0 + 0.0 * theta, 0.0 * theta],
        [-np.sin(theta), 0.0 * theta, np.cos(theta)],
    ])

def rotate_about_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0 * theta],
        [np.sin(theta), np.cos(theta), 0.0 * theta],
        [0.0 * theta, 0.0 * theta, 1.0 + theta * 0.0],
    ])

rmz = rotate_about_z(rz)
rmy = rotate_about_y(ry)
rmx = rotate_about_x(rx)

rots = (rmz.transpose(2, 0, 1) @ rmy.transpose(2, 0, 1)) @ rmx.transpose(2, 0, 1)

basis_x = np.array([1.0, 0.0, 0.0])
basis_y = np.array([0.0, 1.0, 0.0])
basis_z = np.array([0.0, 0.0, 1.0])

ox = (rots @ basis_x).transpose(1, 0)
oy = (rots @ basis_y).transpose(1, 0)
oz = (rots @ basis_z).transpose(1, 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_new, y_new, z_new, label='Interpolated Curve')
ax.quiver(x, y, z, *ox, color='red', length=0.25, normalize=True)
ax.quiver(x, y, z, *oy, color='green', length=0.25, normalize=True)
ax.quiver(x, y, z, *oz, color='blue', length=0.25, normalize=True)
ax.scatter(x, y, z, color='black', label='Original Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
