import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from util import rotate_about_x, rotate_about_y, rotate_about_z

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

# In our space:
# - "Down" is negative Z.
# - "North" is positive Y.
basis_gravity = np.array([0.0, 0.0, -9.8]) # meters/second
basis_magnetic = np.array([0.0, 1.0, 0.0]) # teslas (TODO get an appropriate value)

# Our data comes as an initial position and orientation followed by deltas.
# Use cummsum so that we get the absolute position and orientation of the sensor.
data = PositionAndOrientationData('sample_position_and_orientation_03')
cumsum_data = data.df.cumsum()

# Raw Samples
t = cumsum_data['time']
x = cumsum_data['x']
y = cumsum_data['y']
z = cumsum_data['z']
theta_x = cumsum_data['rx'].values * 2 * np.pi
theta_y = cumsum_data['ry'].values * 2 * np.pi
theta_z = cumsum_data['rz'].values * 2 * np.pi

# Cubic Splines of Position and Orientation
cs_x = CubicSpline(t, x)
cs_y = CubicSpline(t, y)
cs_z = CubicSpline(t, z)
cs_theta_x = CubicSpline(t, theta_x)  # TODO: Is a cubic spline good for this quantity? Would linear be better?
cs_theta_y = CubicSpline(t, theta_y)
cs_theta_z = CubicSpline(t, theta_z)

# Accelerations
cs_ax = cs_x.derivative(2)
cs_ay = cs_y.derivative(2)
cs_az = cs_z.derivative(2)

# Angular Velocities
cs_omega_x = cs_theta_x.derivative(1)
cs_omega_y = cs_theta_y.derivative(1)
cs_omega_z = cs_theta_z.derivative(1)

# Discretization
dsc_t = np.linspace(t[0], t[len(t) - 1], 100) # TODO: Parameterize the fineness
dsc_x = cs_x(dsc_t)
dsc_y = cs_y(dsc_t)
dsc_z = cs_z(dsc_t)
dsc_theta_x = cs_theta_x(dsc_t)
dsc_theta_y = cs_theta_y(dsc_t)
dsc_theta_z = cs_theta_z(dsc_t)
dsc_ax = cs_ax(dsc_t)
dsc_ay = cs_ay(dsc_t)
dsc_az = cs_az(dsc_t)
dsc_omega_x = cs_omega_x(dsc_t)
dsc_omega_y = cs_omega_y(dsc_t)
dsc_omega_z = cs_omega_z(dsc_t)

# Sensor Orientation Matrixes
dsc_rot_mat_x = rotate_about_x(dsc_theta_x)
dsc_rot_mat_y = rotate_about_y(dsc_theta_y)
dsc_rot_mat_z = rotate_about_z(dsc_theta_z)
# Reference Action Matrixes
dsc_rot_mat_x_rev = rotate_about_x(-dsc_theta_x)
dsc_rot_mat_y_rev = rotate_about_y(-dsc_theta_y)
dsc_rot_mat_z_rev = rotate_about_z(-dsc_theta_z)
dsc_rot_action_rev = (
    dsc_rot_mat_z_rev.transpose(2, 0, 1) @
    dsc_rot_mat_y_rev.transpose(2, 0, 1) @
    dsc_rot_mat_x_rev.transpose(2, 0, 1)
)

# We use dsc_rot_action_rev since the rotation action acts inversely on the reference vectors
dsc_gravity = ((dsc_rot_action_rev) @ basis_gravity).T
print(f'{dsc_gravity.shape=}')
dsc_magnetic = ((dsc_rot_action_rev) @ basis_magnetic).T
print(f'{dsc_magnetic.shape=}')

accel_vector_absolute = np.stack([dsc_ax, dsc_ay, dsc_az]).T
print(f'{accel_vector_absolute.shape=}')
accel_vector_oriented = dsc_rot_action_rev @ accel_vector_absolute[:,:,np.newaxis]
accel_vector = accel_vector_absolute.T + dsc_gravity
print(f'{accel_vector.shape=}')
magnetic_vector = dsc_magnetic
print(f'{magnetic_vector.shape=}')

out = np.concat([[dsc_t], accel_vector, [dsc_omega_x], [dsc_omega_y], [dsc_omega_z], magnetic_vector])
print(f'{out.shape=}')

with open('out.csv', 'w') as fout:
    for t,ax,ay,az,ox,oy,oz,mx,my,mz in out.transpose(1, 0):
        fout.write(','.join(str(value) for value in [t,ax,ay,az,ox,oy,oz,mx,my,mz]) + '\n')

# Stuff for pretty pictures
rmz_samples = rotate_about_z(theta_z)
rmy_samples = rotate_about_y(theta_y)
rmx_samples = rotate_about_x(theta_x)
rots_samples = (rmz_samples.transpose(2, 0, 1) @ rmy_samples.transpose(2, 0, 1)) @ rmx_samples.transpose(2, 0, 1)
basis_x = np.array([1.0, 0.0, 0.0])
basis_y = np.array([0.0, 1.0, 0.0])
basis_z = np.array([0.0, 0.0, 1.0])
rots = (dsc_rot_mat_z.transpose(2, 0, 1) @ dsc_rot_mat_y.transpose(2, 0, 1)) @ dsc_rot_mat_x.transpose(2, 0, 1)
ox_samples = (rots_samples @ basis_x).transpose(1, 0)
oy_samples = (rots_samples @ basis_y).transpose(1, 0)
oz_samples = (rots_samples @ basis_z).transpose(1, 0)

# Pretty pictures
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.plot(dsc_x, dsc_y, dsc_z, label='Interpolated Curve')
ax.quiver(x, y, z, *ox_samples, color='red', length=0.25, normalize=True)
ax.quiver(x, y, z, *oy_samples, color='green', length=0.25, normalize=True)
ax.quiver(x, y, z, *oz_samples, color='blue', length=0.25, normalize=True)
ax.scatter(x, y, z, color='black', label='Original Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

# TODO: How does this look for more dense Hamiltonian cycles
