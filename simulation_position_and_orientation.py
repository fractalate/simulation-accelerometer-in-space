import json
import os
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
data = PositionAndOrientationData('sample_position_and_orientation_01')
cumsum_data = data.df.cumsum()

# Raw Samples
t = cumsum_data['time']
x = cumsum_data['x']
y = cumsum_data['y']
z = cumsum_data['z']
theta_x = cumsum_data['rx'].values * 2 * np.pi
theta_y = cumsum_data['ry'].values * 2 * np.pi
theta_z = cumsum_data['rz'].values * 2 * np.pi

# returns (target, noisy) out data
def interpolate_and_add_noise(t, x, y, z, theta_x, theta_y, theta_z, num_out_samples=1000):
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
     dsc_t = np.linspace(t[0], t[len(t) - 1], num_out_samples)
     dsc_theta_x = cs_theta_x(dsc_t)
     dsc_theta_y = cs_theta_y(dsc_t)
     dsc_theta_z = cs_theta_z(dsc_t)
     dsc_ax = cs_ax(dsc_t)
     dsc_ay = cs_ay(dsc_t)
     dsc_az = cs_az(dsc_t)
     dsc_omega_x = cs_omega_x(dsc_t)
     dsc_omega_y = cs_omega_y(dsc_t)
     dsc_omega_z = cs_omega_z(dsc_t)

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
     #print(f'{dsc_gravity.shape=}')
     dsc_magnetic = ((dsc_rot_action_rev) @ basis_magnetic).T
     #print(f'{dsc_magnetic.shape=}')

     accel_vector_absolute = np.stack([dsc_ax, dsc_ay, dsc_az]).T
     #print(f'{accel_vector_absolute.shape=}')
     # Need to expand accel_vector_absolute for @ broadcasting. (N, 3, 3) @ (N, 3, 1) -> (N, 3)
     accel_vector_oriented = (dsc_rot_action_rev @ accel_vector_absolute[:,:,np.newaxis]).squeeze()
     accel_vector = accel_vector_oriented.T + dsc_gravity
     #print(f'{accel_vector.shape=}')
     magnetic_vector = dsc_magnetic
     print(f'{magnetic_vector.shape=}')

     out = np.concat([[dsc_t], accel_vector, [dsc_omega_x], [dsc_omega_y], [dsc_omega_z], magnetic_vector])

     #print(f'{out.shape=}')
     #with open('out_target.csv', 'w') as fout:
     #     for t,ax,ay,az,ox,oy,oz,mx,my,mz in out.transpose(1, 0):
     #          fout.write(','.join(str(value) for value in [t,ax,ay,az,ox,oy,oz,mx,my,mz]) + '\n')

     # Make some noise!!!
     noisy_accel_vector = accel_vector + np.random.normal(size=accel_vector.shape, scale=1.0)  # TODO parameterize
     noisy_magnetic_vector = magnetic_vector + np.random.normal(size=magnetic_vector.shape, scale=0.1)  # TODO parameterize
     noisy_dsc_omega_x = dsc_omega_x + np.random.normal(size=dsc_omega_x.shape, scale=0.1)  # TODO parameterize
     noisy_dsc_omega_y = dsc_omega_y + np.random.normal(size=dsc_omega_y.shape, scale=0.1)  # TODO parameterize
     noisy_dsc_omega_z = dsc_omega_z + np.random.normal(size=dsc_omega_z.shape, scale=0.1)  # TODO parameterize

     out_noisy = np.concat([[dsc_t], noisy_accel_vector, [noisy_dsc_omega_x], [noisy_dsc_omega_y], [noisy_dsc_omega_z], noisy_magnetic_vector])
     #print(f'{out_noisy.shape=}')
     #with open('out_noisy.csv', 'w') as fout:
     #     for t,ax,ay,az,ox,oy,oz,mx,my,mz in out_noisy.transpose(1, 0):
     #          fout.write(','.join(str(value) for value in [t,ax,ay,az,ox,oy,oz,mx,my,mz]) + '\n')

     return out.T, out_noisy.T

# out is one of the returned items from interpolate_and_add_noise()
def save_out_data(filename, out):
     with open(filename, 'w') as fout:
          fout.write('t,ax,ay,az,gx,gy,gz,mx,my,mz\n')
          for t,ax,ay,az,ox,oy,oz,mx,my,mz in out:
               fout.write(','.join(str(value) for value in [t,ax,ay,az,ox,oy,oz,mx,my,mz]) + '\n')

if __name__ == '__main__':
     for run in range(100, -1, -1):
          t = np.arange(10)
          x = np.random.uniform(-0.5, 0.5, size=t.size).cumsum()
          y = np.random.uniform(-0.5, 0.5, size=t.size).cumsum()
          z = np.random.uniform(-0.5, 0.5, size=t.size).cumsum()
          theta_x = np.random.uniform(0.0, 0.75 * np.pi, size=t.size).cumsum()
          theta_y = np.random.uniform(0.0, 0.75 * np.pi, size=t.size).cumsum()
          theta_z = np.random.uniform(0.0, 0.75 * np.pi, size=t.size).cumsum()

          num_out_samples = 1000
          target, noisy = interpolate_and_add_noise(t, x, y, z, theta_x, theta_y, theta_z, num_out_samples=num_out_samples)

          os.makedirs('data', exist_ok=True)

          save_out_data(f'data/out_{run:03d}_target.csv', target)
          save_out_data(f'data/out_{run:03d}_noisy.csv', noisy)

          if run == 0:
               cs_x = CubicSpline(t, x)
               cs_y = CubicSpline(t, y)
               cs_z = CubicSpline(t, z)

               dsc_t = np.linspace(t[0], t[len(t) - 1], num_out_samples)
               dsc_x = cs_x(dsc_t)
               dsc_y = cs_y(dsc_t)
               dsc_z = cs_z(dsc_t)

               # Some redundant stuff for pretty pictures
               cs_theta_x = CubicSpline(t, theta_x)  # TODO: Is a cubic spline good for this quantity? Would linear be better?
               cs_theta_y = CubicSpline(t, theta_y)
               cs_theta_z = CubicSpline(t, theta_z)

               dsc_theta_x = cs_theta_x(dsc_t)
               dsc_theta_y = cs_theta_y(dsc_t)
               dsc_theta_z = cs_theta_z(dsc_t)

               dsc_rot_mat_x = rotate_about_x(dsc_theta_x)
               dsc_rot_mat_y = rotate_about_y(dsc_theta_y)
               dsc_rot_mat_z = rotate_about_z(dsc_theta_z)

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

               fig = plt.figure()
               ax = fig.add_subplot(111, projection='3d')
               ax.set_box_aspect([1, 1, 1])
               ax.plot(dsc_x, dsc_y, dsc_z, label='Interpolated Curve')
               ax.quiver(x, y, z, *ox_samples, color='red', length=0.05, normalize=True)
               ax.quiver(x, y, z, *oy_samples, color='green', length=0.05, normalize=True)
               ax.quiver(x, y, z, *oz_samples, color='blue', length=0.05, normalize=True)
               ax.scatter(x, y, z, color='black', label='Original Points')
               ax.set_xlabel('X')
               ax.set_ylabel('Y')
               ax.set_zlabel('Z')
               ax.legend()
               plt.show()

"""
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
"""

"""
# Sensor Orientation Matrixes
dsc_rot_mat_x = rotate_about_x(dsc_theta_x)
dsc_rot_mat_y = rotate_about_y(dsc_theta_y)
dsc_rot_mat_z = rotate_about_z(dsc_theta_z)
"""

# TODO: How does this look for more dense Hamiltonian cycles

"""
set datafile separator ','
plot 'out_000_target.csv' using 1:2 with lines title 'ax', \
     'out_000_target.csv' using 1:3 with lines title 'ay', \
     'out_000_target.csv' using 1:4 with lines title 'az'

set datafile separator ','
plot 'out_000_target.csv' using 1:5 with lines title 'gx', \
     'out_000_target.csv' using 1:6 with lines title 'gy', \
     'out_000_target.csv' using 1:7 with lines title 'gz'

set datafile separator ','
plot 'out_000_target.csv' using 1:8 with lines title 'mx', \
     'out_000_target.csv' using 1:9 with lines title 'my', \
     'out_000_target.csv' using 1:10 with lines title 'mz'
     
set datafile separator ','
plot 'out_000_noisy.csv' using 1:2 with lines title 'ax', \
     'out_000_noisy.csv' using 1:3 with lines title 'ay', \
     'out_000_noisy.csv' using 1:4 with lines title 'az'

set datafile separator ','
plot 'out_000_noisy.csv' using 1:5 with lines title 'gx', \
     'out_000_noisy.csv' using 1:6 with lines title 'gy', \
     'out_000_noisy.csv' using 1:7 with lines title 'gz'

set datafile separator ','
plot 'out_000_noisy.csv' using 1:8 with lines title 'mx', \
     'out_000_noisy.csv' using 1:9 with lines title 'my', \
     'out_000_noisy.csv' using 1:10 with lines title 'mz'
"""
