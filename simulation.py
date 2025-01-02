from typing import Annotated
from numpy.typing import NDArray

import json
import numpy as np

# Vector with 1 component.
Vec1 = Annotated[NDArray[np.float64], (1,)]
# Vector with 3 components.
Vec3 = Annotated[NDArray[np.float64], (3,)]

# Orientation is a unit vector indicating the direction that the top of the sensor is facing.
# The default orientation is one where the sensor's top is facing upwards, indicated by a vector
# pointing solely in the Z direction.
def ORIENTATION_DEFAULT() -> Vec3:
    return np.array([0.0, 0.0, 1.0])

def GRAVITY_VECTOR(g = 9.8) -> Vec3:
    return np.array([0.0, 0.0, -g])

def rotate_about_x(vector: Vec3, theta: Vec1) -> Vec3:
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ]) @ vector

def rotate_about_y(vector: Vec3, theta: Vec1) -> Vec3:
    return np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ]) @ vector

def rotate_about_z(vector: Vec3, theta: Vec1) -> Vec3:
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]) @ vector

"""
# The sensor's orientation is described as rotations about the x, y, and z axes.
# As such, the gravitational vector that the sensor would read is calculated by
# applying such rotations in reverse to the default gravity vector. For example,
# if the sensor is rotated pi/6 radians clockwise about the x axis, then the
# gravity vector is rotated pu/6 radians anti-clockwise about the x axis.
sensor_rotations: Vec3 = np.array([np.pi/2, 0.0, 0.0])
rx, ry, rz = sensor_rotations
gravity = rotate_about_z(rotate_about_y(rotate_about_x(GRAVITY_VECTOR(), -rx), -ry), -rz)
print(gravity)
"""

# We want to generate some simulated control data for the sensor where the sensor is
# placed flat on a surface in arbitrary rotated orientations (sensor is sitting on a
# table, but is maybe rotated and not facing any particular direction). However a
# sensor unit may not be perfectly square, so it will be tilted a little bit.
# Rotation of the unit should be determined from a uniform random variable.
# Tilt of the unit should be determined by a normal random variable.
# It should be possible to rotate the unit about the x axis to handle tilt and to
# rotate the unit about the z axis for arbitrary rotations.

def create_rotations(tilt_std_dev: Vec1 = np.pi / 32.0) -> Vec3:
    theta_x = np.random.normal(loc=0.0, scale=tilt_std_dev)
    theta_z = np.random.uniform(0.0, 2 * np.pi)
    return np.array([theta_x, 0.0, theta_z])

def create_gravity_vector(tilt_std_dev: Vec1 = np.pi / 32.0, g = 9.8) -> Vec3:
    rx, ry, rz = create_rotations(tilt_std_dev=tilt_std_dev)
    return rotate_about_z(rotate_about_y(rotate_about_x(GRAVITY_VECTOR(g=g), -rx), -ry), -rz)

# Sensor units and the software reading the values aren't perfect. In order to simulate
# the possible uneven timing of samples, choose times from the period of time of interest
# uniformly at the unit's average sample rate.
def pick_times(samples_per_second: Vec1, seconds: Vec1):
    number_of_samples = max(1, np.int64(np.floor(seconds * samples_per_second)))
    samples = [np.random.uniform(0.0, seconds) for _ in range(number_of_samples)]
    samples.sort()
    return np.array(samples)

for run_no in range(1, 10+1):
    true_reading = create_gravity_vector()
    times = pick_times(30.0, 30.0)

    fout_metadata_name = f'out_{run_no:03d}.metadata.json'
    print(f'Generating {fout_metadata_name}...')
    with open(fout_metadata_name, 'w') as fout:
        fout.write(json.dumps({
            "true_reading": list(true_reading),
        }, indent=2))
        fout.write('\n')

    fout_name = f'out_{run_no:03d}.csv'
    print(f'Generating {fout_name}...')
    with open(fout_name, 'w') as fout:
        for t in times:
            # Variation of 0.2 on roughly 9.8 is about 2% variation.
            reading = true_reading + np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
            fout.write(f'{t},{reading[0]},{reading[1]},{reading[2]}\n')
