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

# TODO - This really depends on where in the world you are. Is it worth modelling that fact in
# the produced samples? Or is the fact that the gravity vector should be orthogonal to it enough?
def COMPASS_VECTOR() -> Vec3:
    return np.array([1.0, 0.0, 0.0])

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

def make_tilt_vector(tilt_std_dev: Vec1 = np.pi / 32.0) -> Vec3:
    theta_x = np.random.normal(loc=0.0, scale=tilt_std_dev)
    theta_z = np.random.uniform(0.0, 2 * np.pi)
    return np.array([theta_x, 0.0, theta_z])

def create_gravity_vector(tilt_vector: Vec3, g = 9.8) -> Vec3:
    rx, ry, rz = tilt_vector
    return rotate_about_z(rotate_about_y(rotate_about_x(GRAVITY_VECTOR(g=g), -rx), -ry), -rz)

# TODO when the unit can simulate movement, this will need to be implemented.
def create_gyroscope_vector(tilt_vector: Vec3) -> Vec3:
    return np.array([0.0, 0.0, 0.0])

def create_compass_vector(tilt_vector: Vec3) -> Vec3:
    rx, ry, rz = tilt_vector
    return rotate_about_z(rotate_about_y(rotate_about_x(COMPASS_VECTOR(), -rx), -ry), -rz)

# Sensor units and the software reading the values aren't perfect. In order to simulate
# the possible uneven timing of samples, choose times from the period of time of interest
# uniformly at the unit's average sample rate.
def pick_times(samples_per_second: Vec1, seconds: Vec1):
    number_of_samples = max(1, np.int64(np.floor(seconds * samples_per_second)))
    samples = [np.random.uniform(0.0, seconds) for _ in range(number_of_samples)]
    samples.sort()
    return np.array(samples)

for run_no in range(0, 10+1):
    if run_no == 0:
        # A set of reference samples.
        tilt_vector = np.array([0.0, 0.0, 0.0])
        true_accelerometer_reading = GRAVITY_VECTOR()
        true_gyroscope_reading = np.array([0.0, 0.0, 0.0])
        true_compass_reading = COMPASS_VECTOR()
    else:
        tilt_vector = make_tilt_vector()
        true_accelerometer_reading = create_gravity_vector(tilt_vector)
        true_gyroscope_reading = create_gyroscope_vector(tilt_vector)
        true_compass_reading = create_compass_vector(tilt_vector)

    times = pick_times(132.0, 60.0)

    fout_metadata_name = f'out_{run_no:03d}.metadata.json'
    print(f'Generating {fout_metadata_name}...')
    with open(fout_metadata_name, 'w') as fout:
        fout.write(json.dumps({
            "true_accelerometer_reading": list(true_accelerometer_reading),
            "true_gyroscope_reading": list(true_gyroscope_reading),
            "true_compass_reading": list(true_compass_reading),
        }, indent=2))
        fout.write('\n')

    fout_name = f'out_{run_no:03d}.csv'
    print(f'Generating {fout_name}...')
    with open(fout_name, 'w') as fout:
        for t in times:
            # Variation of 0.2 on roughly 9.8 is about 2% variation.
            # Variation of 1.0 on roughly 9.8 is about 10% variation.
            accel = true_accelerometer_reading + np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
            # Variation of 0.1 on roughly 1.0 is about 10% variation. TODO unsure on reasonable units to use, so the variation is just there for general noise.
            gyro = true_gyroscope_reading + np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)])
            # Variation of 0.1 on roughly 1.0 is about 10% variation.
            compass = true_compass_reading + np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)])
            fout.write(f'{t},{accel[0]},{accel[1]},{accel[2]},{gyro[0]},{gyro[1]},{gyro[2]},{compass[0]},{compass[1]},{compass[2]}\n')

"""
set datafile separator ','
plot 'out_000.csv' using 1:2 with lines title 'ax', \
     'out_000.csv' using 1:3 with lines title 'ay', \
     'out_000.csv' using 1:4 with lines title 'az'

set datafile separator ','
plot 'out_000.csv' using 1:5 with lines title 'gx', \
     'out_000.csv' using 1:6 with lines title 'gy', \
     'out_000.csv' using 1:7 with lines title 'gz'

set datafile separator ','
plot 'out_000.csv' using 1:8 with lines title 'mx', \
     'out_000.csv' using 1:9 with lines title 'my', \
     'out_000.csv' using 1:10 with lines title 'mz'


set datafile separator ','
plot 'out_002.csv' using 1:2 with lines title 'ax', \
     'out_002.csv' using 1:3 with lines title 'ay', \
     'out_002.csv' using 1:4 with lines title 'az'

set datafile separator ','
plot 'out_002.csv' using 1:5 with lines title 'gx', \
     'out_002.csv' using 1:6 with lines title 'gy', \
     'out_002.csv' using 1:7 with lines title 'gz'

set datafile separator ','
plot 'out_002.csv' using 1:8 with lines title 'mx', \
     'out_002.csv' using 1:9 with lines title 'my', \
     'out_002.csv' using 1:10 with lines title 'mz'
"""

# I found some of my old samples and these will plot the acceleration, gyroscope,
# and magnetometer readings. Some notes on them: they use different units that I'm
# not quite sure how to translate to typical physical quantities. The unit outputs
# integer quantities.
#
# Some of the quantities (especially for the gyroscope) appear to be biased. In
# most of my old samples, the unit was stationary to collect reference samples
# for this kind of analysis.
#
# TODO Look through sample data some more to get an idea of how the data can vary.
#
# Preliminary notes: capturing roughly 132 or 133 samples per second

"""
set datafile separator ','
plot 'samples.20220531.172046.log' using 1:2 with lines title 'ax', \
     'samples.20220531.172046.log' using 1:3 with lines title 'ay', \
     'samples.20220531.172046.log' using 1:4 with lines title 'az'

set datafile separator ','
plot 'samples.20220531.172046.log' using 1:5 with lines title 'gx', \
     'samples.20220531.172046.log' using 1:6 with lines title 'gy', \
     'samples.20220531.172046.log' using 1:7 with lines title 'gz'

set datafile separator ','
plot 'samples.20220531.172046.log' using 1:8 with lines title 'mx', \
     'samples.20220531.172046.log' using 1:9 with lines title 'my', \
     'samples.20220531.172046.log' using 1:10 with lines title 'mz'
"""
