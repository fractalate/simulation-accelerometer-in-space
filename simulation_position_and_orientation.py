import argparse
import numpy as np
import pathlib
from scipy.interpolate import CubicSpline
import sys

from util import rotate_about_x, rotate_about_y, rotate_about_z


parser = argparse.ArgumentParser(description="Simulation of Accelerometer in Space")
parser.add_argument("-o", "--output", default="out", help="output directory to save simulation files (will be created)")
parser.add_argument("-f", "--force", action="store_true", default=False, help="set if you want to generate output in an existing output directory")
parser.add_argument("-p", "--number-of-points", type=int, default=10, help="number of points to interpolate through in the simulation")
parser.add_argument("-s", "--number-of-samples", type=int, default=300, help="number of samples to take on interpolated curves")
parser.add_argument("-d", "--duration-of-simulation", type=float, default=10.0, help="length of simulation in seconds")
parser.add_argument("-n", "--number-of-simulations", type=float, default=10.0, help="number of simulations to produce")  # TODO validation
parser.add_argument("--step-size-x", type=float, default=1.0, help="x step size")
parser.add_argument("--step-size-y", type=float, default=1.0, help="y step size")
parser.add_argument("--step-size-z", type=float, default=1.0, help="z step size")
parser.add_argument("--turn-size-x", type=float, default=3.0/8.0*np.pi, help="x turn size in radians")
parser.add_argument("--turn-size-y", type=float, default=3.0/8.0*np.pi, help="y turn size in radians")
parser.add_argument("--turn-size-z", type=float, default=3.0/8.0*np.pi, help="z turn size in radians")
parser.add_argument("--noise-accelerometer", type=float, default=0.05, help="accelerometer noise in m/s^2")  # TODO validation
parser.add_argument("--noise-magnetometer", type=float, default=1.25e-6, help="magnetometer noise in tesla")  # TODO validation
parser.add_argument("--noise-rotation", type=float, default=0.01*np.pi, help="rotation noise in radians")  # TODO validation

args = parser.parse_args()

if args.number_of_points <= 1:
    print(f"number of points {args.number_of_points} must be > 1", file=sys.stderr)
    sys.exit(1)

if args.number_of_samples <= 1:
    print(f"number of samples {args.number_of_samples} must be > 1", file=sys.stderr)
    sys.exit(1)

if args.duration_of_simulation <= 0:
    print(f"duration of simulation {args.duration_of_simulation} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.step_size_x <= 0:
    print(f"step size x {args.step_size_x} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.step_size_y <= 0:
    print(f"step size y {args.step_size_y} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.step_size_z <= 0:
    print(f"step size z {args.step_size_z} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.turn_size_x <= 0:
    print(f"turn size x {args.turn_size_x} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.turn_size_y <= 0:
    print(f"turn size y {args.turn_size_y} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.turn_size_z <= 0:
    print(f"turn size z {args.turn_size_z} must be > 0", file=sys.stderr)
    sys.exit(1)

out_dir = pathlib.Path(args.output)
if out_dir.exists() and not args.force:
    print(f"output directory {args.output} already exists", file=sys.stderr)
    sys.exit(1)
out_dir.mkdir(exist_ok=args.force)


BASIS_GRAVITY = np.array([0.0, 0.0, -9.8])  # Meters/Second
BASIS_MAGNETIC = np.array([0.0, 5.0e-5, 0.0])  # Tesla


def create_reference_locations_and_orientations(
    number_of_points,
    step_size_x,
    step_size_y,
    step_size_z,
    turn_size_x,
    turn_size_y,
    turn_size_z,
):
    x = np.random.uniform(-step_size_x, step_size_x, size=number_of_points).cumsum()
    y = np.random.uniform(-step_size_y, step_size_y, size=number_of_points).cumsum()
    z = np.random.uniform(-step_size_z, step_size_z, size=number_of_points).cumsum()
    # We want x0, y0, z0 = (0, 0, 0) since the starting position doesn't matter and the
    # model we train on the data can assume the position starts there too.
    x -= x[0]
    y -= y[0]
    z -= z[0]

    thetas_initial = np.random.uniform(0.0, 2.0*np.pi, size=3)
    theta_x = np.random.uniform(-turn_size_x, turn_size_x, size=number_of_points).cumsum() + thetas_initial[0]
    theta_y = np.random.uniform(-turn_size_y, turn_size_y, size=number_of_points).cumsum() + thetas_initial[1]
    theta_z = np.random.uniform(-turn_size_z, turn_size_z, size=number_of_points).cumsum() + thetas_initial[2]

    return x, y, z, theta_x, theta_y, theta_z


def create_reference_times(
    number_of_points,
    duration_of_simulation,
):
    return np.linspace(0.0, duration_of_simulation, number_of_points)


def create_interpolators(
    t,
    x,
    y,
    z,
    theta_x,
    theta_y,
    theta_z,
):
    interpolator_x = CubicSpline(t, x)
    interpolator_y = CubicSpline(t, y)
    interpolator_z = CubicSpline(t, z)

    interpolator_theta_x = CubicSpline(t, theta_x)
    interpolator_theta_y = CubicSpline(t, theta_y)
    interpolator_theta_z = CubicSpline(t, theta_z)

    return interpolator_x, interpolator_y, interpolator_z, interpolator_theta_x, interpolator_theta_y, interpolator_theta_z


def create_noisy_sample_times(
    number_of_samples,
    duration_of_simulation,
):
    sample_t = np.linspace(0.0, duration_of_simulation, number_of_samples)

    duration_per_sample = duration_of_simulation / number_of_samples
    jitter_time = duration_per_sample / 5  # TODO maybe it make sense to have this be a parameter?

    while True:
        suitable = True

        # We want times to be roughly equally spaced, so use a normal distribution so we're
        # roughly on the expected references times.
        jitters = np.random.normal(loc=0.0, scale=jitter_time, size=number_of_samples)
        maybe_sample_t = sample_t + jitters

        # Reject jitters that put the samples too close together or reorder them.
        for i in range(number_of_samples - 1):
            if maybe_sample_t[i] >= maybe_sample_t[i + 1]:
                suitable = False
                break

        if suitable:
            # TODO not sure if I should normalize so t0 = 0. ideally we wouldn't have to since essentially
            # the model we train should be concerned with the time deltas primarily.
            sample_t = maybe_sample_t
            break

    return sample_t


def create_target_sample_locations(
    sample_t,
    interpolator_x,
    interpolator_y,
    interpolator_z,
):
    sample_x = interpolator_x(sample_t)
    sample_y = interpolator_y(sample_t)
    sample_z = interpolator_z(sample_t)

    return sample_x, sample_y, sample_z


def calculate_sample_sensor_readings(
    sample_t,
    interpolator_x,
    interpolator_y,
    interpolator_z,
    interpolator_theta_x,
    interpolator_theta_y,
    interpolator_theta_z,
):
    # 2nd derivative of position is acceleration.
    interpolator_acceleration_x = interpolator_x.derivative(2)
    interpolator_acceleration_y = interpolator_y.derivative(2)
    interpolator_acceleration_z = interpolator_z.derivative(2)

    sample_acceleration_absolute_x = interpolator_acceleration_x(sample_t)
    sample_acceleration_absolute_y = interpolator_acceleration_y(sample_t)
    sample_acceleration_absolute_z = interpolator_acceleration_z(sample_t)
    sample_acceleration_absolute = np.stack([
        sample_acceleration_absolute_x,
        sample_acceleration_absolute_y,
        sample_acceleration_absolute_z,
    ]).T

    sample_theta_x = interpolator_theta_x(sample_t)
    sample_theta_y = interpolator_theta_y(sample_t)
    sample_theta_z = interpolator_theta_z(sample_t)

    # 1st derivative of orientation angle is angular velocity.
    interpolator_angular_velocity_x = interpolator_theta_x.derivative(1)
    interpolator_angular_velocity_y = interpolator_theta_y.derivative(1)
    interpolator_angular_velocity_z = interpolator_theta_z.derivative(1)

    sample_angular_velocity_x = interpolator_angular_velocity_x(sample_t)
    sample_angular_velocity_y = interpolator_angular_velocity_y(sample_t)
    sample_angular_velocity_z = interpolator_angular_velocity_z(sample_t)

    # Theta measures the orientation of the unit and rotation has the inverse reaction on the
    # sensor reading, so invert the angles and create the naive rotation matrix ("naive" because
    # it's not the 4D rotation matrix or quaternions which are generally better in practice).
    rotate_x = rotate_about_x(-sample_theta_x)
    rotate_y = rotate_about_y(-sample_theta_y)
    rotate_z = rotate_about_z(-sample_theta_z)
    rotation_action_matrix = (
        rotate_z.transpose(2, 0, 1) @
        rotate_y.transpose(2, 0, 1) @
        rotate_x.transpose(2, 0, 1)
    )

    sample_gravity = (rotation_action_matrix @ BASIS_GRAVITY).T
    # Need to expand sample_acceleration_absolute for @ broadcasting. (N, 3, 3) @ (N, 3, 1) -> (N, 3)
    accel_vector_oriented = (rotation_action_matrix @ sample_acceleration_absolute[:,:,np.newaxis]).squeeze()
    sample_acceleration = accel_vector_oriented.T + sample_gravity

    sample_magnetic = (rotation_action_matrix @ BASIS_MAGNETIC).T

    sample_angular_velocity = np.stack([
        sample_angular_velocity_x,
        sample_angular_velocity_y,
        sample_angular_velocity_z,
    ]).T

    return (
        sample_acceleration,
        sample_magnetic,
        sample_angular_velocity,
    )


def run_simulation(
    number_of_points,
    duration_of_simulation,
    number_of_samples,
    step_size_x,
    step_size_y,
    step_size_z,
    turn_size_x,
    turn_size_y,
    turn_size_z,
    noise_accelerometer,
    noise_magnetometer,
    noise_rotation,
):
    t = create_reference_times(
        number_of_points=number_of_points,
        duration_of_simulation=duration_of_simulation,
    )

    x, y, z, theta_x, theta_y, theta_z = create_reference_locations_and_orientations(
        number_of_points=number_of_points,
        step_size_x=step_size_x,
        step_size_y=step_size_y,
        step_size_z=step_size_z,
        turn_size_x=turn_size_x,
        turn_size_y=turn_size_y,
        turn_size_z=turn_size_z,
    )

    (
        interpolator_x, interpolator_y, interpolator_z,
        interpolator_theta_x, interpolator_theta_y, interpolator_theta_z 
    ) = create_interpolators(
        t=t,
        x=x, y=y, z=z,
        theta_x=theta_x,
        theta_y=theta_y,
        theta_z=theta_z,
    )

    sample_t = create_noisy_sample_times(
        number_of_samples=number_of_samples,
        duration_of_simulation=duration_of_simulation,
    )

    (
        sample_x,
        sample_y,
        sample_z,
    ) = create_target_sample_locations(
        sample_t=sample_t,
        interpolator_x=interpolator_x,
        interpolator_y=interpolator_y,
        interpolator_z=interpolator_z,
    )

    (
        sample_acceleration,
        sample_magnetic,
        sample_angular_velocity,
    ) = calculate_sample_sensor_readings(
        sample_t=sample_t,
        interpolator_x=interpolator_x,
        interpolator_y=interpolator_y,
        interpolator_z=interpolator_z,
        interpolator_theta_x=interpolator_theta_x,
        interpolator_theta_y=interpolator_theta_y,
        interpolator_theta_z=interpolator_theta_z,
    )

    # We want white noise on the sensor, so use a uniform distribution.
    noisy_acceleration = sample_acceleration + np.random.uniform(-noise_accelerometer, noise_accelerometer, size=sample_acceleration.shape)
    noisy_magnetic_vector = sample_magnetic + np.random.uniform(-noise_magnetometer, noise_magnetometer, size=sample_magnetic.shape)
    noisy_angular_velocity = sample_angular_velocity + np.random.uniform(-noise_rotation, noise_rotation, size=sample_angular_velocity.shape)

    return (
        sample_t,
        sample_x,
        sample_y,
        sample_z,
        noisy_acceleration,
        noisy_magnetic_vector,
        noisy_angular_velocity,
    )


(
    sample_t,
    sample_x,
    sample_y,
    sample_z,
    noisy_acceleration,
    noisy_magnetic_vector,
    noisy_angular_velocity,
) = run_simulation(
    number_of_points=args.number_of_points,
    duration_of_simulation=args.duration_of_simulation,
    number_of_samples=args.number_of_samples,
    step_size_x=args.step_size_x,
    step_size_y=args.step_size_y,
    step_size_z=args.step_size_z,
    turn_size_x=args.turn_size_x,
    turn_size_y=args.turn_size_y,
    turn_size_z=args.turn_size_z,
    noise_accelerometer=args.noise_accelerometer,
    noise_magnetometer=args.noise_magnetometer,
    noise_rotation=args.noise_rotation,
)

# TODO write to output file
