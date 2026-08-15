import argparse
import json
import numpy as np
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
import sys

from util import rotate_about_x, rotate_about_y, rotate_about_z


BASIS_GRAVITY = np.array([0.0, 0.0, -9.8])  # m/s^2 "down"
BASIS_MAGNETIC = np.array([0.0, 5.0e-5, 0.0])  # tesla "north"

DEFAULT_NOISE_ACCELEROMETER = 0.75  # m/s^2
DEFAULT_NOISE_MAGNETOMETER = 1.25e-6  # tesla
DEFAULT_NOISE_GYROSCOPE = 0.05*np.pi  # radians


parser = argparse.ArgumentParser(description="Simulation of Accelerometer in Space")
parser.add_argument("-o", "--output", default="out", help="output directory to save simulation files (will be created)")
parser.add_argument("-f", "--force", action="store_true", default=False, help="set if you want to generate output in an existing output directory")
parser.add_argument("-p", "--number-of-points", type=int, default=10, help="number of points to interpolate through in the simulation")
parser.add_argument("-s", "--number-of-samples", type=int, default=1500, help="number of samples to take on interpolated curves")
parser.add_argument("-n", "--number-of-simulations", type=int, default=10, help="number of simulations to produce")
parser.add_argument("-d", "--duration-of-simulation", type=float, default=10.0, help="length of simulation in seconds")
parser.add_argument("--maximum-velocity", type=float, default=10.0, help="maximum velocity")
parser.add_argument("--maximum-angular-velocity", type=float, default=1.0/8.0*np.pi, help="maximum angular velocity")
parser.add_argument("--noise-accelerometer", type=float, default=DEFAULT_NOISE_ACCELEROMETER, help="accelerometer noise in m/s^2")
parser.add_argument("--noise-magnetometer", type=float, default=DEFAULT_NOISE_MAGNETOMETER, help="magnetometer noise in tesla")
parser.add_argument("--noise-gyroscope", type=float, default=DEFAULT_NOISE_GYROSCOPE, help="gyroscope noise in radians")

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

if args.number_of_simulations <= 0:
    print(f"number of simulations {args.number_of_simulations} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.maximum_velocity <= 0:
    print(f"maximum velocity {args.maximum_velocity} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.maximum_angular_velocity <= 0:
    print(f"maximum angular velocity {args.maximum_angular_velocity} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.noise_accelerometer < 0:
    print(f"accelerometer noise {args.noise_accelerometer} must be >= 0", file=sys.stderr)
    sys.exit(1)

if args.noise_magnetometer < 0:
    print(f"magnetometer noise {args.noise_magnetometer} must be >= 0", file=sys.stderr)
    sys.exit(1)

if args.noise_gyroscope < 0:
    print(f"gyroscope noise {args.noise_gyroscope} must be >= 0", file=sys.stderr)
    sys.exit(1)

out_dir = pathlib.Path(args.output)
if out_dir.exists() and not args.force:
    print(f"output directory {args.output} already exists", file=sys.stderr)
    sys.exit(1)
out_dir.mkdir(exist_ok=args.force)


class Reference():
    def __init__(self, time, velocity, angular_velocity, initial_angular_velocity):
        self.time = time
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.initial_angular_velocity = initial_angular_velocity


class Interpolators():
    def __init__(self, reference):
        self.velocity = CubicSpline(reference.time, reference.velocity)
        self.position = self.velocity.antiderivative()
        self.acceleration = self.velocity.derivative()
        self.angular_velocity = CubicSpline(reference.time, reference.angular_velocity)
        self.rotation = self.angular_velocity.antiderivative()
        self.rotation.c[-1, :] += reference.initial_angular_velocity


class Samples():
    def __init__(self, time, interpolators):
        self.time = time
        self.velocity = interpolators.velocity(time)
        self.position = interpolators.position(time)
        self.acceleration = interpolators.acceleration(time)
        self.angular_velocity = interpolators.angular_velocity(time)
        self.rotation = interpolators.rotation(time)


class Sensor():
    def __init__(self, samples):
        # XXX do these @ work right if we have 3 samples?
        inverse_rotations = -Rotation.from_rotvec(samples.rotation).as_matrix()
        self.accelerometer = (
            inverse_rotations @ (samples.acceleration - BASIS_GRAVITY)[:,:,np.newaxis]
        ).squeeze()
        self.magnetometer = inverse_rotations @ BASIS_MAGNETIC
        self.gyroscope = samples.angular_velocity.copy()


class NoisySensor():
    def __init__(self, sensor, noise_accelerometer, noise_magnetometer, noise_gyroscope):
        self.accelerometer = sensor.accelerometer + np.random.uniform(-noise_accelerometer, noise_accelerometer, size=sensor.accelerometer.shape)
        self.magnetometer = sensor.magnetometer + np.random.uniform(-noise_magnetometer, noise_magnetometer, size=sensor.magnetometer.shape)
        self.gyroscope = sensor.gyroscope + np.random.uniform(-noise_gyroscope, noise_gyroscope, size=sensor.gyroscope.shape)


def create_reference_times(
    number_of_points,
    duration_of_simulation,
):
    return np.linspace(0.0, duration_of_simulation, number_of_points)


def create_sample_times(
    number_of_samples,
    duration_of_simulation,
):
    sample_times = np.linspace(0.0, duration_of_simulation, number_of_samples)

    duration_per_sample = duration_of_simulation / number_of_samples
    jitter_time = duration_per_sample / 5  # TODO maybe it make sense to have this be a parameter?

    while True:
        suitable = True

        # We want times to be roughly equally spaced, so use a normal distribution so we're
        # roughly on the expected references times.
        jitters = np.random.normal(loc=0.0, scale=jitter_time, size=number_of_samples)
        maybe_sample_times = sample_times + jitters

        # Reject jitters that put the samples too close together or reorder them.
        for i in range(number_of_samples - 1):
            if maybe_sample_times[i] >= maybe_sample_times[i + 1]:
                suitable = False
                break

        if suitable:
            # TODO not sure if I should normalize so t0 = 0. ideally we wouldn't have to since essentially
            # the model we train should be concerned with the time deltas primarily.
            sample_times = maybe_sample_times
            break

    return sample_times


def create_random_reference_velocity_and_angular_velocities(
    number_of_points,
    maximum_velocity,
    maximum_angular_velocity,
):
    reference_velocity = np.random.uniform(-maximum_velocity, maximum_velocity, size=(number_of_points, 3))
    reference_angular_velocity = np.random.uniform(-maximum_angular_velocity, maximum_angular_velocity, size=(number_of_points, 3))

    return reference_velocity, reference_angular_velocity


def save_simulation(out_dir, file_name_key, parameters, reference, samples, sensor, noisy_sensor):
        reference_schema = pa.schema([
            ("time", pa.float64()),
            ("velocity_x", pa.float64()),
            ("velocity_y", pa.float64()),
            ("velocity_z", pa.float64()),
            ("angular_velocity_x", pa.float64()),
            ("angular_velocity_y", pa.float64()),
            ("angular_velocity_z", pa.float64()),
        ]).with_metadata({
            b"parameters": json.dumps(parameters).encode(),
        })

        with pq.ParquetWriter(out_dir / f"reference_{file_name_key}.parquet", reference_schema) as writer:
            table = pa.table({
                "time": reference.time,
                "velocity_x": reference.velocity[:, 0],
                "velocity_y": reference.velocity[:, 1],
                "velocity_z": reference.velocity[:, 2],
                "angular_velocity_x": reference.angular_velocity[:, 0],
                "angular_velocity_y": reference.angular_velocity[:, 1],
                "angular_velocity_z": reference.angular_velocity[:, 2],
            }, schema=reference_schema)
            writer.write_table(table)

        target_schema = pa.schema([
            ("time", pa.float64()),
            ("x", pa.float64()),
            ("y", pa.float64()),
            ("z", pa.float64()),
            ("velocity_x", pa.float64()),
            ("velocity_y", pa.float64()),
            ("velocity_z", pa.float64()),
            ("acceleration_x", pa.float64()),
            ("acceleration_y", pa.float64()),
            ("acceleration_z", pa.float64()),
            ("angular_velocity_x", pa.float64()),
            ("angular_velocity_y", pa.float64()),
            ("angular_velocity_z", pa.float64()),
            ("rotation_x", pa.float64()),
            ("rotation_y", pa.float64()),
            ("rotation_z", pa.float64()),
            ("accelerometer_x", pa.float64()),
            ("accelerometer_y", pa.float64()),
            ("accelerometer_z", pa.float64()),
            ("magnetometer_x", pa.float64()),
            ("magnetometer_y", pa.float64()),
            ("magnetometer_z", pa.float64()),
            ("gyroscope_x", pa.float64()),
            ("gyroscope_y", pa.float64()),
            ("gyroscope_z", pa.float64()),
        ])

        with pq.ParquetWriter(out_dir / f"target_{file_name_key}.parquet", target_schema) as writer:
            table = pa.table({
                "time": samples.time,
                "x": samples.position[:, 0],
                "y": samples.position[:, 1],
                "z": samples.position[:, 2],
                "velocity_x": samples.velocity[:, 0],
                "velocity_y": samples.velocity[:, 1],
                "velocity_z": samples.velocity[:, 2],
                "acceleration_x": samples.acceleration[:, 0],
                "acceleration_y": samples.acceleration[:, 1],
                "acceleration_z": samples.acceleration[:, 2],
                "angular_velocity_x": samples.angular_velocity[:, 0],
                "angular_velocity_y": samples.angular_velocity[:, 1],
                "angular_velocity_z": samples.angular_velocity[:, 2],
                "rotation_x": samples.rotation[:, 0],
                "rotation_y": samples.rotation[:, 1],
                "rotation_z": samples.rotation[:, 2],
                "accelerometer_x": sensor.accelerometer[:, 0],
                "accelerometer_y": sensor.accelerometer[:, 1],
                "accelerometer_z": sensor.accelerometer[:, 2],
                "magnetometer_x": sensor.magnetometer[:, 0],
                "magnetometer_y": sensor.magnetometer[:, 1],
                "magnetometer_z": sensor.magnetometer[:, 2],
                "gyroscope_x": sensor.gyroscope[:, 0],
                "gyroscope_y": sensor.gyroscope[:, 1],
                "gyroscope_z": sensor.gyroscope[:, 2],
            }, schema=target_schema)
            writer.write_table(table)

        noisy_schema = pa.schema([
            ("time", pa.float64()),
            ("accelerometer_x", pa.float64()),
            ("accelerometer_y", pa.float64()),
            ("accelerometer_z", pa.float64()),
            ("magnetometer_x", pa.float64()),
            ("magnetometer_y", pa.float64()),
            ("magnetometer_z", pa.float64()),
            ("gyroscope_x", pa.float64()),
            ("gyroscope_y", pa.float64()),
            ("gyroscope_z", pa.float64()),
        ])

        with pq.ParquetWriter(out_dir / f"noisy_{file_name_key}.parquet", noisy_schema) as writer:
            table = pa.table({
                "time": samples.time,
                "accelerometer_x": noisy_sensor.accelerometer[:, 0],
                "accelerometer_y": noisy_sensor.accelerometer[:, 1],
                "accelerometer_z": noisy_sensor.accelerometer[:, 2],
                "magnetometer_x": noisy_sensor.magnetometer[:, 0],
                "magnetometer_y": noisy_sensor.magnetometer[:, 1],
                "magnetometer_z": noisy_sensor.magnetometer[:, 2],
                "gyroscope_x": noisy_sensor.gyroscope[:, 0],
                "gyroscope_y": noisy_sensor.gyroscope[:, 1],
                "gyroscope_z": noisy_sensor.gyroscope[:, 2],
            }, schema=noisy_schema)
            writer.write_table(table)


def run_sample_01():
    parameters = {
        "description": "a stationary unit",
    }

    number_of_points = 10
    number_of_samples = 1000
    duration_of_simulation = 10.0

    reference_velocity = np.zeros(shape=(number_of_points, 3))
    reference_angular_velocity = np.zeros(shape=(number_of_points, 3))
    reference_times = create_reference_times(number_of_points, duration_of_simulation)

    sample_times = create_sample_times(number_of_samples, duration_of_simulation)

    reference = Reference(
        reference_times,
        reference_velocity,
        reference_angular_velocity,
        np.array([0.0, 0.0, 0.0]),
    )
    interpolators = Interpolators(reference)
    samples = Samples(sample_times, interpolators)
    sensor = Sensor(samples)
    noisy_sensor = NoisySensor(
        sensor,
        noise_accelerometer=DEFAULT_NOISE_ACCELEROMETER,
        noise_magnetometer=DEFAULT_NOISE_MAGNETOMETER,
        noise_gyroscope=DEFAULT_NOISE_GYROSCOPE,
    )

    save_simulation(out_dir, "sample01", parameters, reference, samples, sensor, noisy_sensor)

    return reference, samples, sensor, noisy_sensor


def run_sample_02():
    parameters = {
        "description": "a unit accelerating downward",
    }

    number_of_points = 10
    number_of_samples = 1000
    duration_of_simulation = 10.0

    reference_velocity = np.zeros(shape=(number_of_points, 3))
    reference_velocity[:, 2] = np.linspace(0.0, -10, number_of_points).cumsum()
    reference_angular_velocity = np.zeros(shape=(number_of_points, 3))
    reference_times = create_reference_times(number_of_points, duration_of_simulation)

    sample_times = create_sample_times(number_of_samples, duration_of_simulation)

    reference = Reference(
        reference_times,
        reference_velocity,
        reference_angular_velocity,
        np.array([0.0, 0.0, 0.0]),
    )
    interpolators = Interpolators(reference)
    samples = Samples(sample_times, interpolators)
    sensor = Sensor(samples)
    noisy_sensor = NoisySensor(
        sensor,
        noise_accelerometer=DEFAULT_NOISE_ACCELEROMETER,
        noise_magnetometer=DEFAULT_NOISE_MAGNETOMETER,
        noise_gyroscope=DEFAULT_NOISE_GYROSCOPE,
    )

    save_simulation(out_dir, "sample02", parameters, reference, samples, sensor, noisy_sensor)

    return reference, samples, sensor, noisy_sensor


def run_sample_03():
    parameters = {
        "description": "a unit doing a barrel roll",
    }

    number_of_points = 10
    number_of_samples = 1000
    duration_of_simulation = 10.0

    reference_velocity = np.zeros(shape=(number_of_points, 3))
    reference_angular_velocity = (
        np.zeros(shape=(number_of_points, 3)) + np.array([1.0, 0.0, 0.0])
    )

    reference_times = create_reference_times(number_of_points, duration_of_simulation)

    sample_times = create_sample_times(number_of_samples, duration_of_simulation)

    reference = Reference(
        reference_times,
        reference_velocity,
        reference_angular_velocity,
        np.array([0.0, 0.0, 0.0]),
    )
    interpolators = Interpolators(reference)
    samples = Samples(sample_times, interpolators)
    sensor = Sensor(samples)
    noisy_sensor = NoisySensor(
        sensor,
        noise_accelerometer=DEFAULT_NOISE_ACCELEROMETER,
        noise_magnetometer=DEFAULT_NOISE_MAGNETOMETER,
        noise_gyroscope=DEFAULT_NOISE_GYROSCOPE,
    )

    save_simulation(out_dir, "sample03", parameters, reference, samples, sensor, noisy_sensor)

    return reference, samples, sensor, noisy_sensor


def run_sample_04():
    parameters = {
        "description": "sensor go spinny",
    }

    number_of_points = 10
    number_of_samples = 1000
    duration_of_simulation = 10.0

    reference_velocity = np.zeros(shape=(number_of_points, 3))
    reference_angular_velocity = (
        np.zeros(shape=(number_of_points, 3)) + np.array([0.0, 0.0, 1.0])
    )

    reference_times = create_reference_times(number_of_points, duration_of_simulation)

    sample_times = create_sample_times(number_of_samples, duration_of_simulation)

    reference = Reference(
        reference_times,
        reference_velocity,
        reference_angular_velocity,
        np.array([0.0, 0.0, 0.0]),
    )
    interpolators = Interpolators(reference)
    samples = Samples(sample_times, interpolators)
    sensor = Sensor(samples)
    noisy_sensor = NoisySensor(
        sensor,
        noise_accelerometer=DEFAULT_NOISE_ACCELEROMETER,
        noise_magnetometer=DEFAULT_NOISE_MAGNETOMETER,
        noise_gyroscope=DEFAULT_NOISE_GYROSCOPE,
    )

    save_simulation(out_dir, "sample04", parameters, reference, samples, sensor, noisy_sensor)

    return reference, samples, sensor, noisy_sensor


def run_simulation(
    number_of_points,
    number_of_samples,
    duration_of_simulation,
    maximum_velocity,
    maximum_angular_velocity,
    noise_accelerometer,
    noise_magnetometer,
    noise_gyroscope,
):
    reference_velocity, reference_angular_velocity = create_random_reference_velocity_and_angular_velocities(
        number_of_points,
        maximum_velocity,
        maximum_angular_velocity,
    )
    reference_times = create_reference_times(number_of_points, duration_of_simulation)

    sample_times = create_sample_times(number_of_samples, duration_of_simulation)

    initial_angular_velocity = np.random.uniform(0.0, 2.0*np.pi, size=3)

    reference = Reference(
        reference_times,
        reference_velocity,
        reference_angular_velocity,
        initial_angular_velocity,
    )
    interpolators = Interpolators(reference)
    samples = Samples(sample_times, interpolators)
    sensor = Sensor(samples)
    noisy_sensor = NoisySensor(
        sensor,
        noise_accelerometer=noise_accelerometer,
        noise_magnetometer=noise_magnetometer,
        noise_gyroscope=noise_gyroscope,
    )

    return reference, samples, sensor, noisy_sensor


def simulation_harness(
    number_of_simulations,
    number_of_points,
    number_of_samples,
    duration_of_simulation,
    maximum_velocity,
    maximum_angular_velocity,
    noise_accelerometer,
    noise_magnetometer,
    noise_gyroscope,
):
    parameters = {
        "number_of_points": number_of_points,
        "number_of_samples": number_of_samples,
        "duration_of_simulation": duration_of_simulation,
        "maximum_velocity": maximum_velocity,
        "maximum_angular_velocity": maximum_angular_velocity,
        "noise_accelerometer": noise_accelerometer,
        "noise_magnetometer": noise_magnetometer,
        "noise_gyroscope": noise_gyroscope,
    }

    for simulation_no in range(number_of_simulations):
        reference, samples, sensor, noisy_sensor = run_simulation(
            number_of_points=number_of_points,
            number_of_samples=number_of_samples,
            duration_of_simulation=duration_of_simulation,
            maximum_velocity=maximum_velocity,
            maximum_angular_velocity=maximum_angular_velocity,
            noise_accelerometer=noise_accelerometer,
            noise_magnetometer=noise_magnetometer,
            noise_gyroscope=noise_gyroscope,
        )

        save_simulation(out_dir, simulation_no, parameters, reference, samples, sensor, noisy_sensor)

if __name__ == "__main__":
    simulation_harness(
        number_of_simulations=args.number_of_simulations,
        number_of_points=args.number_of_points,
        number_of_samples=args.number_of_samples,
        duration_of_simulation=args.duration_of_simulation,
        maximum_velocity=args.maximum_velocity,
        maximum_angular_velocity=args.maximum_angular_velocity,
        noise_accelerometer=args.noise_accelerometer,
        noise_magnetometer=args.noise_magnetometer,
        noise_gyroscope=args.noise_gyroscope,
    )
