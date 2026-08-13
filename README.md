# Simulation of an Accelerometer in Space

Tools for producing synthetic sensor readings of an accelerometer, gyroscope, and/or magnetometer under various kinds of motion.

Assumptions:

* We don't travel very far so that the curvature of the planet isn't a concern:
  - North is always the same direction.
  - Gravity always points down as the sensor moves in space.

## Simulation

The simulation models an accelerometer, magnetometer, and gyroscope starting with zero velocity at the origin and outputs synthetic, noisy sensor readings as it moves through space and changes orientation. The velocity must start at zero since it cannot otherwise be inferred from the readings. This is key because we intend to train a machine learning model on the artificial sensor data to predict the position and orientation of the sensor.

For position/velocity/acceleration data, we create a sequence of $P$ varying reference velocity values as 3D vectors, $\vec{v}_{i}$, with equal time displacement where $\vec{v}_{0} = 0$. We interpolate through these values with a differentiable function (a cubic spline in this case) to create a continuous function $\vec{v}(t)$ that satisfies $\vec{v}(t_{i}) = \vec{v}_{i}$ for each reference time $t_{i}$. The integral of this function gives us the position of the sensor, $\vec{x}(t)$, and the derivative gives us its acceleration, $\vec{a}(t)$ (without consideration for its orientation).

For orientation, we create a sequence of $P$ varying reference angular velocities as 3D vectors $\vec{\omega}_{i}$ and also select an initial rotation vector $\vec{R}_{0}$. We similarly interpolate through $\vec{\omega}_{i}$ to create a continuous function $\vec{\omega}(t)$ that satisfies $\vec{\omega}(t_{i}) = \vec{\omega}_{i}$ for each reference time $t_{i}$. The integral of this function plus initial rotation $\vec{R}_{0}$ is its rotation, $\vec{R}(t)$, satisfying $\vec{R}(t_{0}) = \vec{R}_{0}$.

To produce accelerometer and magnetometer readings we apply Rodrigues' rotation formula, which we'll denote with $\mathcal{R}$, in various ways. Let $\vec{g}$ be a vector representing the acceleration due to gravity (it points "down") and $\vec{m}$ be a vector representing the planet's magnetic flux density (it points "north").

The target accelerometer reading is $\vec{a_{t}}(t) = \mathcal{R}(-\vec{R}(t),\vec{g})+\mathcal{R}(-\vec{R}(t),\vec{a}(t))$. The target magnetometer reading is $\vec{m_{t}}(t) = \mathcal{R}(-\vec{R}(t),\vec{m})$. The target gyroscope reading is $\vec{\omega_{t}}(t) = \vec{\omega}(t)$.

We choose $S$ sample times $w_{k}$ where $t_{0} \le w_{k} \le t_{P-1}$ to sample the readings. Values for $w_{k}$ are initially chosen with equal spacing over the time range, but are "jittered" by altering them by amounts chosen from a normal distribution since the sensor operates at a regular interval, but not perfectly. Noisy readings are produced by sampling the true readings at times $w_{k}$ and then adding white noise (via a uniform distribution) to produce $\vec{a_n}(w_{k})$, $\vec{m_n}(w_{k})$, and $\vec{\omega_n}(w_{k})$.

Reference data, target data, and noisy data are saved to separate files:

* Reference data: $t_{i}$, $\vec{x}(t_{i})$, $\vec{v}(t_{i})$, $\vec{a}(t_{i})$, $\vec{\omega}(t_{i})$, $\vec{R}(t_{i})$.
* Target data: $w_{k}$, $\vec{x}(w_{k})$, $\vec{v}(w_{k})$, $\vec{a}(w_{k})$, $\vec{a_{t}}(w_{k})$, $\vec{m_{t}}(w_{k})$, $\vec{\omega_{t}}(w_{k})$, $\vec{R}(w_{k})$.
* Noisy data:  $w_{k}$, $\vec{a_n}(w_{k})$, $\vec{m_n}(w_{k})$, $\vec{\omega_n}(w_{k})$

Then a model can be trained from the noisy data to predict whichever data is interesting in the target data.

## Setup

Required libraries:

```bash
pip install numpy scipy matplotlib pyarrow
```

## Examples

```
python3 simulation_position_and_orientation.py -o out
```

This will create synthetic sensor data and supporting information in the `out` directory:

* `reference_#.parquet` - Data about sample true reference points.
* `target_#.parquet` - Perfect sensor readings with sensor timing jitter.
* `noisy_#.parquet` - Noisy sensor readings with the same sensor timing jitter.
