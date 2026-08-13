# Simulation of an Accelerometer in Space

Tools for producing synthetic sensor readings of an accelerometer, gyroscope, and/or magnetometer under various kinds of motion.

Assumptions:

* We don't travel very far so that the curvature of the planet isn't a concern:
  - North is always the same direction.
  - Gravity always points down as the sensor moves in space.

## Simulation

The simulation models an accelerometer, magnetometer, and gyroscope starting with zero velocity at the origin and outputs synthetic, noisy sensor readings as it moves through space and changes orientation. The velocity must start at zero since it cannot otherwise be inferred from the readings. This is key because we intend to train a machine learning model on the artificial sensor data to predict the position and orientation of the sensor.

For position/velocity/acceleration data, we create a sequence of $P`$ varying reference velocity values as 3D vectors, $`\vec{v}_i`$, with equal time displacement where $`\vec{v}_0 = 0`$. We interpolate through these values with a differentiable function (a cubic spline in this case) to create a continuous function $`\vec{v}(t)`$ that satisfies $`\vec{v}(t_i) = \vec{v}_i`$ for each reference time $`t_i`$. The integral of this function gives us the position of the sensor, $`\vec{x}(t)`$, and the derivative gives us its acceleration, $`\vec{a}(t)`$ (without consideration for its orientation).

For orientation, we create a sequence of $`P`$ varying reference angular velocities as 3D vectors $`\vec{\omega}_i`$ and also select an initial rotation vector $`\vec{R}_0`$. We similarly interpolate through $`\vec{\omega}_i`$ to create a continuous function $`\vec{\omega}(t)`$ that satisfies $`\vec{\omega}(t_i) = \vec{\omega}_i`$ for each reference time $`t_i`$. The integral of this function plus initial rotation $`\vec{R}_0`$ is its rotation, $`\vec{R}(t)`$, satisfying $`\vec{R}(t_0) = \vec{R}_0`$.

To produce accelerometer and magnetometer readings we apply Rodrigues' rotation formula, which we'll denote with $`\mathcal{R}`$, in various ways. Let $`\vec{g}`$ be a vector representing the acceleration due to gravity (it points "down") and $`\vec{m}`$ be a vector representing the planet's magnetic flux density (it points "north").

The target accelerometer reading is $`\vec{a_t}(t) = \mathcal{R}(-\vec{R}(t),\vec{g})+\mathcal{R}(-\vec{R}(t),\vec{a}(t))`$. The target magnetometer reading is $`\vec{m_t}(t) = \mathcal{R}(-\vec{R}(t),\vec{m})`$. The target gyroscope reading is $`\vec{\omega_t}(t) = \vec{\omega}(t)`$.

We choose $`S`$ sample times $`w_k`$ where $`t_0 \le w_k \le t_{P-1}`$ to sample the readings. Values for $`w_k`$ are initially chosen with equal spacing over the time range, but are "jittered" by altering them by amounts chosen from a normal distribution since the sensor operates at a regular interval, but not perfectly. Noisy readings are produced by sampling the true readings at times $`w_k`$ and then adding white noise (via a uniform distribution) to produce $`\vec{a_n}(w_k)`$, $`\vec{m_n}(w_k)`$, and $`\vec{\omega_n}(w_k)`$.

Reference data, target data, and noisy data are saved to separate files:

* Reference data: $`t_i`$, $`\vec{x}(t_i)`$, $`\vec{v}(t_i)`$, $`\vec{a}(t_i)`$, $`\vec{\omega}(t_i)`$, $`\vec{R}(t_i)`$.
* Target data: $`w_k`$, $`\vec{x}(w_k)`$, $`\vec{v}(w_k)`$, $`\vec{a}(w_k)`$, $`\vec{a_t}(w_k)`$, $`\vec{m_t}(w_k)`$, $`\vec{\omega_t}(w_k)`$, $`\vec{R}(w_k)`$.
* Noisy data:  $`w_k`$, $`\vec{a_n}(w_k)`$, $`\vec{m_n}(w_k)`$, $`\vec{\omega_n}(w_k)`$.

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
