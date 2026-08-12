# Simulation of an Accelerometer in Space

Tools for producing synthetic sensor readings of an accelerometer, gyroscope, and/or magnetometer under various kinds of motion.

Assumptions:

* We don't travel very far so that the curvature of the planet isn't a concern:
  - North is always the same direction.
  - Gravity always points down as the sensor moves in space.

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
