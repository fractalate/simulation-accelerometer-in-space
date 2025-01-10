# Simulation of an Accelerometer in Space

Tools for producing synthetic sensor readings of an accelerometer, gyroscope, and/or magnetometer under various kinds of motion.

## Sample Files

* `sample_position_and_orientation_##.###`
  - Extension `csv`:
    - The first reading in the file is an absolute position and orientation record. All subsequent records are the change in position and orientation.
    - Columns:
      - `time` - Change in time in seconds.
      - `x`, `y`, and `z` - Change in position in 3D Euclidean space in meters.
      - `rx`, `ry`, and `rz` - Change in rotational orientation about the X, Y, and Z axes in portions of a full rotation (1.0 is a full rotation).
  - Extension `metadata.json`:
    - Contains a JSON object with the following attributes:
    <!--
      - `compass_vector.x`, `compass_vector.y`, `compass_vector.z` under the assumption that the simulation takes place in an unvarying magnetic field (e.g. for human scale motion over short distances in Earth's magnetic field), which direction and strength the magnetic field points relative to Earth's magnetic field strength.
      E.g. `{"x":0.0,"y":1.0,"z":0.0}` points "north".
    -->
  - Specific samples:
    - Sample Positional and Orientation Data 01:
      - [`sample_position_and_orientation_01.csv`](./sample_position_and_orientation_01.csv)
      - [`sample_position_and_orientation_01.metadata.json`](./sample_position_and_orientation_01.metadata.json)
      - A sample in which the sensor travels from the origin to every vertex of a 1 meter cube maintaining a consistent orientation, finally returning to the origin. (TODO: validate the sample files).
