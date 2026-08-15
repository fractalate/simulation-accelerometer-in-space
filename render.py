import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import pyarrow.parquet as pq


parser = argparse.ArgumentParser(description="Render Simulation Data")
parser.add_argument("input", help="input parquet file to render")
parser.add_argument("-f", "--fields", required=True, help="comma-separated list of fields to plot")

args = parser.parse_args()

input_file = pathlib.Path(args.input)
if not input_file.exists():
    print(f"input file {args.input} does not exist", file=sys.stderr)
    sys.exit(1)

fields = [field.strip() for field in args.fields.split(",") if field.strip()]
if len(fields) == 0:
    print("at least one field must be selected", file=sys.stderr)
    sys.exit(1)

table = pq.read_table(input_file)
field_names = table.schema.names

if "time" not in field_names:
    print(f"input file {args.input} does not have a time field", file=sys.stderr)
    sys.exit(1)

missing_fields = [field for field in fields if field not in field_names]
if len(missing_fields) > 0:
    print(
        f"input file {args.input} does not have selected field(s): {', '.join(missing_fields)}",
        file=sys.stderr,
    )
    print(f"available fields: {', '.join(field_names)}", file=sys.stderr)
    sys.exit(1)

time = table["time"].to_numpy()

for field in fields:
    plt.plot(time, table[field].to_numpy(), label=field)

plt.xlabel("time")
plt.ylabel("value")
plt.title(input_file.name)
plt.legend()
plt.tight_layout()
plt.show()
