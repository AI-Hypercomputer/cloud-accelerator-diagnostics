# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines command line interface for `tpu-info` tool.

Top-level functions should be added to `project.scripts` in `pyproject.toml`.
"""

from tpu_info import device
from tpu_info import metrics
import grpc
import rich.console
import rich.table


def _bytes_to_gib(size: int) -> float:
  return size / (1 << 30)


def print_chip_info():
  """Print local TPU devices and libtpu runtime metrics."""
  # TODO(wcromar): Merge all of this info into one table
  chip_type, count = device.get_local_chips()
  if not chip_type:
    print("No TPU chips found.")
    return

  console = rich.console.Console()

  table = rich.table.Table(title="TPU Chips", title_justify="left")
  table.add_column("Chip")
  table.add_column("Type")
  table.add_column("Devices")
  # TODO(wcromar): this may not match the libtpu runtime metrics
  # table.add_column("HBM (per core)")
  table.add_column("PID")

  chip_paths = [device.chip_path(chip_type, index) for index in range(count)]
  chip_owners = device.get_chip_owners()

  for chip in chip_paths:
    owner = chip_owners.get(chip)

    table.add_row(
        chip,
        str(chip_type),
        str(chip_type.value.devices_per_chip),
        str(owner),
    )

  console.print(table)

  table = rich.table.Table(title="TPU Utilization", title_justify="left")
  table.add_column("Device")
  table.add_column("Memory usage")
  table.add_column("Duty cycle", justify="right")

  try:
    device_usage = metrics.get_chip_usage(chip_type)
  except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      print(
          "WARNING: Libtpu metrics unavailable. Is there a framework using the"
          " TPU? See"
          " https://github.com/google/cloud-accelerator-diagnostics/tree/main/tpu_info"
          " for more information"
      )
    else:
      print(f"ERROR: {e}")

    device_usage = [metrics.Usage(i, -1, -1, -1) for i in range(count)]

  # TODO(wcromar): take alternative ports as a flag
  print("Connected to libtpu at grpc://localhost:8431...")
  for chip in device_usage:
    if chip.memory_usage < 0:
      memory_usage = "N/A"
    else:
      memory_usage = (
          f"{_bytes_to_gib(chip.memory_usage):.2f} GiB /"
          f" {_bytes_to_gib(chip.total_memory):.2f} GiB"
      )
    if chip.duty_cycle_pct < 0:
      duty_cycle_pct = "N/A"
    else:
      duty_cycle_pct = f"{chip.duty_cycle_pct:.2f}%"
    table.add_row(
        str(chip.device_id),
        memory_usage,
        duty_cycle_pct
        if chip_type.value.devices_per_chip == 1 or chip.device_id % 2 == 0
        else "",
    )

  console.print(table)
