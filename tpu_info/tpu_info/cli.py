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

  table = rich.table.Table(
      title="TPU Runtime Utilization", title_justify="left"
  )
  table.add_column("Device")
  table.add_column("HBM usage")
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

  table = rich.table.Table(title="TensorCore Utilization", title_justify="left")
  table.add_column("Chip ID")
  table.add_column("TensorCore Utilization", justify="right")

  try:
    # pylint: disable=g-import-not-at-top
    from libtpu import sdk  # pytype: disable=import-error

    tensorcore_util_data = sdk.monitoring.get_metric("tensorcore_util").data()
  except ImportError as e:
    print(f"WARNING: ImportError: {e}.")
  except AttributeError as e:
    print(
        f"WARNING: {e}. Please check if the latest libtpu is used"
    )
  except RuntimeError as e:
    print(
        f"WARNING: {e}. Please check if the latest vbar control agent is used."
    )
  else:
    for i in range(len(tensorcore_util_data)):
      tc_data = f"{tensorcore_util_data[i]}%"
      table.add_row(
          str(i),
          tc_data,
      )
    console.print(table)

  table = rich.table.Table(
      title="TPU Buffer Transfer Latency", title_justify="left"
  )
  table.add_column("Buffer Size")
  table.add_column("P50", justify="right")
  table.add_column("P90", justify="right")
  table.add_column("P95", justify="right")
  table.add_column("P999", justify="right")

  try:
    buffer_transfer_latency_distributions = (
        metrics.get_buffer_transfer_latency()
    )
  except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      print(
          "WARNING: Buffer Transfer Latency metrics unavailable. Did you start"
          " a MULTI_SLICE workload with"
          " `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?"
      )
    else:
      print(f"ERROR: {e}")

    buffer_transfer_latency_distributions = []

  for distribution in buffer_transfer_latency_distributions:
    table.add_row(
        distribution.buffer_size,
        f"{distribution.p50:.2f} us",
        f"{distribution.p90:.2f} us",
        f"{distribution.p95:.2f} us",
        f"{distribution.p999:.2f} us",
    )
  console.print(table)
