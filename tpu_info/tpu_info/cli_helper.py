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

"""Helper functions for the CLI."""

import importlib.metadata
import subprocess
from typing import Any, Optional

from tpu_info import device
from tpu_info import metrics
import grpc
from rich import console
from rich import panel
from rich import table as rich_table


def _bytes_to_gib(size: int) -> float:
  """Converts bytes to gibibytes."""
  return size / (1 << 30)


def fetch_cli_version() -> str:
  """Returns the version of the current TPU CLI."""
  try:
    version = importlib.metadata.version("tpu-info")
  except importlib.metadata.PackageNotFoundError:
    version = "unknown (package metadata not found)"
  return version


def fetch_libtpu_version() -> str:
  """Returns the version of the current libtpu."""
  try:
    # pylint: disable=g-import-not-at-top
    import libtpu  # pytype: disable=import-error

    version = libtpu.__version__
    return version
  except Exception:  # pylint: disable=broad-exception-caught
    try:
      result = subprocess.run(
          ["pip", "list"],
          capture_output=True,
          text=True,
          check=True,
      )
      for line in result.stdout.splitlines():
        if "libtpu" in line:
          parts = line.split()
          if len(parts) >= 2:
            return parts[1]
      return "unknown (libtpu not found)"
    except subprocess.CalledProcessError as e:
      return f"unknown (error running pip list: {e})"
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"unknown (unexpected error getting libtpu version: {e})"


def fetch_accelerator_type() -> str:
  """Returns the accelerator type of the current TPU."""
  try:
    chip_type, _ = device.get_local_chips()

    if chip_type is None:
      return "unknown (no TPU chips found)"

    return chip_type.value.name
  except Exception as e:  # pylint: disable=broad-exception-caught
    return f"unknown (unexpected error getting accelerator type: {e})"


def get_process_name(pid: int) -> Optional[str]:
  """Returns the process name for a given PID."""
  try:
    with open(f"/proc/{pid}/comm", "r") as f:
      return f.read().strip()
  except FileNotFoundError:
    return None


def fetch_process_table(
    chip_type: device.TpuChip, count: int
) -> rich_table.Table:
  """Returns a rich.table.Table with process info for the given TPU chip."""
  table = rich_table.Table(title="TPU Process Info", title_justify="left")
  table.add_column("Chip")
  table.add_column("PID")
  table.add_column("Process Name")

  chip_paths = [device.chip_path(chip_type, index) for index in range(count)]
  chip_owners = device.get_chip_owners()

  for chip in chip_paths:
    owner = chip_owners.get(chip)
    process_name = get_process_name(owner)
    table.add_row(
        chip,
        str(owner),
        str(process_name),
    )
  return table


class TpuChipsTable:
  """Renders a table with TPU chip information."""

  def render(
      self, chip_type: Any, count: int
  ) -> console.RenderableType:
    """Creates a Rich Table with TPU chip information."""
    table = rich_table.Table(title="TPU Chips", title_justify="left")
    table.add_column("Chip")
    table.add_column("Type")
    table.add_column("Devices")
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
    return table


class TpuRuntimeUtilizationTable:
  """Renders a table with TPU runtime utilization metrics."""

  def render(self, chip_type: Any) -> console.RenderableType:
    """Creates a Rich Table or Panel for TPU runtime utilization."""
    try:
      device_usage = metrics.get_chip_usage(chip_type)
    except grpc.RpcError as e:
      if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
        exception_message = (
            "Libtpu metrics unavailable. Is there a framework using the"
            " TPU? See"
            " [link=https://github.com/google/cloud-accelerator-diagnostics/"
            "tree/main/tpu_info]tpu_info docs[/link]"
            " for more information."
        )
        return panel.Panel(
            f"[yellow]WARNING:[/yellow] {exception_message}",
            title="[b]Runtime Utilization Status[/b]",
            border_style="yellow",
        )
      else:
        exception_message = f"ERROR fetching runtime utilization: {e}"
        return panel.Panel(
            f"[red]{exception_message}[/red]",
            title="[b]Runtime Utilization Error[/b]",
            border_style="red",
        )

    table = rich_table.Table(
        title="TPU Runtime Utilization", title_justify="left"
    )
    table.add_column("Device")
    table.add_column("HBM usage")
    table.add_column("Duty cycle", justify="right")

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
    return table

