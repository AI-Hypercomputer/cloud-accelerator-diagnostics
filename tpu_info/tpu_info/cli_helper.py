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
from typing import Any, Dict, List, Optional, Tuple
from tpu_info import args_helper
from tpu_info import device
from tpu_info import metrics
import grpc
from rich import align
from rich import console
from rich import panel
from rich import table as rich_table
from rich import text


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


def get_tpu_cli_info() -> console.RenderableType:
  """Returns the info of the libtpu version and accelerator type."""
  libtpu_version = fetch_libtpu_version()
  accelerator_type = fetch_accelerator_type()
  tpu_cli_info = text.Text(
      f"{'Libtpu version: '+ libtpu_version:<40}\n"
      f"{'Accelerator type: '+ accelerator_type:<40}\n",
  )
  return align.Align.right(tpu_cli_info)


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


def fetch_metric_tables(
    metric_args: List[str],
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a list of metric tables."""
  renderables: List[console.RenderableType] = []
  try:
    parsed_metrics = args_helper.MetricsParser.parse_metric_args(metric_args)
  except args_helper.MetricParsingError as e:
    exception_message = str(e)
    exception_renderable = panel.Panel(
        f"[red]{exception_message}[/red]",
        title="[b]Metric Parsing Error[/b]",
        border_style="red",
    )
    renderables.append(exception_renderable)
    return renderables

  for metric in parsed_metrics:
    renderables.extend(get_metric_table(metric, chip_type, count))
  return renderables


def get_metric_table(
    metric: Tuple[str, Dict[str, Any]], chip_type: device.TpuChip, count: int
) -> List[console.RenderableType]:
  """Returns a table with the given metric info."""
  metric_name = metric[0]
  renderables: List[console.RenderableType] = []
  transfer_latency_function = lambda: [
      TransferLatencyTables().render(metric_name)
  ]
  metric_functions = {
      "hbm_usage": lambda: get_hbm_usage_table(chip_type, count),
      "duty_cycle_percent": lambda: get_duty_cycle_table(chip_type, count),
      "tensorcore_utilization": lambda: [TensorCoreUtilizationTable().render()],
      "buffer_transfer_latency": transfer_latency_function,
      "host_to_device_transfer_latency": transfer_latency_function,
      "device_to_host_transfer_latency": transfer_latency_function,
      "collective_e2e_latency": transfer_latency_function,
  }
  renderables.extend(metric_functions[metric_name]())
  return renderables


def render_empty_table_with_columns(
    title: str, columns: List[str]
) -> rich_table.Table:
  """Renders an empty table with the given columns and title."""
  table = rich_table.Table(title=title, title_justify="left")
  for column in columns:
    table.add_column(column)
  return table


def get_hbm_usage_table(
    chip_type: device.TpuChip, count: int
) -> List[console.RenderableType]:
  """Returns a table with the HBM usage info."""
  table = render_empty_table_with_columns(
      "TPU HBM Usage", ["Device", "HBM Usage (GiB)"]
  )
  renderables: List[console.RenderableType] = []
  device_usage = get_device_usage(chip_type)

  if isinstance(device_usage, List):
    for chip in device_usage:
      memory_usage = (
          f"{_bytes_to_gib(chip.memory_usage):.2f} GiB /"
          f" {_bytes_to_gib(chip.total_memory):.2f} GiB"
      )
      table.add_row(
          str(chip.device_id),
          memory_usage,
      )
    renderables.append(table)
  else:
    # device_usage is a panel with an error message
    renderables.append(device_usage)
    for device_id in range(count):
      table.add_row(str(device_id), "N/A")
    renderables.append(table)

  return renderables


def get_duty_cycle_table(
    chip_type: device.TpuChip, count: int
) -> List[console.RenderableType]:
  """Returns a table with the duty cycle info."""
  table = render_empty_table_with_columns(
      "TPU Duty Cycle", ["Chip ID", "Duty Cycle (%)"]
  )
  renderables: List[console.RenderableType] = []
  device_usage = get_device_usage(chip_type)
  device_per_chip = chip_type.value.devices_per_chip

  if isinstance(device_usage, List):
    for chip in device_usage:
      duty_cycle_pct = f"{chip.duty_cycle_pct:.2f}%"
      if device_per_chip == 1 or chip.device_id % 2 == 0:
        table.add_row(
            str(chip.device_id // device_per_chip),
            duty_cycle_pct,
        )
    renderables.append(table)
  else:
    # device_usage is a panel with an error message
    renderables.append(device_usage)
    for device_id in range(count):
      if device_per_chip == 1 or device_id % 2 == 0:
        table.add_row(
            str(device_id // device_per_chip),
            "N/A",
        )
    renderables.append(table)
  return renderables


def get_device_usage(
    chip_type: device.TpuChip,
) -> List[metrics.Usage] | panel.Panel:
  """Returns a list of device usage metrics and exception renderable if any."""
  try:
    device_usage = metrics.get_chip_usage(chip_type)
  except grpc.RpcError as e:
    exception_message: str
    exception_renderable: panel.Panel
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      exception_message = (
          "Libtpu metrics unavailable. Is there a framework using the"
          " TPU? See"
          " [link=https://github.com/google/cloud-accelerator-diagnostics/"
          "tree/main/tpu_info]tpu_info docs[/link]"
          " for more information."
      )
      exception_renderable = panel.Panel(
          f"[yellow]WARNING:[/yellow] {exception_message}",
          title="[b]Runtime Utilization Status[/b]",
          border_style="yellow",
      )
    else:
      exception_message = f"ERROR fetching runtime utilization: {e}"
      exception_renderable = panel.Panel(
          f"[red]{exception_message}[/red]",
          title="[b]Runtime Utilization Error[/b]",
          border_style="red",
      )

    return exception_renderable
  return device_usage


class TpuChipsTable:
  """Renders a table with TPU chip information."""

  def render(self, chip_type: Any, count: int) -> console.RenderableType:
    """Creates a Rich Table with TPU chip information."""
    table = render_empty_table_with_columns(
        "TPU Chips", ["Chip", "Type", "Devices", "PID"]
    )

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

  def render(self, chip_type: Any, count: int) -> List[console.RenderableType]:
    """Creates a Rich Table or Panel for TPU runtime utilization."""
    renderables: List[console.RenderableType] = []

    table = render_empty_table_with_columns(
        "TPU Runtime Utilization", ["Device", "HBM Usage (GiB)", "Duty cycle"]
    )
    # TODO(wcromar): take alternative ports as a flag
    # print("Connected to libtpu at grpc://localhost:8431...")

    device_usage = get_device_usage(chip_type)
    devices_per_chip = chip_type.value.devices_per_chip

    if isinstance(device_usage, List):
      for chip in device_usage:
        memory_usage = (
            f"{_bytes_to_gib(chip.memory_usage):.2f} GiB /"
            f" {_bytes_to_gib(chip.total_memory):.2f} GiB"
        )
        duty_cycle_pct = f"{chip.duty_cycle_pct:.2f}%"
        table.add_row(
            str(chip.device_id),
            memory_usage,
            duty_cycle_pct
            if devices_per_chip == 1 or chip.device_id % 2 == 0
            else "",
        )
    else:
      renderables.append(device_usage)
      for device_id in range(count):
        table.add_row(
            str(device_id),
            "N/A",
            "N/A" if devices_per_chip == 1 or device_id % 2 == 0 else "",
        )

    renderables.append(table)
    return renderables


class TensorCoreUtilizationTable:
  """Renders a table with TensorCore utilization metrics."""

  def render(self) -> console.RenderableType:
    """Creates a Rich Table or Panel for TensorCore utilization."""
    try:
      # pylint: disable=g-import-not-at-top
      from libtpu import sdk  # pytype: disable=import-error

      monitoring_module = None
      if hasattr(sdk, "tpumonitoring"):
        monitoring_module = sdk.tpumonitoring
      elif hasattr(sdk, "monitoring"):
        monitoring_module = sdk.monitoring
      if monitoring_module:
        tensorcore_util_data = monitoring_module.get_metric(
            "tensorcore_util"
        ).data()
      else:
        raise AttributeError(
            "Could not find a compatible monitoring module ('tpumonitoring' or"
            " 'monitoring') in the libtpu SDK."
        )
    except ImportError as e:
      return panel.Panel(
          f"[yellow]WARNING: ImportError: {e}. libtpu SDK not available.[/]",
          title="[b]TensorCore Status[/b]",
          border_style="yellow",
      )
    except AttributeError as e:
      return panel.Panel(
          f"[yellow]WARNING: AttributeError: {e}. Please check if the"
          " latest libtpu is used.[/]",
          title="[b]TensorCore Status[/b]",
          border_style="yellow",
      )
    except RuntimeError as e:
      return panel.Panel(
          f"[yellow]WARNING: RuntimeError: {e}. Please check if the latest"
          " vbar control agent is used.[/]",
          title="[b]TensorCore Status[/b]",
          border_style="yellow",
      )

    table = render_empty_table_with_columns(
        "TensorCore Utilization", ["Chip ID", "TensorCore Utilization"]
    )

    for i, util_data in enumerate(tensorcore_util_data):
      table.add_row(
          str(i),
          f"{util_data}%",
      )
    return table


class TransferLatencyTables:
  """Renders a table with latency metrics."""

  metric_display_name_map = {
      "buffer_transfer_latency": "Buffer Transfer Latency",
      "host_to_device_transfer_latency": "Host to Device Transfer Latency",
      "device_to_host_transfer_latency": "Device to Host Transfer Latency",
      "collective_e2e_latency": "Collective End to End Latency",
  }

  def render(self, metric_arg: str) -> console.RenderableType:
    """Creates a Rich Table or Panel for buffer transfer latency."""

    metric_display_name = self.metric_display_name_map[metric_arg]
    table = render_empty_table_with_columns(
        f"TPU {metric_display_name}",
        ["Buffer Size", "P50", "P90", "P95", "P999"],
    )

    try:
      transfer_latency_distributions = metrics.get_transfer_latency(metric_arg)
    except grpc.RpcError as e:
      exception_message: str
      exception_renderable: panel.Panel
      if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
        exception_message = (
            f"{metric_display_name} metrics unavailable. Did you start"
            " a MULTI_SLICE workload with"
            " `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?"
        )
        return panel.Panel(
            f"[yellow]WARNING:[/yellow] {exception_message}",
            title=f"[b]{metric_display_name} Status[/b]",
            border_style="yellow",
        )

      else:
        exception_message = f"ERROR fetching {metric_display_name}: {e}"
        return panel.Panel(
            f"[red]{exception_message}[/red]",
            title=f"[b]{metric_display_name} Error[/b]",
            border_style="red",
        )

    for distribution in transfer_latency_distributions:
      table.add_row(
          distribution.buffer_size,
          f"{distribution.p50:.2f} us",
          f"{distribution.p90:.2f} us",
          f"{distribution.p95:.2f} us",
          f"{distribution.p999:.2f} us",
      )
    return table
