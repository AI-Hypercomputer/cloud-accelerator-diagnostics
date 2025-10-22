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
import multiprocessing
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from tpu_info import device
from tpu_info import metrics
import grpc
from packaging import version
from rich import align
from rich import console
from rich import panel
from rich import table as rich_table
from rich import text


# Define global variables to hold the libraries (populated if checks succeed).
libtpu = None
libtpu_sdk = None


def _check_library_safety():
  """Canary function to be run in a separate process and not called directly.

  This function is used to check if the libtpu import will succeed. It is run
  in a separate process (via _initialize_libtpu_safely()) to avoid any side
  effects in the main process, avoiding the main process and program from
  exiting/crashing.

  Its only job is to attempt the import. If there is an ImportError, the process
  exists with code 1. If it segfaults, the process dies. If it succeeds, the
  process exits cleanly with code 0.
  """
  try:
    # We only need to import it to see if it crashes.
    # pylint: disable=g-import-not-at-top
    import libtpu  # pytype: disable=import-error
    from libtpu import sdk  # pytype: disable=import-error
  except ImportError:
    # Handles the case where the library isn't installed.
    # Exit with a special code so the parent knows what happened.
    sys.exit(1)


def _initialize_libtpu_safely() -> str:
  """Manages process to perform the real import in main process if check passes.

  This function is used to check if the libtpu import will succeed. It runs
  _check_library_safety() in a separate process as a check to avoid any side
  effects in the main process, avoiding the main process and program from 
  exiting/crashing. The check then informs this function whether it is safe to
  import libtpu in the main process through exit codes.

  If the check passes, this function imports libtpu and libtpu.sdk in the main
  process. If the check fails, this function returns an error message and does
  not import libtpu or libtpu.sdk (keeping the global variables None).

  Set of exit codes checked from running the canary process:
  - `0`: Check passed, attempt to import libtpu in main process.
  - `-1`: Check failed, do not import libtpu in main process.
  - `11` or equivalent `-signal.SIGSEGV`: Check failed with segfault. Proceed
    without importing libtpu.
  - `1`: ImportError in cannary process (libtpu not found). Proceed without
    importing libtpu.
  - All other exit codes: Check failed with unknown exit code. Proceed without
    importing libtpu.


  Returns:
    A string indicating the result of the check. "OK" if successful, otherwise
    an error message.
  """
  # Make sure we're modifying the global variables, not local ones.
  global libtpu, libtpu_sdk

  # Run the canary process in a separate process.
  process = multiprocessing.Process(target=_check_library_safety)
  process.start()
  process.join()
  # Check the result of the canary process.
  if process.exitcode == 0:
    # Check succeeded so now it's safe to import here.
    try:
      # These imports now populate the global variables.
      # pylint: disable=g-import-not-at-top
      import libtpu as libtpu_temp  # pytype: disable=import-error
      from libtpu import sdk as libtpu_sdk_temp  # pytype: disable=import-error
      libtpu = libtpu_temp
      libtpu_sdk = libtpu_sdk_temp
      ok_message = "OK"
      return ok_message
    except ImportError:
      # This should never happen, but if it does, give an error message.
      error_message = (
          "ERROR: Import of libtpu failed despite safety check passing."
      )
      return error_message
  elif process.exitcode == -signal.SIGSEGV:
    # This is the case where the canary process segfaulted but we'll continue
    # anyway because it's better to try to run without TPU than to error out.
    error_message = (
        "WARNING: libtpu segfaulted during canary process. TPU may not work. "
    )
    return error_message
  elif process.exitcode == 1:
    # This is the case where the library isn't installed.
    error_message = "ERROR: libtpu not found."
    return error_message

  error_message = f"Check failed with unknown exit code: {process.exitcode}."
  return error_message


libtpu_import_status_message = _initialize_libtpu_safely()
if libtpu_import_status_message != "OK":
  print(libtpu_import_status_message)


def _bytes_to_gib(size: int) -> float:
  """Converts bytes to gibibytes."""
  return size / (1 << 30)


def _get_libtpusdk_version() -> str | None:
  """Returns the version of the libtpu SDK or None if not found."""
  try:
    libtpu_version = fetch_libtpu_version()
  except ImportError:  # Assume libtpu is not installed, so no version
    return None
  return libtpu_version


def is_incompatible_python_version() -> bool:
  """Checks if the current Python version is compatible with the libtpu SDK.

  Specifically, this function checks if the current Python version is
  compatible with the libtpu SDK by comparing the Python version to the
  libtpu version.

  The following are considered incompatible:
  - libtpu is not available to be imported (not installed)
  - Python version >= 3.12 and libtpu <= 0.0.20
  - Python version != 3.11.x and libtpu == 0.0.20


  Returns:
    True if the current Python version is incompatible, False otherwise.
  """
  # If libtpu is not imported, automatically incompatible
  libtpu_version = _get_libtpusdk_version()
  if libtpu_version is None:  # pytype: disable=name-error
    return True
  else:
    try:
      current_libtpu_version = version.parse(libtpu_version)
    except version.InvalidVersion:
      return True

  # If Python version >= 3.12 & libtpu <= 0.0.20 --> incompatible
  if (
      sys.version_info >= (3, 12)
      and current_libtpu_version <= version.parse("0.0.20")
  ):
    return True
  # If python version != 3.11.x & libtpu == 0.0.20 --> incompatible
  elif (
      current_libtpu_version == version.parse("0.0.20")
      and not (sys.version_info.major == 3 and sys.version_info.minor == 11)
  ):
    return True
  else:  # Otherwise assume compatible
    return False


def get_py_compat_warning_panel() -> panel.Panel:
  """Returns a Rich Panel with a Python compatibility warning."""
  python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
  libtpusdk_version = _get_libtpusdk_version() or "missing"
  warning_text = (
      f"Some features are disabled due to an incompatibility between"
      f" the libtpu SDK (version {libtpusdk_version}) and"
      f" Python {python_version}."
      "\n\nFor full functionality, please use a different Python environment."
  )
  return panel.Panel(
      f"[yellow]{warning_text}[/yellow]",
      title="[bold yellow]Compatibility Warning[/bold yellow]",
      border_style="yellow",
  )


def fetch_cli_version() -> str:
  """Returns the version of the current TPU CLI."""
  try:
    tpu_info_version = importlib.metadata.version("tpu-info")
  except importlib.metadata.PackageNotFoundError:
    tpu_info_version = "unknown (package metadata not found)"
  return tpu_info_version


def fetch_libtpu_version() -> str:
  """Returns the version of the current libtpu."""
  try:
    if libtpu is not None:
      libtpu_version = libtpu.__version__
      return libtpu_version
    else:
      raise ImportError("libtpu not imported")
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
  return align.Align.left(tpu_cli_info)


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
    validated_metrics: List[Tuple[str, Optional[Dict[str, Any]]]],
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a list of metric tables."""
  renderables: List[console.RenderableType] = []
  for metric in validated_metrics:
    renderables.extend(get_metric_table(metric, chip_type, count))
  return renderables


def get_metric_table(
    metric: Tuple[str, Optional[Dict[str, Any]]],
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a table with the given metric info."""
  metric_name, filters = metric
  renderables: List[console.RenderableType] = []
  transfer_latency_function = lambda: [
      TransferLatencyTables().render(metric_name, filters)
  ]
  metric_functions = {
      "hbm_usage": lambda: get_hbm_usage_table(chip_type, count),
      "duty_cycle_percent": lambda: get_duty_cycle_table(chip_type, count),
      "tensorcore_utilization": lambda: [
          TensorCoreUtilizationTable().render(count)
      ],
      "buffer_transfer_latency": transfer_latency_function,
      "host_to_device_transfer_latency": transfer_latency_function,
      "device_to_host_transfer_latency": transfer_latency_function,
      "collective_e2e_latency": transfer_latency_function,
      "grpc_tcp_min_rtt": transfer_latency_function,
      "grpc_tcp_delivery_rate": transfer_latency_function,
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
    chip_type: device.TpuChip,
    count: int,
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
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a table with the duty cycle info."""
  table = render_empty_table_with_columns(
      "TPU Duty Cycle", ["Core ID", "Duty Cycle (%)"]
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
    if chip_type is device.TpuChip.V7X:
      device_usage = metrics.get_chip_usage_new(chip_type)
    else:
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


class ActualTpuChipsTable:
  """Renders a table with TPU chip information from actual devices found."""

  def get_representative_core(
      self,
      chip: device.ChipInfo,
      representative_index: int = 0,
  ) -> device.CoreInfo | None:
    """Returns the representative core for a chip."""
    if len(chip.cores) >= 1:
      # Use representative core if it exists, otherwise use first core.
      if representative_index in chip.cores:
        return chip.cores[representative_index]
      else:
        return next(iter(chip.cores.values()))
    else:  # No cores on chip
      return None

  def render(
      self,
      chip_type: device.TpuChip,
      chip_info: list[device.ChipInfo],
      core_detail: bool = False,
    ) -> console.RenderableType:
    """Creates a Rich Table with TPU chip information."""
    columns = [
        "Chip ",  # Chip will be represented by zero-indexed core on chip.
        "Type",
        "Devices",
        "PID",
    ]
    # Present core details (potentially more than one chip as their own row)
    if core_detail:
      columns.extend(["Path",])

    table = render_empty_table_with_columns(
        "TPU Chips",
        columns,
    )

    # For each chip, get the PIDs from cores on chip and add to table.
    chip_owners = device.get_chip_owners()
    for chip in chip_info:
      # Use a single core to represent the chip.
      representative_core = self.get_representative_core(chip)

      # Get the representative core's vfio path to represent the chip.
      if representative_core:
        chip_representation = representative_core.vfio_path
      else:
        chip_representation = "N/A"

      # Get PIDs from core(s) on chip
      if len(chip.cores) >= 2:
        # Likely just 1 PID, but being cautious & handling unique PIDs per core.
        owner_set = set(
            str(chip_owners.get(core.vfio_path))
            for core in chip.cores.values()
            if chip_owners.get(core.vfio_path) is not None
        )
        # If at least one core PID is not None, use a comma-separated list.
        if owner_set:
          owner_str = ", ".join(owner_set)
        else:
          owner_str = "None"
      elif len(chip.cores) == 1:
        # Assume the only core on the chip can have any index.
        if representative_core:
          owner_str = str(
              chip_owners.get(representative_core.vfio_path)
              if chip_owners.get(representative_core.vfio_path) else "N/A"
          )
        else:
          owner_str = "N/A"
      else:
        owner_str = "N/A"
      row_items = [
          chip_representation,
          str(chip_type),
          str(chip_type.value.devices_per_chip),
          owner_str,
      ]
      if core_detail:
        core_path = str([core.vfio_path for core in chip.cores.values()])
        row_items.extend([
            core_path,
        ])
      table.add_row(*row_items)

    return table


class NewTpuRuntimeUtilizationTable:
  """Renders a table with TPU runtime utilization metrics."""

  def render(self, chip_type: Any, count: int) -> List[console.RenderableType]:
    """Creates a Rich Table or Panel for TPU runtime utilization."""
    renderables: List[console.RenderableType] = []

    table = render_empty_table_with_columns(
        "TPU Runtime Utilization", ["Chip", "HBM Usage (GiB)", "Duty cycle"]
    )
    # TODO(wcromar): take alternative ports as a flag
    # print("Connected to libtpu at grpc://localhost:8431...")

    device_usage = get_device_usage(chip_type)

    if isinstance(device_usage, list):
      for chip in device_usage:
        memory_usage = (
            f"{_bytes_to_gib(chip.memory_usage):.2f} GiB /"
            f" {_bytes_to_gib(chip.total_memory):.2f} GiB"
        )
        duty_cycle_pct = f"{chip.duty_cycle_pct:.2f}%"
        table.add_row(
            str(chip.device_id),
            memory_usage,
            duty_cycle_pct,
        )
    else:
      renderables.append(device_usage)
      for device_id in range(count):
        table.add_row(
            str(device_id),
            "N/A",
            "N/A",
        )

    renderables.append(table)
    return renderables


class TpuRuntimeUtilizationTable:
  """Renders a table with TPU runtime utilization metrics."""

  def render(self, chip_type: Any, count: int) -> List[console.RenderableType]:
    """Creates a Rich Table or Panel for TPU runtime utilization."""
    renderables: List[console.RenderableType] = []

    table = render_empty_table_with_columns(
        "TPU Runtime Utilization", ["Chip", "HBM Usage (GiB)", "Duty cycle"]
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

  def render(self, count: int) -> console.RenderableType:
    """Creates a Rich Table or Panel for TensorCore utilization."""
    try:
      if libtpu_sdk is None:
        raise ImportError("libtpu.sdk not available")
      sdk = libtpu_sdk

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
      table = render_empty_table_with_columns(
          "TensorCore Utilization", ["Core ID", "TensorCore Utilization"]
      )
      for i in range(count):
        table.add_row(str(i), "N/A")
      return console.Group(
          panel.Panel(
              f"[yellow]WARNING: RuntimeError: {e}. Please check if the latest"
              " vbar control agent is used.[/]",
              title="[b]TensorCore Status[/b]",
              border_style="yellow",
          ),
          table,
      )

    table = render_empty_table_with_columns(
        "TensorCore Utilization", ["Core ID", "TensorCore Utilization"]
    )
    if tensorcore_util_data:
      for i, util_data in enumerate(tensorcore_util_data):
        table.add_row(
            str(i),
            f"{util_data}%",
        )
    else:
      for i in range(count):
        table.add_row(str(i), "N/A")

    return table


class TransferLatencyTables:
  """Renders a table with latency metrics."""

  metric_display_name_map = {
      "buffer_transfer_latency": "Buffer Transfer Latency",
      "host_to_device_transfer_latency": "Host to Device Transfer Latency",
      "device_to_host_transfer_latency": "Device to Host Transfer Latency",
      "collective_e2e_latency": "Collective End to End Latency",
      "grpc_tcp_min_rtt": "gRPC TCP Minimum RTT",
      "grpc_tcp_delivery_rate": "gRPC TCP Delivery Rate",
  }

  def render(
      self, metric_arg: str, filters: Optional[Dict[str, Any]] = None
  ) -> console.RenderableType:
    """Creates a Rich Table or Panel for buffer transfer latency."""

    metric_display_name = self.metric_display_name_map[metric_arg]
    percentiles_to_show = ["p50", "p90", "p95", "p999"]
    if filters and "percentile" in filters:
      percentiles_to_show = filters["percentile"]

    columns = []
    if metric_arg == "buffer_transfer_latency":
      columns = ["Buffer Size"]
    elif filters and "buffer_size" in filters:
      columns = ["Buffer Size"]

    columns.extend([p.upper() for p in percentiles_to_show])
    unit = "Mbps" if metric_arg == "grpc_tcp_delivery_rate" else "us"
    table = render_empty_table_with_columns(
        f"TPU {metric_display_name}",
        columns,
    )

    try:
      transfer_latency_distributions = metrics.get_transfer_latency(metric_arg)
    except grpc.RpcError as e:
      exception_message: str
      exception_renderable: panel.Panel
      if e.code() == grpc.StatusCode.UNAVAILABLE or e.code() == grpc.StatusCode.NOT_FOUND:  # pytype: disable=attribute-error
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
      row = []
      if (
          metric_arg == "buffer_transfer_latency"
          or (filters and "buffer_size" in filters)
      ):
        row = [distribution.transfer_size]
      for p in percentiles_to_show:
        row.append(f"{getattr(distribution, p):.2f} {unit}")
      table.add_row(*row)
    return table
