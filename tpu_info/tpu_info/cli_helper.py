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

import collections
import importlib.metadata
import multiprocessing
import os
import re
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tpu_info import device
from tpu_info import metrics
import grpc
from packaging import version
from rich import align
from rich import box
from rich import console
from rich import panel
from rich import table as rich_table
from rich import text

if not sys.executable:
  sys.executable = "/usr/bin/python3"


# Define global variables to hold the libraries (populated if checks succeed).
libtpu = None
libtpu_sdk = None
_libtpu_initialized = False
_libtpu_init_message = "Not initialized"


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
    # pylint: disable=g-import-not-at-top,unused-import,redefined-outer-name
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
  # Use 'spawn' context to avoid inheriting parent process's loaded library
  # state (like JAX/protobuf) which causes duplicate descriptor registration
  # crashes.
  ctx = multiprocessing.get_context("spawn")
  process = ctx.Process(target=_check_library_safety)
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


def ensure_libtpu_initialized() -> str:
  """Ensures that libtpu is safely initialized (lazy evaluation).

  Returns:
    A string indicating the result of the check ("OK" or error message).
  """
  global _libtpu_initialized, _libtpu_init_message
  if not _libtpu_initialized:
    _libtpu_init_message = _initialize_libtpu_safely()
    _libtpu_initialized = True
    if _libtpu_init_message != "OK":
      print(_libtpu_init_message, file=sys.stderr)
  return _libtpu_init_message


def _bytes_to_gib(size: int) -> float:
  """Converts bytes to gibibytes."""
  return size / (1 << 30)


def _get_libtpusdk_version() -> str | None:
  """Returns the version of the libtpu SDK or None if not found."""
  ensure_libtpu_initialized()
  try:
    libtpu_version = fetch_libtpu_version()
  except ImportError:  # Assume libtpu is not installed, so no version
    return None
  if "unknown" in libtpu_version:
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
  ensure_libtpu_initialized()
  try:
    if libtpu is not None:
      libtpu_version = libtpu.__version__
      return libtpu_version
    else:
      raise ImportError("libtpu not imported")
  except Exception:  # pylint: disable=broad-exception-caught
    try:
      return importlib.metadata.version("libtpu")
    except importlib.metadata.PackageNotFoundError:
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
  table = rich_table.Table(
      title="TPU Process Info",
      title_justify="left",
      box=box.MARKDOWN,
  )
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

  orbax_metrics = []
  pygrain_metrics = []
  other_metrics = []

  for metric in validated_metrics:
    metric_name, _ = metric
    if metric_name in metrics.ORBAX_SHORT_TO_LONG_MAP:
      orbax_metrics.append(metric_name)
    elif metric_name in metrics.PYGRAIN_SHORT_TO_LONG_MAP:
      pygrain_metrics.append(metric_name)
    else:
      other_metrics.append(metric)

  if orbax_metrics:
    renderables.extend(
        _fetch_prometheus_metrics_batch(
            orbax_metrics,
            "Orbax",
            metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
            "ORBAX_PROMETHEUS_PORT",
            "ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
        )
    )

  if pygrain_metrics:
    renderables.extend(
        _fetch_prometheus_metrics_batch(
            pygrain_metrics,
            "Pygrain",
            metrics.PYGRAIN_PROMETHEUS_DEFAULT_PORT,
            "PYGRAIN_PROMETHEUS_PORT",
            "ENABLE_PYGRAIN_PROMETHEUS_TELEMETRY=true",
        )
    )

  for metric in other_metrics:
    renderables.extend(get_metric_table(metric, chip_type, count))

  return renderables


def _fetch_prometheus_metrics_batch(
    metric_names: List[str],
    telemetry_name: str,
    default_port: int,
    port_env_var: str,
    env_var_name: str,
) -> List[console.RenderableType]:
  """Scrapes Prometheus once and renders tables for a batch of metrics."""
  port_str = os.environ.get(port_env_var)
  if port_str:
    try:
      port = int(port_str)
    except ValueError:
      port = default_port
  else:
    port = default_port

  try:
    all_metrics = metrics.scrape_prometheus(port)
  except metrics.PrometheusConnectionError:
    warning_message = (
        f"Could not connect to {telemetry_name} Prometheus server on port"
        f" {port}.\nEnsure your workload is running with"
        f" [bold]{env_var_name}[/bold] environment variable."
    )
    return [
        panel.Panel(
            warning_message,
            title=(
                f"[bold yellow]{telemetry_name} Telemetry Offline[/bold yellow]"
            ),
            border_style="yellow",
        )
    ]
  except Exception as e:  # pylint: disable=broad-exception-caught
    return [
        panel.Panel(
            f"Error fetching metrics: {e!r}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        )
    ]

  renderables = []
  missing_metrics = []
  consolidate_warnings = len(metric_names) > 1

  for metric_name in metric_names:
    tables = get_prometheus_metric_table_from_families(
        all_metrics,
        metric_name,
        telemetry_name,
        skip_if_missing=consolidate_warnings,
    )
    if not tables and consolidate_warnings:
      missing_metrics.append(metric_name)
    else:
      renderables.extend(tables)

  if missing_metrics:
    missing_list = "\n".join(f"- {m}" for m in missing_metrics)
    warning_panel = panel.Panel(
        "The following metrics were not found on the Prometheus server (they"
        f" may not have been recorded yet):\n{missing_list}",
        title=f"[bold yellow]{telemetry_name} Metrics Not Found[/bold yellow]",
        border_style="yellow",
    )
    renderables.append(warning_panel)

  return renderables


def get_metric_table(
    metric: Tuple[str, Optional[Dict[str, Any]]],
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a table with the given metric info."""
  metric_name, filters = metric
  if (
      metric_name in metrics.ORBAX_SHORT_TO_LONG_MAP
      or metric_name in metrics.PYGRAIN_SHORT_TO_LONG_MAP
  ):
    return get_prometheus_metric_table(metric_name)

  transfer_latency_function = lambda: [
      TransferLatencyTables().render(metric_name, filters)
  ]
  metric_functions = {
      "hbm_usage": lambda: get_hbm_usage_table(chip_type, count),
      "duty_cycle_percent": lambda: get_duty_cycle_table(chip_type, count),
      "tensorcore_utilization": lambda: [
          TensorCoreUtilizationTable().render(count)
      ],
      "runtime_hbm_utilization": lambda: get_runtime_hbm_utilization_table(
          chip_type, count
      ),
      "tensorcore_idle_duration": lambda: get_tensorcore_idle_duration_table(
          chip_type, count
      ),
      "hlo_queue_size": lambda: get_hlo_queue_size_table(chip_type, count),
      "hlo_exec_timing": lambda: get_hlo_exec_timing_table(chip_type, count),
      "buffer_transfer_latency": transfer_latency_function,
      "inbound_buffer_transfer_latency": transfer_latency_function,
      "host_to_device_transfer_latency": transfer_latency_function,
      "device_to_host_transfer_latency": transfer_latency_function,
      "collective_e2e_latency": transfer_latency_function,
      "host_compute_latency": transfer_latency_function,
      "grpc_tcp_min_rtt": transfer_latency_function,
      "grpc_tcp_delivery_rate": transfer_latency_function,
      "core_state": get_tpuz_core_state,
      "sequencer_state": get_tpuz_sequencer_state,
      "sequencer_state_detailed": lambda: get_tpuz_sequencer_state(
          detailed_info=True
      ),
      "queued_programs": get_tpuz_queued_programs,
  }
  return metric_functions[metric_name]()


def get_tpuz_core_state() -> List[console.RenderableType]:
  """Returns a table with the TPUz core state info."""
  data_columns = [
      "Chip ID",
      "Global Core ID",
      "Core Type",
      "xdb Server",
  ]
  table = render_empty_table_with_columns(
      title="Core States",
      columns=data_columns,
  )
  renderables: List[console.RenderableType] = []
  try:
    core_states = metrics.get_tpuz_info(include_hlo_info=False)
    for core_state in core_states:
      table.add_row(
          str(core_state.chip_id),
          str(core_state.global_core_id),
          str(core_state.core_type),
          str(core_state.xdb_server),
      )
  except grpc.RpcError as e:
    exception_message: str
    exception_renderable: panel.Panel
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      exception_message = (
          "TPUz info unavailable. Is there a framework using the"
          " TPU? See"
          " [link=https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/"
          "tree/main/tpu_info]tpu_info docs[/link]"
          " for more information."
      )
      exception_renderable = panel.Panel(
          f"[yellow]WARNING:[/yellow] {exception_message}",
          title="[b]TPUz Status[/b]",
          border_style="yellow",
      )
    else:
      exception_message = f"ERROR fetching TPUz info: {e}"
      exception_renderable = panel.Panel(
          text.Text(exception_message, style="red"),
          title="[b]TPUz Error[/b]",
          border_style="red",
      )
    renderables.append(exception_renderable)

  renderables.append(table)
  return renderables


def get_tpuz_sequencer_state(
    detailed_info: bool = False,
) -> List[console.RenderableType]:
  """Returns a table with the TPUz sequencer state info."""
  data_columns = [
      "Chip ID",
      "Global Core ID",
      "Program Counter",
      "Tracemark",
      "Program ID",
      "Run ID",
      "Sequence Type",
  ]
  title = "Sequencer States"
  if detailed_info:
    # Only consider these columns for detailed info.
    data_columns += [
        "Core Error",
        "HLO Location",
        "HLO Details",
    ]
    title = "Sequencer States (Detailed)"
  table = render_empty_table_with_columns(
      title=title,
      columns=data_columns,
  )
  renderables: List[console.RenderableType] = []
  try:
    # Only need to get HLO info if detailed info is requested since it's
    # computationally expensive.
    if detailed_info:
      core_states = metrics.get_tpuz_info(include_hlo_info=True)
    else:
      core_states = metrics.get_tpuz_info(include_hlo_info=False)

    for core_state in core_states:
      for sequencer in core_state.sequencer_states:
        data_row = [
            str(core_state.chip_id),
            str(core_state.global_core_id),
            str(sequencer.pc),
            str(sequencer.tracemark),
            str(sequencer.program_id),
            str(sequencer.run_id),
            str(sequencer.sequencer_type),
        ]
        if detailed_info:
          data_row += [
              text.Text(str(core_state.error_message)),
              text.Text(str(sequencer.hlo_location)),
              text.Text(str(sequencer.hlo_detailed_info)),
          ]
        table.add_row(*data_row)
  except grpc.RpcError as e:
    exception_message: str
    exception_renderable: panel.Panel
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      exception_message = (
          "TPUz info unavailable. Is there a framework using the"
          " TPU? See"
          " [link=https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/"
          "tree/main/tpu_info]tpu_info docs[/link]"
          " for more information."
      )
      exception_renderable = panel.Panel(
          f"[yellow]WARNING:[/yellow] {exception_message}",
          title="[b]TPUz Status[/b]",
          border_style="yellow",
      )
    else:
      exception_message = f"ERROR fetching TPUz info: {e}"
      exception_renderable = panel.Panel(
          text.Text(exception_message, style="red"),
          title="[b]TPUz Error[/b]",
          border_style="red",
      )
    renderables.append(exception_renderable)

  renderables.append(table)
  return renderables


def get_tpuz_queued_programs() -> List[console.RenderableType]:
  """Returns a table with the TPUz queued programs info."""
  data_columns = [
      "Chip ID",
      "Global Core ID",
      "Run ID",
      "Launch ID",
      "Program Fingerprint",
  ]
  table = render_empty_table_with_columns(
      title="TPUz Queued Programs",
      columns=data_columns,
  )
  renderables: List[console.RenderableType] = []
  try:
    core_states = metrics.get_tpuz_info(include_hlo_info=False)
    for core_state in core_states:
      for program in core_state.queued_programs:
        table.add_row(
            str(core_state.chip_id),
            str(core_state.global_core_id),
            str(program.run_id),
            str(program.launch_id),
            str(program.program_fingerprint),
        )
  except grpc.RpcError as e:
    exception_message: str
    exception_renderable: panel.Panel
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      exception_message = (
          "TPUz info unavailable. Is there a framework using the"
          " TPU? See"
          " [link=https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/"
          "tree/main/tpu_info]tpu_info docs[/link]"
          " for more information."
      )
      exception_renderable = panel.Panel(
          f"[yellow]WARNING:[/yellow] {exception_message}",
          title="[b]TPUz Status[/b]",
          border_style="yellow",
      )
    else:
      exception_message = f"ERROR fetching TPUz info: {e}"
      exception_renderable = panel.Panel(
          text.Text(exception_message, style="red"),
          title="[b]TPUz Error[/b]",
          border_style="red",
      )
    renderables.append(exception_renderable)

  renderables.append(table)
  return renderables


def render_empty_table_with_columns(
    title: Optional[str], columns: List[str]
) -> rich_table.Table:
  """Renders an empty table with the given columns and title."""
  min_width = len(title) + 4 if title else None
  table = rich_table.Table(
      title=title,
      title_justify="left",
      box=box.MARKDOWN,
      min_width=min_width,
  )
  for column in columns:
    table.add_column(column)
  return table


def get_hlo_queue_size_table(
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a table with the HLO queue size info."""
  table = render_empty_table_with_columns(
      "HLO Queue Size", ["Device", "Queue Size"]
  )
  renderables: List[console.RenderableType] = []

  try:
    queue_sizes = metrics.get_hlo_queue_size(chip_type)
    if queue_sizes:
      for qs in queue_sizes:
        table.add_row(str(qs.device_id), str(qs.queue_size))
    else:
      for device_id in range(count):
        table.add_row(str(device_id), "N/A")
    renderables.append(table)

  except grpc.RpcError as e:
    exception_message: str
    exception_renderable: panel.Panel
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      exception_message = (
          "HLO queue size metrics unavailable. Is there a framework using the"
          " TPU? See"
          " [link=https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/"
          "tree/main/tpu_info]tpu_info docs[/link]"
          " for more information."
      )
      exception_renderable = panel.Panel(
          f"[yellow]WARNING:[/yellow] {exception_message}",
          title="[b]HLO Queue Size Status[/b]",
          border_style="yellow",
      )
    else:
      exception_message = f"ERROR fetching HLO queue size: {e}"
      exception_renderable = panel.Panel(
          text.Text(exception_message, style="red"),
          title="[b]HLO Queue Size Error[/b]",
          border_style="red",
      )
    renderables.append(exception_renderable)

  return renderables


def get_hlo_exec_timing_table(
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a table with the HLO execution timing info."""
  table = render_empty_table_with_columns(
      "HLO Execution Timing", ["Device", "Mean", "P50", "P90", "P95", "P999"]
  )
  renderables: List[console.RenderableType] = []

  try:
    timings = metrics.get_hlo_exec_timing(chip_type)
    if timings:
      for t in timings:
        table.add_row(
            str(t.device_id),
            f"{t.mean:.2f}",
            f"{t.p50:.2f}",
            f"{t.p90:.2f}",
            f"{t.p95:.2f}",
            f"{t.p999:.2f}",
        )
    else:
      for device_id in range(count):
        table.add_row(str(device_id), "N/A", "N/A", "N/A", "N/A", "N/A")
    renderables.append(table)

  except grpc.RpcError as e:
    exception_message: str
    exception_renderable: panel.Panel
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      exception_message = (
          "HLO execution timing metrics unavailable. Is there a framework using"
          " the TPU? See"
          " [link=https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/"
          "tree/main/tpu_info]tpu_info docs[/link]"
          " for more information."
      )
      exception_renderable = panel.Panel(
          f"[yellow]WARNING:[/yellow] {exception_message}",
          title="[b]HLO Execution Timing Status[/b]",
          border_style="yellow",
      )
    else:
      exception_message = f"ERROR fetching HLO execution timing: {e}"
      exception_renderable = panel.Panel(
          text.Text(exception_message, style="red"),
          title="[b]HLO Execution Timing Error[/b]",
          border_style="red",
      )
    renderables.append(exception_renderable)

  return renderables


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


def get_runtime_hbm_utilization_table(
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a table with the runtime HBM utilization info."""
  table = render_empty_table_with_columns(
      "TPU Runtime HBM Utilization", ["Device", "Utilization (%)"]
  )
  renderables: List[console.RenderableType] = []
  try:
    bw_utils = metrics.get_runtime_hbm_utilization()
    for device_id, util in bw_utils:
      table.add_row(str(device_id), f"{util:.2f}%")
  except grpc.RpcError as e:
    exception_message = f"ERROR fetching HBM bandwidth utilization: {e}"
    exception_renderable = panel.Panel(
        text.Text(exception_message, style="red"),
        title="[b]HBM BW Util Error[/b]",
        border_style="red",
    )
    renderables.append(exception_renderable)
    for device_id in range(count):
      table.add_row(str(device_id), "N/A")

  renderables.append(table)
  return renderables


def get_tensorcore_idle_duration_table(
    chip_type: device.TpuChip,
    count: int,
) -> List[console.RenderableType]:
  """Returns a table with the TensorCore idle duration info."""
  table = render_empty_table_with_columns(
      "TPU TensorCore Idle Duration", ["Device", "Idle Duration (s)"]
  )
  renderables: List[console.RenderableType] = []
  try:
    idle_durations = metrics.get_tensorcore_idle_duration()
    for device_id, duration in idle_durations:
      table.add_row(str(device_id), f"{duration:.2f}s")
  except grpc.RpcError as e:
    exception_message = f"ERROR fetching TensorCore idle duration: {e}"
    exception_renderable = panel.Panel(
        text.Text(exception_message, style="red"),
        title="[b]TensorCore Idle Duration Error[/b]",
        border_style="red",
    )
    renderables.append(exception_renderable)
    for device_id in range(count):
      table.add_row(str(device_id), "N/A")

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
          " [link=https://github.com/AI-Hypercomputer/cloud-accelerator-diagnostics/"
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
          text.Text(exception_message, style="red"),
          title="[b]Runtime Utilization Error[/b]",
          border_style="red",
      )

    return exception_renderable
  return device_usage


class TpuChipsTable:
  """Renders a table with TPU chip information from devices found."""

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


class TensorCoreUtilizationTable:
  """Renders a table with TensorCore utilization metrics."""

  def render(self, count: int) -> console.RenderableType:
    """Creates a Rich Table or Panel for TensorCore utilization."""
    ensure_libtpu_initialized()
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
      "inbound_buffer_transfer_latency": "Inbound Buffer Transfer Latency",
      "host_to_device_transfer_latency": "Host to Device Transfer Latency",
      "device_to_host_transfer_latency": "Device to Host Transfer Latency",
      "collective_e2e_latency": "Collective End to End Latency",
      "host_compute_latency": "Host Compute Latency",
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
    if metric_arg in (
        "buffer_transfer_latency",
        "inbound_buffer_transfer_latency",
        "host_compute_latency",
    ):
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
            text.Text(exception_message, style="red"),
            title=f"[b]{metric_display_name} Error[/b]",
            border_style="red",
        )

    for distribution in transfer_latency_distributions:
      row = []
      if metric_arg in (
          "buffer_transfer_latency",
          "inbound_buffer_transfer_latency",
          "host_compute_latency",
      ) or (filters and "buffer_size" in filters):
        row = [distribution.transfer_size]
      for p in percentiles_to_show:
        row.append(f"{getattr(distribution, p):.2f} {unit}")
      table.add_row(*row)
    return table


def calculate_percentile(
    buckets: Sequence[Tuple[float, float]], percentile: float
) -> float:
  """Calculates percentile from cumulative histogram buckets.

  Args:
    buckets: A sequence of (upper_bound, cumulative_count) tuples, sorted by
      upper_bound.
    percentile: The percentile to calculate (between 0.0 and 1.0).

  Returns:
    The estimated value at the given percentile.
  """
  if not buckets:
    return 0.0
  _, total_count = buckets[-1]
  if total_count == 0:
    return 0.0

  target_count = total_count * percentile

  prev_upper_bound = 0.0
  prev_count = 0.0
  for upper_bound, count in buckets:
    if count >= target_count:
      if upper_bound == float("inf"):
        return prev_upper_bound
      if count == prev_count:
        return upper_bound
      # Linear interpolation
      ratio = (target_count - prev_count) / (count - prev_count)
      return prev_upper_bound + ratio * (upper_bound - prev_upper_bound)
    prev_upper_bound = upper_bound
    prev_count = count
  return buckets[-1][0]


def get_unit_from_name(name: str) -> str:
  """Infers the unit of measurement from the metric name.

  Args:
    name: The name of the metric.

  Returns:
    A string representing the unit (e.g., 'GB/s', 'GB', 'B', 's', 'ns'),
    or an empty string if the unit cannot be inferred.
  """
  name = name.lower()
  if re.search(r"(_|^)(gbytes_per_sec|gb_per_sec|gb_s)(_|$)", name):
    return "GiB/s"
  if re.search(r"(_|^)(gbytes|gb)(_|$)", name):
    return "GiB"
  if re.search(r"(_|^)bytes(_|$)", name):
    return "B"
  if re.search(
      r"(_|^)(duration_secs|duration_sec|seconds|secs|sec)(_|$)", name
  ):
    return "s"
  if re.search(r"(_|^)(duration_ns|ns)(_|$)", name):
    return "ns"
  return ""


INTEGER_METRICS = frozenset({
    "pygrain_dataset_prefetch_buffer_ready_count",
    "orbax_write_start_count",
    "orbax_write_async_start_count",
    "orbax_write_success_count",
    "orbax_write_preemption_count",
    "orbax_write_storage_type",
    "orbax_read_storage_type",
    "checkpoint_write_async_commit_future_count",
    "checkpoint_write_old_steps_examined_count",
    "orbax_read_start_count",
    "orbax_read_async_start_count",
    "pygrain_dataloader_iterator_get_next_duration",
})


def should_format_as_integer(name: str) -> bool:
  """Determines if a metric should be formatted as a clean integer.

  Args:
    name: The name of the metric.

  Returns:
    True if the metric should be formatted as an integer, False otherwise.
  """
  name = name.lower()
  return any(
      name == metric or name.endswith(metric) for metric in INTEGER_METRICS
  )


def format_metric_title(name: str) -> str:
  """Formats a short metric name to a human-readable table title.

  Args:
    name: The short metric name (e.g. 'pygrain_dataset_next_duration').

  Returns:
    A formatted title string (e.g. 'Pygrain Dataset Next Duration').
  """
  return " ".join(word.capitalize() for word in name.split("_"))


def render_single_prometheus_scalar(
    family: metrics.Metric, title: Optional[str], schema_key: str
) -> rich_table.Table:
  """Renders a single Prometheus scalar metric (no Metric column).

  Args:
    family: The Prometheus Metric family to render.
    title: The title of the rendered table.
    schema_key: The short name of the metric to look up extra columns.

  Returns:
    A rich Table object representing the metric.
  """
  extra_cols = metrics.METRIC_COLUMNS.get(schema_key, [])
  value_header = metrics.METRIC_VALUE_HEADERS.get(schema_key, "Value")
  columns = extra_cols + [value_header]
  table = render_empty_table_with_columns(title, columns)

  unit = get_unit_from_name(family.name)

  for sample in family.samples:
    if should_format_as_integer(schema_key):
      value_str = f"{int(round(sample.value))} {unit}".strip()
    elif sample.value.is_integer():
      value_str = f"{int(sample.value)} {unit}".strip()
    else:
      value_str = f"{sample.value:.2f} {unit}".strip()
    row = []
    labels = dict(sample.labels)
    for col in extra_cols:
      label_key = metrics.COLUMN_TO_LABEL_MAP.get(col)
      if label_key:
        row.append(labels.get(label_key, "N/A"))
      else:
        row.append("N/A")
    row.append(value_str)

    table.add_row(*row)

  return table


def render_single_prometheus_histogram(
    family: metrics.Metric, title: Optional[str], schema_key: str
) -> rich_table.Table:
  """Renders a single Prometheus histogram metric (no Metric column).

  Args:
    family: The Prometheus Metric family to render.
    title: The title of the rendered table.
    schema_key: The short name of the metric to look up extra columns.

  Returns:
    A rich Table object representing the metric with percentiles.
  """
  extra_cols = metrics.METRIC_COLUMNS.get(schema_key, [])
  columns = extra_cols + ["P5", "P50", "P95", "P99"]
  table = render_empty_table_with_columns(title, columns)

  unit = get_unit_from_name(family.name)
  samples_by_label_key = collections.defaultdict(list)

  for sample in family.samples:
    if sample.name.endswith("_bucket"):
      labels = dict(sample.labels)
      le_str = labels.pop("le", None)
      if le_str is None:
        continue
      upper_bound = float("inf") if le_str == "+Inf" else float(le_str)
      label_key = tuple(sorted(labels.items()))
      samples_by_label_key[label_key].append((upper_bound, sample.value))

  for label_key, buckets in samples_by_label_key.items():
    buckets.sort(key=lambda x: x[0])
    p5 = calculate_percentile(buckets, 0.05)
    p50 = calculate_percentile(buckets, 0.5)
    p95 = calculate_percentile(buckets, 0.95)
    p99 = calculate_percentile(buckets, 0.99)

    if should_format_as_integer(schema_key):
      p5_str = f"{int(round(p5))} {unit}".strip()
      p50_str = f"{int(round(p50))} {unit}".strip()
      p95_str = f"{int(round(p95))} {unit}".strip()
      p99_str = f"{int(round(p99))} {unit}".strip()
    else:
      p5_str = f"{p5:.2f} {unit}".strip()
      p50_str = f"{p50:.2f} {unit}".strip()
      p95_str = f"{p95:.2f} {unit}".strip()
      p99_str = f"{p99:.2f} {unit}".strip()

    row = []
    labels = dict(label_key)
    for col in extra_cols:
      label_key_name = metrics.COLUMN_TO_LABEL_MAP.get(col)
      if label_key_name:
        row.append(labels.get(label_key_name, "N/A"))
      else:
        row.append("N/A")
    row.extend([p5_str, p50_str, p95_str, p99_str])

    table.add_row(*row)

  return table


def get_prometheus_metric_table_from_families(
    all_metrics: Sequence[metrics.Metric],
    metric_name: str,
    telemetry_name: str,
    *,
    skip_if_missing: bool = True,
) -> List[console.RenderableType]:
  """Renders a single metric table from a pre-scraped list of families.

  Args:
    all_metrics: A sequence of pre-scraped Prometheus Metric families.
    metric_name: The short name of the metric (e.g.
      'pygrain_dataset_next_duration').
    telemetry_name: The name of the telemetry source (e.g. 'Orbax', 'Pygrain').
    skip_if_missing: If True, returns an empty list if the metric is not found
      in all_metrics. If False, returns a Panel with a warning.

  Returns:
    A list of rich RenderableType objects containing the rendered tables or
    panels.
  """
  long_name = metrics.ORBAX_SHORT_TO_LONG_MAP.get(
      metric_name
  ) or metrics.PYGRAIN_SHORT_TO_LONG_MAP.get(metric_name)
  if long_name is None:
    return [panel.Panel(f"Unknown metric: {metric_name}", border_style="red")]

  prom_name = long_name.strip("/").replace("/", "_")

  target_family = next(
      (family for family in all_metrics if family.name == prom_name), None
  )

  if target_family is None:
    if skip_if_missing:
      return []
    else:
      return [
          panel.Panel(
              f"Metric {metric_name} not found on Prometheus server.",
              title=f"[bold yellow]{telemetry_name} Metrics[/bold yellow]",
              border_style="yellow",
          )
      ]

  title = format_metric_title(metric_name)
  title_renderable = text.Text(title, style="table.title")

  if target_family.type in ("counter", "gauge"):
    table = render_single_prometheus_scalar(
        target_family, None, schema_key=metric_name
    )
  elif target_family.type == "histogram":
    table = render_single_prometheus_histogram(
        target_family, None, schema_key=metric_name
    )
  else:
    return [
        panel.Panel(
            f"Unsupported metric type: {target_family.type}",
            border_style="red",
        )
    ]

  return [title_renderable, table]


def get_prometheus_metric_table(
    metric_name: str,
) -> List[console.RenderableType]:
  """Returns tables with a single Prometheus metric.

  Args:
    metric_name: The name of the metric to display (e.g.,
      'pygrain_dataset_next_duration').

  Returns:
    A list of rich RenderableType objects representing the metric tables.
  """
  if metric_name in metrics.ORBAX_SHORT_TO_LONG_MAP:
    telemetry_name = "Orbax"
    env_var_name = "ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true"
    port_env_var = "ORBAX_PROMETHEUS_PORT"
    default_port = metrics.ORBAX_PROMETHEUS_DEFAULT_PORT
  elif metric_name in metrics.PYGRAIN_SHORT_TO_LONG_MAP:
    telemetry_name = "Pygrain"
    env_var_name = "ENABLE_PYGRAIN_PROMETHEUS_TELEMETRY=true"
    port_env_var = "PYGRAIN_PROMETHEUS_PORT"
    default_port = metrics.PYGRAIN_PROMETHEUS_DEFAULT_PORT
  else:
    return [panel.Panel(f"Unknown metric: {metric_name}", border_style="red")]

  port_str = os.environ.get(port_env_var)
  port = default_port
  if port_str:
    try:
      port = int(port_str)
    except ValueError:
      pass

  try:
    all_metrics = metrics.scrape_prometheus(port)
  except metrics.PrometheusConnectionError:
    warning_message = (
        f"Could not connect to {telemetry_name} Prometheus server on port"
        f" {port}.\nEnsure your workload is running with"
        f" [bold]{env_var_name}[/bold] environment variable."
    )
    return [
        panel.Panel(
            warning_message,
            title=(
                f"[bold yellow]{telemetry_name} Telemetry Offline[/bold yellow]"
            ),
            border_style="yellow",
        )
    ]
  except Exception as e:  # pylint: disable=broad-exception-caught
    return [
        panel.Panel(
            f"Error fetching metrics: {e!r}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        )
    ]

  return get_prometheus_metric_table_from_families(
      all_metrics, metric_name, telemetry_name, skip_if_missing=False
  )
