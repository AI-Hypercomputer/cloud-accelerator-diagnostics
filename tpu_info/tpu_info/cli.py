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

import datetime
import sys
import time
from typing import Any, List

from tpu_info import args
from tpu_info import cli_helper
from tpu_info import device
from tpu_info import metrics
import grpc
import rich
from rich import console
from rich import live
from rich import panel
from rich import table as rich_table
from rich import text


# The minimum refresh rate in seconds, corresponding to a max of 30 FPS.
MIN_REFRESH_RATE_SECONDS = 1.0 / 30


def _bytes_to_gib(size: int) -> float:
  return size / (1 << 30)


# TODO(vidishasethi): b/418938764 - Modularize by extracting
#  each table's rendering logic into its own dedicated helper function.
def _fetch_and_render_tables(
    chip_type: Any, count: int
) -> List[console.RenderableType]:
  """Fetches all TPU data and prepares a list of Rich Table objects for display."""
  renderables: List[console.RenderableType] = []

  table = rich_table.Table(title="TPU Chips", title_justify="left")
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

  renderables.append(table)

  table = rich_table.Table(
      title="TPU Runtime Utilization", title_justify="left"
  )
  table.add_column("Device")
  table.add_column("HBM usage")
  table.add_column("Duty cycle", justify="right")

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
    renderables.append(exception_renderable)

    device_usage = [metrics.Usage(i, -1, -1, -1) for i in range(count)]

  # TODO(wcromar): take alternative ports as a flag
  # print("Connected to libtpu at grpc://localhost:8431...")
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

  renderables.append(table)

  table = rich_table.Table(title="TensorCore Utilization", title_justify="left")
  table.add_column("Chip ID")
  table.add_column("TensorCore Utilization", justify="right")

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
    renderables.append(
        panel.Panel(
            f"[yellow]WARNING: ImportError: {e}. libtpu SDK not available.[/]",
            title="[b]TensorCore Status[/b]",
            border_style="yellow",
        )
    )
  except AttributeError as e:
    renderables.append(
        panel.Panel(
            f"[yellow]WARNING: AttributeError: {e}. Please check if the"
            " latest libtpu is used.[/]",
            title="[b]TensorCore Status[/b]",
            border_style="yellow",
        )
    )
  except RuntimeError as e:
    renderables.append(
        panel.Panel(
            f"[yellow]WARNING: RuntimeError: {e}. Please check if the latest"
            " vbar control agent is used.[/]",
            title="[b]TensorCore Status[/b]",
            border_style="yellow",
        )
    )
  else:
    for i in range(len(tensorcore_util_data)):
      tc_data = f"{tensorcore_util_data[i]}%"
      table.add_row(
          str(i),
          tc_data,
      )
    renderables.append(table)

  table = rich_table.Table(
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
    exception_message: str
    exception_renderable: panel.Panel
    if e.code() == grpc.StatusCode.UNAVAILABLE:  # pytype: disable=attribute-error
      exception_message = (
          "Buffer Transfer Latency metrics unavailable. Did you start"
          " a MULTI_SLICE workload with"
          " `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?"
      )
      renderables.append(
          panel.Panel(
              f"[yellow]WARNING:[/yellow] {exception_message}",
              title="[b]Buffer Transfer Latency Status[/b]",
              border_style="yellow",
          )
      )

    else:
      exception_message = f"ERROR fetching buffer transfer latency: {e}"
      renderables.append(
          panel.Panel(
              f"[red]{exception_message}[/red]",
              title="[b]Buffer Transfer Latency Error[/b]",
              border_style="red",
          )
      )

    buffer_transfer_latency_distributions = []

  for distribution in buffer_transfer_latency_distributions:
    table.add_row(
        distribution.buffer_size,
        f"{distribution.p50:.2f} us",
        f"{distribution.p90:.2f} us",
        f"{distribution.p95:.2f} us",
        f"{distribution.p999:.2f} us",
    )
  renderables.append(table)

  return renderables


def _get_runtime_info(rate: float) -> text.Text:
  """Returns a Rich Text with runtime info for the streaming mode."""
  current_ts = time.time()
  last_updated_time_str = datetime.datetime.fromtimestamp(current_ts).strftime(
      "%Y-%m-%d %H:%M:%S"
  )
  runtime_status = text.Text(
      f"\nRefresh rate: {rate}s\nLast update: {last_updated_time_str}",
      justify="right",
  )
  return runtime_status


def print_chip_info():
  """Print local TPU devices and libtpu runtime metrics."""
  cli_args = args.parse_arguments()
  if cli_args.version:
    print(f"tpu-info version: {cli_helper.fetch_cli_version()}")
    return
  # TODO(wcromar): Merge all of this info into one table
  chip_type, count = device.get_local_chips()
  if not chip_type:
    print("No TPU chips found.")
    return

  if cli_args.streaming:
    if cli_args.rate <= 0:
      print("Error: Refresh rate must be positive.", file=sys.stderr)
      return

   # Warn user and cap the data refresh rate if it's faster than the maximum
    # supported screen refresh rate (30 FPS).
    effective_rate = cli_args.rate
    if cli_args.rate < MIN_REFRESH_RATE_SECONDS:
      console_obj = rich.console.Console(stderr=True)
      console_obj.print(
          f"[yellow]WARNING: Provided rate {cli_args.rate:.3f}s is faster than"
          " the supported maximum. Capping at"
          f" {MIN_REFRESH_RATE_SECONDS:.3f}s.[/yellow]"
      )
      effective_rate = MIN_REFRESH_RATE_SECONDS
    print(
        f"Starting streaming mode (refresh rate: {effective_rate:.1f}s). Press"
        " Ctrl+C to exit."
    )
    # Determine a screen refresh rate that can keep up with the data refresh
    # rate.
    data_refresh_hz = 1.0 / effective_rate
    # Aim for screen updates to be slightly more frequent than data updates.
    target_screen_fps = data_refresh_hz * 1.2
    # Ensure a reasonable minimum (4 FPS) and maximum (30 FPS) screen refresh
    # rate.
    screen_refresh_per_second = min(max(4, int(target_screen_fps)), 30)
    try:
      renderables = _fetch_and_render_tables(chip_type, count)
      streaming_status = _get_runtime_info(cli_args.rate)

      if not renderables and chip_type:
        print(
            "No data tables could be generated. Exiting streaming.",
            file=sys.stderr,
        )
        return

      display = console.Group(
          streaming_status, *(renderables if renderables else [])
      )

      with live.Live(
          display,
          refresh_per_second=screen_refresh_per_second,
          screen=True,
          vertical_overflow="visible",
      ) as live_display:
        while True:
          try:
            time.sleep(effective_rate)
            new_renderables = _fetch_and_render_tables(chip_type, count)
            streaming_status = _get_runtime_info(effective_rate)
            display = console.Group(
                streaming_status, *(new_renderables if new_renderables else [])
            )
            live_display.update(display)
          except Exception as e:
            print(
                "\nFATAL ERROR during streaming update cycle, stopping stream:"
                f" {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            raise e
    except KeyboardInterrupt:
      print("\nExiting streaming mode.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          "\nAn unexpected error occurred in streaming mode :"
          f" {type(e).__name__}: {e}",
          file=sys.stderr,
      )

  else:
    renderables = _fetch_and_render_tables(chip_type, count)

    if renderables:
      console_obj = console.Console()
      for item in renderables:
        console_obj.print(item)
