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
from tpu_info import args_helper
from tpu_info import cli_helper
from tpu_info import device
from tpu_info import metrics
import rich
from rich import align
from rich import console
from rich import live
from rich import panel
from rich import text


# The minimum refresh rate in seconds, corresponding to a max of 30 FPS.
MIN_REFRESH_RATE_SECONDS = 1.0 / 30


def _fetch_and_render_tables(
    chip_type: Any, count: int
) -> List[console.RenderableType]:
  """Fetches all TPU data and prepares a list of Rich Table objects for display."""
  renderables: List[console.RenderableType] = []

  renderables.append(cli_helper.get_tpu_cli_info())

  # Different table for V7X due to two devices (cores) per chip.
  if chip_type is device.TpuChip.V7X:
    actual_chips = device.get_actual_chips()
    renderables.append(
        cli_helper.ActualTpuChipsTable().render(
            chip_type=chip_type,
            chip_info=actual_chips,
            core_detail=False,
        )
    )
  else:
    renderables.append(cli_helper.TpuChipsTable().render(chip_type, count))

  renderables.extend(
      cli_helper.TpuRuntimeUtilizationTable().render(chip_type, count)
  )
  # Do not render this table if the Python version is incompatible.
  if not cli_helper.is_incompatible_python_version():
    renderables.append(cli_helper.TensorCoreUtilizationTable().render(count))
  renderables.append(
      cli_helper.TransferLatencyTables().render("buffer_transfer_latency")
  )
  renderables.append(
      cli_helper.TransferLatencyTables().render("grpc_tcp_min_rtt")
  )
  renderables.append(
      cli_helper.TransferLatencyTables().render("grpc_tcp_delivery_rate")
  )
  return renderables


def _get_runtime_info(rate: float) -> align.Align:
  """Returns a Rich Text with runtime info for the streaming mode."""
  utc_time = datetime.datetime.now(datetime.timezone.utc)
  last_updated_time_str = utc_time.strftime("%Y-%m-%d %H:%M:%S %Z")
  status_text = text.Text(
      f"{'Refresh rate: '+ str(rate)+'s':<42}\n"
      f"{'Last update: ' + last_updated_time_str:<42}"
  )
  return align.Align.right(status_text)


def print_chip_info():
  """Print local TPU devices and libtpu runtime metrics."""
  cli_args = args.parse_arguments()
  console_obj = console.Console()
  is_incompatible = cli_helper.is_incompatible_python_version()

  # Gives warning but doesn't exit the program at this stage; incompatible
  # Python version will cause the program to skip rendering certain tables.
  if is_incompatible:
    console_obj.print(cli_helper.get_py_compat_warning_panel())

  if cli_args.version:
    print(f"- tpu-info version: {cli_helper.fetch_cli_version()}")
    if is_incompatible:
      print("- libtpu version: N/A (incompatible environment)")
      print("- accelerator type: N/A (incompatible environment)")
    else:
      print(f"- libtpu version: {cli_helper.fetch_libtpu_version()}")
      print(f"- accelerator type: {cli_helper.fetch_accelerator_type()}")
    return

  if cli_args.list_metrics:
    console_obj.print(
        panel.Panel(
            "\n".join(
                f"\t{metric}"
                for metric in metrics.VALID_METRICS
            ),
            title="[b]Supported Metrics[/b]",
            title_align="left",
        ),
    )
    return

  # TODO(wcromar): Merge all of this info into one table
  chip_type, count = device.get_local_chips()
  if not chip_type:
    print("No TPU chips found.")
    return

  if cli_args.process:
    table = cli_helper.fetch_process_table(chip_type, count)
    console_obj.print(table)
    return

  if cli_args.metric:
    try:
      validated_metrics = args_helper.MetricsParser.parse_metric_args(
          cli_args.metric
      )
      renderables = cli_helper.fetch_metric_tables(
          validated_metrics, chip_type, count
      )
      for item in renderables:
        console_obj.print(item)
    except args_helper.MetricParsingError as e:
      console_obj.print(
          panel.Panel(
              f"[red]{e}[/red]",
              title="[b]Metric Parsing Error[/b]",
              border_style="red",
          )
      )
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
      for item in renderables:
        console_obj.print(item)
