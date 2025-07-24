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

"""Argument parsing for the tpu-info tool."""

import argparse


def parse_arguments():
  """Parses command line arguments for the tpu-info tool."""
  parser = argparse.ArgumentParser(
      description="Display TPU info and metrics.",
      formatter_class=argparse.RawTextHelpFormatter,
  )
  parser.add_argument(
      "-v",
      "--version",
      action="store_true",
      help=(
          "Displays the tpu-info version, libtpu version and accelerator type."
      ),
  )
  parser.add_argument(
      "-p",
      "--process",
      action="store_true",
      help="Displays the process ID and name for each TPU chip",
  )
  parser.add_argument(
      "--streaming",
      action="store_true",
      help="Enable streaming mode to refresh metrics continuously",
  )
  parser.add_argument(
      "--rate",
      type=float,
      default=1.0,
      help=(
          "Refresh rate in seconds for streaming mode (default: 1.0; effective"
          " when streaming is implemented)."
      ),
  )
  parser.add_argument(
      "--list_metrics",
      action="store_true",
      help="List all supported metrics for metric flag.",
  )
  parser.add_argument(
      "--metric",
      nargs="+",
      help=(
          "Metric to display. Supported metrics:\n- hbm_usage: High bandwidth"
          " memory usage/ total memoery for each device.\n- duty_cycle_percent:"
          " Duty cycle percentage for each chip.\n- tensorcore_utilization: "
          " Current percentage of the Tensorcore that is utilized per chip.\n-"
          " buffer_transfer_latency: The buffer transfer latency"
          " distribution.\n- host_to_device_transfer_latency: The host to"
          " device transfer latency distribution.\n-"
          " device_to_host_transfer_latency: The device to host transfer"
          " latency distribution.\n- collective_e2e_latency: The cumulative"
          " distribution of end to end collective latency for multislice"
          " traffic."
      ),
  )
  return parser.parse_args()
