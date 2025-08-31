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
import dataclasses
from typing import Optional


@dataclasses.dataclass
class MetricRequest:
  """Represents a user's request for a single metric from the CLI.

  This class is used by the argument parser to create a link
  between a `--metric` flag and its associated filter string
  from a `-f` flag. A list of these objects is the final output of
  the `parse_arguments()` function, which is then passed on for validation
  and processing.
  """

  name: str
  filter_str: Optional[str] = None


class _MetricAndFilterAction(argparse.Action):
  """Custom action to associate --metric and --filter flags."""

  def __call__(self, parser, namespace, values, option_string=None):
    # Initialize the list on first use.
    if (
        not hasattr(namespace, self.dest)
        or getattr(namespace, self.dest) is None
    ):
      setattr(namespace, self.dest, [])

    metrics_list = getattr(namespace, self.dest)

    if option_string == "--metric":
      metrics_list.append(MetricRequest(name=values))
    elif option_string in ("-f", "--filter"):
      if not metrics_list:
        raise argparse.ArgumentError(
            self, f"{option_string} must be used after --metric."
        )
      last_metric = metrics_list[-1]
      if last_metric.filter_str is not None:
        raise argparse.ArgumentError(
            self, f"Only one filter is allowed per metric ({last_metric.name})."
        )
      last_metric.filter_str = values


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
      action=_MetricAndFilterAction,
      help=(
          "Metric to display. Can be specified multiple times. Use"
          " --list_metrics to see all supported metrics."
      ),
  )
  parser.add_argument(
      "-f",
      "--filter",
      dest="metric",
      action=_MetricAndFilterAction,
      help=(
          "An optional filter string for the preceding --metric flag. Example:"
          " -f 'percentile:[p50,p90], core_type:tensorcore'"
      ),
  )
  return parser.parse_args()
