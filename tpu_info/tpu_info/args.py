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
      help="Displays the tpu cli version",
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
  return parser.parse_args()
