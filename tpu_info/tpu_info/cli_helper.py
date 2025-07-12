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

from tpu_info import device


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
