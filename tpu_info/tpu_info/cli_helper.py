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


def fetch_cli_version() -> str:
  """Returns the version of the current TPU CLI."""
  try:
    version = importlib.metadata.version("tpu-info")
  except importlib.metadata.PackageNotFoundError:
    version = "unknown (package metadata not found)"
  return version
