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

"""Helper functions for args.py."""

import re
from typing import Any, Dict, List, Optional, Tuple
from tpu_info import args
from tpu_info import metrics


class MetricParsingError(Exception):
  """Generic exception for all metric parsing errors."""

  def __init__(self, message: str):
    super().__init__(message)


def _parse_filter_str(filter_str: str) -> Dict[str, Any]:
  """Parses a raw filter string (e.g., 'key:value, list_key:[v1,v2]') into a dictionary.

  This function handles simple key-value pairs and list values enclosed in
  square brackets.

  Args:
    filter_str: The raw filter string from the command line.

  Returns:
    A dictionary representing the parsed filter.
  """
  parsed_filter = {}
  # This regex splits the string by commas, but ignores commas that are inside
  # square brackets. It uses a negative lookahead `(?!...)` to assert that a
  # comma is not followed by a sequence of non-'[' characters and then a ']'.
  pairs = re.split(r",(?![^[]*\])", filter_str)
  for pair in pairs:
    if ":" not in pair:
      raise MetricParsingError(f"Invalid filter pair: {pair}")
    key, value = [p.strip() for p in pair.split(":", 1)]
    if not key:
      raise MetricParsingError("Filter key cannot be empty.")

    # Handle list values (e.g., [p50,p90])
    if value.startswith("[") and value.endswith("]"):
      parsed_filter[key] = [v.strip() for v in value[1:-1].split(",")]
    elif value.startswith("[") and not value.endswith("]"):
      raise MetricParsingError(f"Unbalanced brackets in filter: {value}")
    else:
      parsed_filter[key] = value
  return parsed_filter


class MetricsParser:
  """Class for parsing metric arguments."""

  @classmethod
  def parse_metric_args(
      cls,
      metric_args: List[args.MetricRequest],
  ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """Parses and validates metric arguments."""
    parsed_metrics = []
    for metric in metric_args:
      if metric.name not in metrics.VALID_METRICS:
        raise MetricParsingError(
            f"ERROR: Invalid metric '{metric.name}'. "
            "Use '--list_metrics' to view all supported metrics."
        )

      if not metric.filter_str:
        parsed_metrics.append((metric.name, None))
        continue

      allowed_filters = metrics.METRIC_FILTER_SCHEMA.get(metric.name)
      if not allowed_filters:
        raise MetricParsingError(
            f"Metric '{metric.name}' does not support filters."
        )

      try:
        parsed_filter = _parse_filter_str(metric.filter_str)
      except MetricParsingError as e:
        raise MetricParsingError(
            f"Failed to parse filter for metric '{metric.name}': {e}"
        ) from e

      for key in parsed_filter:
        if key not in allowed_filters:
          raise MetricParsingError(
              f"Invalid filter key '{key}' for metric '{metric.name}'. "
              f"Allowed keys: {', '.join(allowed_filters)}"
          )

      parsed_metrics.append((metric.name, parsed_filter))

    return parsed_metrics

