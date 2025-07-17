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

from typing import Any, Dict, List, Optional, Tuple
from tpu_info import metrics


class MetricParsingError(Exception):
  """Generic exception for all metric parsing errors."""

  def __init__(self, message: str):
    super().__init__(message)


class MetricsParser:
  """Class for parsing metric arguments."""

  @classmethod
  def parse_metric_args(
      cls,
      metric_args: List[str],
  ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """Parses metric arguments into list of tuples (metric_name, {param:val param:val})."""
    # TODO: b/430450556 - Add support for metric with parameters.
    # This function currently considers all metrics as having no parameters.
    parsed_metrics = []

    for arg in metric_args:
      if arg not in metrics.VALID_METRICS_WITH_PARAMS:
        raise MetricParsingError(
            f"ERROR: Invalid metric '{arg}'. "
            "Use '--list_metrics' to view all supported metrics."
        )
      parsed_metrics.append((arg, None))

    return parsed_metrics
