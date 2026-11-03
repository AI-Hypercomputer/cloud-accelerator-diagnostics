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

"""Unit tests for metric argument parsing and filter string validation."""

from absl.testing import absltest
from absl.testing import parameterized
from tpu_info import args
from tpu_info import args_helper


class MetricFilterParserTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "single_key_value",
          "key:value",
          {"key": "value"},
      ),
      (
          "with_list_value",
          "percentile:[p50, p90,p95]",
          {"percentile": ["p50", "p90", "p95"]},
      ),
      (
          "multiple_key_values",
          "key1:value1, key2:value2",
          {"key1": "value1", "key2": "value2"},
      ),
      (
          "mixed_values",
          "percentile:[p99], core_type:tensorcore",
          {"percentile": ["p99"], "core_type": "tensorcore"},
      ),
      ("extra_whitespace", " key : value ", {"key": "value"}),
      ("trailing_comma", "percentile:p50, ", {"percentile": "p50"}),
  )
  def test_parse_filter_str_success(self, filter_str, expected):
    self.assertEqual(args_helper._parse_filter_str(filter_str), expected)

  @parameterized.named_parameters(
      ("missing_colon", "key value"),
      ("empty_key", ":value"),
      ("empty_value", "key:"),
      ("unbalanced_opening_bracket", "key:[value"),
      ("unbalanced_closing_bracket", "key:value]"),
      ("multiple_opening_brackets", "key:[v1[v2]"),
      ("empty_string", ""),
      ("whitespace_only", "   "),
  )
  def test_parse_filter_str_failure(self, filter_str):
    with self.assertRaises(args_helper.MetricParsingError):
      args_helper._parse_filter_str(filter_str)

  def test_parse_metric_args_success(self):
    """Tests successful parsing and validation of metric arguments."""
    metric_args = [
        args.MetricRequest(name="hbm_usage"),
        args.MetricRequest(
            name="buffer_transfer_latency",
            filter_str="percentile:[p50, p90]",
        ),
    ]
    parsed = args_helper.MetricsParser.parse_metric_args(metric_args)
    self.assertEqual(
        parsed,
        [
            ("hbm_usage", None),
            (
                "buffer_transfer_latency",
                {"percentile": ["p50", "p90"]},
            ),
        ],
    )

  @parameterized.named_parameters(
      (
          "invalid_metric_name",
          [args.MetricRequest(name="invalid_metric")],
          "ERROR: Invalid metric",
      ),
      (
          "unsupported_filter",
          [args.MetricRequest(name="hbm_usage", filter_str="key:value")],
          "does not support filters",
      ),
      (
          "invalid_filter_key",
          [
              args.MetricRequest(
                  name="buffer_transfer_latency",
                  filter_str="invalid_key:value",
              )
          ],
          "Invalid filter key",
      ),
      (
          "malformed_filter_string",
          [
              args.MetricRequest(
                  name="buffer_transfer_latency",
                  filter_str="percentile:[p50",
              )
          ],
          "Failed to parse filter",
      ),
  )
  def test_parse_metric_args_failure(self, metric_args, expected_error):
    """Tests failing cases for parsing and validation."""
    with self.assertRaises(args_helper.MetricParsingError) as e:
      args_helper.MetricsParser.parse_metric_args(metric_args)
    self.assertIn(expected_error, str(e.exception))


if __name__ == "__main__":
  absltest.main()
