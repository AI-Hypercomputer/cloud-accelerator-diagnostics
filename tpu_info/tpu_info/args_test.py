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

"""Unit tests for command line arguments parsing and subcommand routing."""

import io
import sys
from unittest import mock
from absl.testing import absltest
from tpu_info import args


class ArgsTest(absltest.TestCase):

  def test_default_values(self):
    with mock.patch.object(sys, "argv", ["tpu_info_script_name"]):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "dashboard")
    self.assertFalse(parsed_args.version)
    self.assertEqual(parsed_args.format, "table")
    self.assertFalse(parsed_args.streaming)
    self.assertEqual(parsed_args.rate, 1.0)

  def test_version_flag_set(self):
    """Tests --version flag."""
    with mock.patch.object(sys, "argv", ["tpu_info_script_name", "--version"]):
      parsed_args = args.parse_arguments()
    self.assertTrue(
        parsed_args.version, "--version flag should set version to True."
    )

  def test_explicit_empty_argv(self):
    """Tests that an explicit empty list for argv does not fallback to sys.argv."""
    with mock.patch.object(sys, "argv", ["tpu_info_script_name", "--version"]):
      parsed_args = args.parse_arguments(argv=[])
      self.assertFalse(parsed_args.version)
      self.assertEqual(parsed_args.subcommand, "dashboard")

  def test_none_argv_defaults_to_sys_argv(self):
    """Tests that explicit argv=None defaults to sys.argv[1:]."""
    with mock.patch.object(sys, "argv", ["tpu_info_script_name", "--version"]):
      parsed_args = args.parse_arguments(argv=None)
      self.assertTrue(parsed_args.version)

  def test_legacy_process_routing(self):
    """Tests that legacy --process or -p routes to pmon subcommand."""
    with mock.patch.object(sys, "argv", ["tpu_info_script_name", "-p"]):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "pmon")

    with mock.patch.object(sys, "argv", ["tpu_info_script_name", "--process"]):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "pmon")

    with mock.patch.object(
        sys, "argv", ["tpu_info_script_name", "-p", "--format", "json"]
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "pmon")
    self.assertEqual(parsed_args.format, "json")

  def test_legacy_list_metrics_routing(self):
    """Tests that legacy --list-metrics routes to query subcommand."""
    with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
      with mock.patch.object(
          sys, "argv", ["tpu_info_script_name", "--list-metrics"]
      ):
        parsed_args = args.parse_arguments()
      self.assertEqual(parsed_args.subcommand, "query")
      self.assertTrue(parsed_args.list_metrics)
      self.assertIn(
          "WARNING: Legacy flag '--list-metrics' is deprecated.",
          mock_stderr.getvalue(),
      )

    with mock.patch.object(
        sys, "argv", ["tpu_info_script_name", "--list_metrics"]
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "query")
    self.assertTrue(parsed_args.list_metrics)

    with mock.patch.object(
        sys,
        "argv",
        ["tpu_info_script_name", "--list-metrics", "--format", "csv"],
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "query")
    self.assertTrue(parsed_args.list_metrics)
    self.assertEqual(parsed_args.format, "csv")

  def test_legacy_metric_routing(self):
    """Tests that legacy --metric routes to query subcommand."""
    with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
      argv = ["tpu_info_script_name", "--metric", "hbm_usage"]
      with mock.patch.object(sys, "argv", argv):
        parsed_args = args.parse_arguments()
      self.assertEqual(parsed_args.subcommand, "query")
      self.assertEqual(
          parsed_args.metric, [args.MetricRequest(name="hbm_usage")]
      )
      stderr_output = mock_stderr.getvalue()
      self.assertIn(
          "WARNING: Legacy metric query flags are deprecated.", stderr_output
      )
      self.assertNotIn("--list-metrics", stderr_output)

    with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
      argv_group = ["tpu_info_script_name", "-g", "runtime"]
      with mock.patch.object(sys, "argv", argv_group):
        parsed_args = args.parse_arguments()
      self.assertEqual(parsed_args.subcommand, "query")
      self.assertEqual(parsed_args.group, ["runtime"])
      stderr_output = mock_stderr.getvalue()
      self.assertIn(
          "WARNING: Legacy metric query flags are deprecated.", stderr_output
      )
      self.assertNotIn("--list-metrics", stderr_output)

  def test_format_flag(self):
    """Tests --format flag on subcommands."""
    with mock.patch.object(
        sys, "argv", ["tpu_info_script_name", "--format", "json"]
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "dashboard")
    self.assertEqual(parsed_args.format, "json")

    with mock.patch.object(
        sys,
        "argv",
        ["tpu_info_script_name", "dashboard", "--format", "csv"],
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "dashboard")
    self.assertEqual(parsed_args.format, "csv")

  def test_default_dashboard_options(self):
    """Tests that dashboard options without subcommand route to dashboard."""
    with mock.patch.object(
        sys, "argv", ["tpu_info_script_name", "--streaming"]
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "dashboard")
    self.assertTrue(parsed_args.streaming)

    with mock.patch.object(
        sys, "argv", ["tpu_info_script_name", "--rate", "2.5"]
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "dashboard")
    self.assertEqual(parsed_args.rate, 2.5)

    with mock.patch.object(
        sys, "argv", ["tpu_info_script_name", "--no-streaming"]
    ):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "dashboard")
    self.assertFalse(parsed_args.streaming)

  def test_options_preceding_subcommands(self):
    """Tests that global options preceding subcommands parse correctly."""
    test_cases = [
        (["--format", "json", "topo"], "topo", "json"),
        (["--format", "csv", "pmon"], "pmon", "csv"),
        (["--format", "json", "dashboard", "--streaming"], "dashboard", "json"),
    ]
    for argv_suffix, expected_sub, expected_fmt in test_cases:
      argv = ["tpu_info_script_name"] + argv_suffix
      with mock.patch.object(sys, "argv", argv):
        parsed_args = args.parse_arguments()
      self.assertEqual(parsed_args.subcommand, expected_sub)
      self.assertEqual(parsed_args.format, expected_fmt)

    # Preceding format flag with query and metric
    argv = [
        "tpu_info_script_name",
        "--format",
        "json",
        "query",
        "--metric",
        "hbm_usage",
    ]
    with mock.patch.object(sys, "argv", argv):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "query")
    self.assertEqual(parsed_args.format, "json")
    self.assertEqual(
        parsed_args.metric, [args.MetricRequest(name="hbm_usage")]
    )

  def test_options_preceding_legacy_flags(self):
    """Tests that global options preceding legacy flags route correctly."""
    # Preceding format flag with legacy -p
    argv_p = ["tpu_info_script_name", "--format", "json", "-p"]
    with mock.patch.object(sys, "argv", argv_p):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "pmon")
    self.assertEqual(parsed_args.format, "json")

    # Preceding format flag with legacy --metric
    argv_m = [
        "tpu_info_script_name",
        "--format",
        "json",
        "--metric",
        "hbm_usage",
    ]
    with mock.patch.object(sys, "argv", argv_m):
      parsed_args = args.parse_arguments()
    self.assertEqual(parsed_args.subcommand, "query")
    self.assertEqual(parsed_args.format, "json")
    self.assertEqual(
        parsed_args.metric, [args.MetricRequest(name="hbm_usage")]
    )

    # Preceding format flag with version
    argv_v = ["tpu_info_script_name", "--format", "json", "--version"]
    with mock.patch.object(sys, "argv", argv_v):
      parsed_args = args.parse_arguments()
    self.assertTrue(parsed_args.version)
    self.assertEqual(parsed_args.format, "json")


class ArgsMetricFilterTest(absltest.TestCase):

  def test_single_metric_no_filter(self):
    argv = ["tpu_info_script_name", "query", "--metric", "hbm_usage"]
    with mock.patch.object(sys, "argv", argv):
      parsed_args = args.parse_arguments()
    expected = [args.MetricRequest(name="hbm_usage", filter_str=None)]
    self.assertEqual(parsed_args.metric, expected)

  def test_comma_separated_metrics(self):
    argv = [
        "tpu_info_script_name",
        "query",
        "--metric",
        "hbm_usage,duty_cycle_percent",
    ]
    with mock.patch.object(sys, "argv", argv):
      parsed_args = args.parse_arguments()
    expected = [
        args.MetricRequest(name="hbm_usage", filter_str=None),
        args.MetricRequest(name="duty_cycle_percent", filter_str=None),
    ]
    self.assertEqual(parsed_args.metric, expected)

  def test_space_separated_metrics(self):
    argv = [
        "tpu_info_script_name",
        "query",
        "--metric",
        "hbm_usage",
        "duty_cycle_percent",
    ]
    with mock.patch.object(sys, "argv", argv):
      parsed_args = args.parse_arguments()
    expected = [
        args.MetricRequest(name="hbm_usage", filter_str=None),
        args.MetricRequest(name="duty_cycle_percent", filter_str=None),
    ]
    self.assertEqual(parsed_args.metric, expected)

  def test_multiple_metrics_with_filter(self):
    argv = [
        "tpu_info_script_name",
        "query",
        "--metric",
        "hbm_usage",
        "duty_cycle_percent",
        "-f",
        "percentile:[p50,p90]",
    ]
    with mock.patch.object(sys, "argv", argv):
      parsed_args = args.parse_arguments()
    # The filter applies to the last parsed metric (duty_cycle_percent)
    expected = [
        args.MetricRequest(name="hbm_usage", filter_str=None),
        args.MetricRequest(
            name="duty_cycle_percent", filter_str="percentile:[p50,p90]"
        ),
    ]
    self.assertEqual(parsed_args.metric, expected)

  def test_filter_long_form(self):
    argv = [
        "tpu_info_script_name",
        "query",
        "--metric",
        "hlo_execution_time",
        "--filter",
        "percentile:[p99]",
    ]
    with mock.patch.object(sys, "argv", argv):
      parsed_args = args.parse_arguments()
    expected = [
        args.MetricRequest(
            name="hlo_execution_time", filter_str="percentile:[p99]"
        )
    ]
    self.assertEqual(parsed_args.metric, expected)

  def test_filter_without_metric_fails(self):
    argv = ["tpu_info_script_name", "query", "-f", "some_filter"]
    with mock.patch.object(sys, "argv", argv):
      with self.assertRaises(SystemExit):
        args.parse_arguments()

  def test_multiple_filters_for_one_metric_fails(self):
    argv = [
        "tpu_info_script_name",
        "query",
        "--metric",
        "hbm_usage",
        "-f",
        "filter1",
        "-f",
        "filter2",
    ]
    with mock.patch.object(sys, "argv", argv):
      with self.assertRaises(SystemExit):
        args.parse_arguments()


class PreprocessLegacyArgsTest(absltest.TestCase):

  def test_fallback_to_dashboard(self):
    """Tests fallback to dashboard subcommand when no subcommand is specified."""
    self.assertEqual(
        args.preprocess_legacy_args(["--streaming"]),
        ["dashboard", "--streaming"],
    )
    self.assertEqual(
        args.preprocess_legacy_args(["--rate", "2.0"]),
        ["dashboard", "--rate", "2.0"],
    )
    self.assertEqual(
        args.preprocess_legacy_args(["--format", "json"]),
        ["dashboard", "--format", "json"],
    )

  def test_list_metrics_condition(self):
    """Tests that list-metrics is only triggered when the flag is present."""
    with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
      result = args.preprocess_legacy_args(["--list-metrics"])
      self.assertEqual(result, ["query", "--list-metrics"])
      self.assertIn(
          "WARNING: Legacy flag '--list-metrics'", mock_stderr.getvalue()
      )

    with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
      result = args.preprocess_legacy_args(["--list_metrics"])
      self.assertEqual(result, ["query", "--list-metrics"])
      self.assertIn(
          "WARNING: Legacy flag '--list-metrics'", mock_stderr.getvalue()
      )

    with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
      result = args.preprocess_legacy_args(["--metric", "hbm_usage"])
      self.assertEqual(result, ["query", "--metric", "hbm_usage"])
      self.assertNotIn("--list-metrics", mock_stderr.getvalue())
      self.assertIn(
          "Legacy metric query flags are deprecated", mock_stderr.getvalue()
      )


if __name__ == "__main__":
  absltest.main()
