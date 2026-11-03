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

import argparse
import contextlib
import datetime
import io
import json
import sys
from unittest import mock

from absl.testing import absltest
from tpu_info import args
from tpu_info import cli
from tpu_info import cli_helper
from tpu_info import device
from tpu_info import metrics
import grpc
import rich
from rich import console
from rich import live
from rich import panel
from rich import table as rich_table
from rich import text


Live = live.Live
DEFAULT_MOCK_CLI_ARGS = argparse.Namespace(
    version=False,
    process=False,
    streaming=False,
    rate=1.0,
    list_metrics=False,
    metric=None,
    group=None,
    subcommand="dashboard",
    format="table",
)

# Mock data for tensorcore_util metric.
mock_tc_util_data = [
    "1.00",
    "2.00",
    "3.00",
    "4.00",
]

# Mock libtpu module since it is not available in the test environment.
mock_libtpu = mock.Mock()
mock_libtpu.sdk.tpumonitoring.get_metric.return_value.data.return_value = (
    mock_tc_util_data
)
mock_libtpu.sdk.monitoring = mock_libtpu.sdk.tpumonitoring


class CliTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            cli_helper,
            "_initialize_libtpu_safely",
            return_value="OK",
            autospec=True,
        )
    )
    # Reset lazy init globals after each test in this suite
    def cleanup():
      cli_helper._libtpu_initialized = False
      cli_helper._libtpu_init_message = "Not initialized"
    self.addCleanup(cleanup)

  def _set_parser_args(self, **kwargs):
    return argparse.Namespace(**{**vars(DEFAULT_MOCK_CLI_ARGS), **kwargs})

  def test_print_version_info_compatible(self):
    """Tests the --version flag in a compatible environment."""
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(version=True),
    ), mock.patch.object(
        cli_helper,
        "fetch_cli_version",
        return_value="1.2.3",
    ), mock.patch.object(
        cli_helper,
        "fetch_libtpu_version",
        return_value="4.5.6",
    ), mock.patch.object(
        cli_helper,
        "fetch_accelerator_type",
        return_value="v4",
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("tpu-info version: 1.2.3", output)
      self.assertIn("libtpu version: 4.5.6", output)
      self.assertIn("accelerator type: v4", output)
      self.assertNotIn("Compatibility Warning", output)

  def test_print_version_info_incompatible(self):
    """Tests the --version flag in an incompatible environment."""
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(version=True),
    ), mock.patch.object(
        cli_helper,
        "fetch_cli_version",
        return_value="1.2.3",
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=True
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("Compatibility Warning", output)
      self.assertIn("tpu-info version: 1.2.3", output)
      self.assertIn("libtpu version: N/A", output)
      self.assertIn("accelerator type: N/A", output)

  def test_list_metrics(self):
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query", list_metrics=True
        ),
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("Supported Telemetry Metrics", output)
      self.assertIn("hbm_usage", output)
      self.assertIn("duty_cycle_percent", output)
      self.assertIn("tensorcore_utilization", output)
      self.assertIn("buffer_transfer_latency", output)
      self.assertIn("host_to_device_transfer_latency", output)
      self.assertIn("device_to_host_transfer_latency", output)
      self.assertIn("collective_e2e_latency", output)
      self.assertIn("runtime_hbm_utilization", output)

  def test_group_filtering(self):
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query", group=["runtime"]
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli_helper, "fetch_metric_tables"
    ) as mock_fetch:
      cli.print_chip_info()
      mock_fetch.assert_called_once()
      requested_names = [m[0] for m in mock_fetch.call_args[0][0]]
      self.assertIn("hbm_usage", requested_names)
      self.assertIn("duty_cycle_percent", requested_names)

  def test_print_process_info(self):
    """Tests the pmon subcommand output."""
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(subcommand="pmon"),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli_helper, "render_process_monitor_table"
    ) as mock_render, mock.patch(
        "rich.console.Console"
    ) as mock_console_cls, mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      mock_console_instance = mock_console_cls.return_value
      cli.print_chip_info()

      mock_render.assert_called_once()
      mock_console_instance.print.assert_called_once_with(
          mock_render.return_value
      )

  def test_print_chip_info_success_path(self):
    """Integration test for successful path of print_chip_info."""
    chip_owners = {"/dev/accel0": 123}
    chip_usage = [metrics.Usage(0, 1073741824, 34088157184, 99.9)]
    mock_sdk = mock.Mock()
    mock_sdk.tpumonitoring.get_metric.return_value.data.return_value = [
        "1.00"
    ]

    with contextlib.redirect_stdout(io.StringIO()) as f, mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 1)
    ), mock.patch.object(
        device, "get_chip_owners", return_value=chip_owners
    ), mock.patch.object(
        metrics, "get_chip_usage", return_value=chip_usage
    ), mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk,
    ), mock.patch.object(
        metrics, "get_runtime_hbm_utilization", return_value=[(0, 45.2)]
    ), mock.patch.object(
        args, "parse_arguments", return_value=DEFAULT_MOCK_CLI_ARGS
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("tpu-info Version", output)
      self.assertIn("libtpu Version", output)
      self.assertIn("TPU Device Status", output)
      self.assertIn("TPU Process Monitor", output)
      self.assertIn("1.00 GiB", output)
      self.assertIn("99.9%", output)
      self.assertIn("45.2%", output)

  def test_print_chip_info_explicit_metric(self):
    """Tests that metrics are printed when requested explicitly."""
    chip_owners = {"/dev/accel0": 123}
    chip_usage = [metrics.Usage(0, 1073741824, 34088157184, 99.9)]
    with contextlib.redirect_stdout(io.StringIO()) as f, mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 1)
    ), mock.patch.object(
        device, "get_chip_owners", return_value=chip_owners
    ), mock.patch.object(
        metrics, "get_chip_usage", return_value=chip_usage
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="runtime_hbm_utilization")]
        ),
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch.object(
        metrics, "get_runtime_hbm_utilization", return_value=[(0, 10.5)]
    ):
      cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("TPU Runtime HBM Utilization", output)
      self.assertIn("10.50%", output)
      self.assertNotIn("TPU Chips", output)
      self.assertNotIn("TPU Runtime Utilization", output)

    with mock.patch.object(
        device, "get_local_chips", return_value=(None, 0)
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=DEFAULT_MOCK_CLI_ARGS,
    ), mock.patch.object(
        args, "parse_arguments", return_value=DEFAULT_MOCK_CLI_ARGS
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f:
        cli.print_chip_info()

      output = f.getvalue()

      self.assertIn("No TPU chips found.", output)

  def test_print_chip_info_rpc_error(self):
    class FakeRpcError(grpc.RpcError):

      def code(self):
        return grpc.StatusCode.UNAVAILABLE

    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        metrics,
        "get_chip_usage",
        side_effect=FakeRpcError,
    ), mock.patch.dict(
        sys.modules, {"libtpu": mock_libtpu}
    ), mock.patch.object(
        args, "parse_arguments", return_value=DEFAULT_MOCK_CLI_ARGS
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()

      output = f.getvalue()
      self.assertIn("Telemetry Warning", output)
      self.assertIn("Libtpu metrics unavailable.", output)

  def test_print_chip_info_error(self):
    class FakeRpcError(grpc.RpcError):

      def code(self):
        return grpc.StatusCode.UNKNOWN

    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(),
    ), mock.patch.object(
        device,
        "get_local_chips",
        return_value=(device.TpuChip.V4, 4),
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        metrics,
        "get_chip_usage",
        side_effect=FakeRpcError,
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()

      output = f.getvalue()
      self.assertIn("ERROR", output)

  def test_tensorcore_fallback_to_old_libtpu(self):
    """Tests the fallback path with the old 'monitoring' module."""
    mock_sdk = mock.Mock()

    del mock_sdk.tpumonitoring
    mock_sdk.monitoring.get_metric.return_value.data.return_value = (
        mock_tc_util_data
    )

    with mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk,
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="tensorcore_utilization")]
        ),
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()

    self.assertIn("TensorCore Utilization", output)
    self.assertIn("4.00%", output)
    mock_sdk.monitoring.get_metric.assert_called_with("tensorcore_util")

  def test_print_chip_info_tensorcore_util(self):
    mock_sdk = mock.Mock()
    mock_sdk.tpumonitoring.get_metric.return_value.data.return_value = (
        mock_tc_util_data
    )
    with mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk,
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="tensorcore_utilization")]
        ),
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()

      self.assertIn("TensorCore Utilization", output)
      self.assertIn("1.00%", output)
      self.assertIn("2.00%", output)
      self.assertIn("3.00%", output)
      self.assertIn("4.00%", output)

  @mock.patch.dict(sys.modules, {"libtpu": None})
  def test_print_chip_info_tensorcore_util_import_error(self):
    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="tensorcore_utilization")]
        ),
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("ImportError", output)

  def test_print_chip_info_tensorcore_util_attribute_error(self):
    mock_sdk_attr_error = mock.Mock()
    mock_sdk_attr_error.tpumonitoring.get_metric.side_effect = AttributeError()

    with mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk_attr_error,
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="tensorcore_utilization")]
        ),
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("Please check if the latest libtpu is used", output)

    mock_sdk_runtime_error = mock.Mock()
    mock_sdk_runtime_error.tpumonitoring.get_metric.side_effect = RuntimeError()
    with mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk_runtime_error,
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="tensorcore_utilization")]
        ),
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      self.assertIn("latest vbar control agent is", output)

  def test_streaming_uses_correct_dynamic_refresh_rate(self):
    """Verifies Live is called with a dynamic, capped FPS for various rates."""
    test_cases = [
        ("slow_rate_uses_min_fps", 2.0, 4),
        ("fast_rate_is_dynamic", 0.1, 12),
        ("very_fast_rate_uses_max_fps", 0.01, 30),
    ]

    for name, rate, expected_fps in test_cases:
      with self.subTest(name=name):
        mock_live_instance = mock.Mock(spec=Live)
        mock_live_instance.__enter__ = mock.Mock(
            return_value=mock_live_instance
        )
        mock_live_instance.__exit__ = mock.Mock(return_value=None)

        with mock.patch.object(
            device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
        ), mock.patch.object(
            args,
            "parse_arguments",
            return_value=self._set_parser_args(streaming=True, rate=rate),
        ), mock.patch.object(
            cli_helper,
            "get_dashboard",
            return_value=rich.table.Table(title="Initial"),
        ), mock.patch(
            "time.sleep", side_effect=KeyboardInterrupt
        ), mock.patch.object(
            cli.live,
            "Live",
            return_value=mock_live_instance,
        ) as mock_live_constructor, mock.patch.object(
            cli_helper, "is_incompatible_python_version", return_value=False
        ):
          cli.print_chip_info()
          mock_live_constructor.assert_called_once()
          called_kwargs = mock_live_constructor.call_args[1]
          self.assertEqual(
              called_kwargs.get("refresh_per_second"), expected_fps
          )

  def test_print_chip_info_streaming_mode_runtime_status(self):
    mock_live_instance = mock.Mock(spec=live.Live)
    mock_live_instance.__enter__ = mock.Mock(return_value=mock_live_instance)
    mock_live_instance.__exit__ = mock.Mock(return_value=None)
    fake_utc_now = datetime.datetime(
        2025, 7, 23, 9, 45, 0, tzinfo=datetime.timezone.utc
    )
    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        device, "get_chips", return_value=[]
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(streaming=True, rate=1.0),
    ), mock.patch.object(
        cli.live,
        "Live",
        return_value=mock_live_instance,
    ) as mock_live_constructor, mock.patch.object(
        cli_helper.datetime,
        "datetime",
    ) as mock_dt, mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch(
        "time.sleep", side_effect=KeyboardInterrupt
    ):
      mock_dt.now.return_value = fake_utc_now
      try:
        cli.print_chip_info()
        display_group = mock_live_constructor.call_args[0][0]
        console_obj = console.Console(width=80, color_system=None)
        with console_obj.capture() as capture:
          console_obj.print(display_group)
        display_content = capture.get()
        expected_timestamp = "2025-07-23 09:45:00 UTC"
        self.assertIn(expected_timestamp, display_content)
        self.assertIn("Refresh Rate: 1.0s", display_content)
      except KeyboardInterrupt:
        pass

  def test_print_chip_info_streaming_mode(self):
    keyboard_event = KeyboardInterrupt()

    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(streaming=True, rate=1.0),
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with mock.patch.object(
          cli_helper,
          "get_dashboard",
          side_effect=keyboard_event,
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()

        output = f.getvalue()
        self.assertIn("Starting streaming mode", output)
        self.assertIn("Exiting streaming mode.", output)

  def test_print_chip_info_streaming_mode_neg_rate(self):
    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(streaming=True, rate=-1.1),
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with contextlib.redirect_stderr(io.StringIO()) as captured_stderr:
        cli.print_chip_info()
      self.assertIn(
          "Error: Refresh rate must be positive.", captured_stderr.getvalue()
      )

  def test_print_chip_info_streaming_mode_error(self):
    unexpected_error = Exception("test error")

    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(streaming=True, rate=1.0),
    ), mock.patch.object(
        cli_helper,
        "get_dashboard",
        side_effect=unexpected_error,
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with contextlib.redirect_stderr(io.StringIO()) as captured_stderr:
        with self.assertRaises(Exception):
          cli.print_chip_info()
      self.assertIn("fatal error", captured_stderr.getvalue().lower())

  def test_print_chip_info_streaming_mode_live_update(self):
    mock_live = mock.Mock(spec=live.Live)
    mock_live.__enter__ = mock.Mock(return_value=mock_live)
    mock_live.__exit__ = mock.Mock(return_value=None)

    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(streaming=True, rate=1.0),
    ), mock.patch.object(
        cli_helper,
        "get_dashboard",
        side_effect=[
            mock.Mock(name="initial_dashboard"),
            mock.Mock(name="update1_dashboard"),
            mock.Mock(name="update2_dashboard"),
            KeyboardInterrupt(),
        ],
    ), mock.patch.object(
        cli.live,
        "Live",
        return_value=mock_live,
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      try:
        cli.print_chip_info()
      except KeyboardInterrupt:
        pass

    self.assertEqual(mock_live.update.call_count, 2)

  def test_print_chip_info_metric_parsing_error(self):
    """Tests that a MetricParsingError is handled correctly."""
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="invalid_metric")]
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ), mock.patch("rich.console.Console.print") as mock_print:
      cli.print_chip_info()
      # Check that a Panel with the error message was printed.
      self.assertEqual(mock_print.call_count, 1)
      call_args, _ = mock_print.call_args
      printed_object = call_args[0]
      self.assertIsInstance(printed_object, panel.Panel)
      self.assertIn("Metric Parsing Error", str(printed_object.title))
      self.assertIn(
          "ERROR: Invalid metric",
          str(printed_object.renderable),
      )

  def test_print_chip_info_streaming_mode_live_update_error(self):
    mock_live_instance = mock.Mock(spec=live.Live)
    mock_live_instance.__enter__ = mock.Mock(return_value=mock_live_instance)
    mock_live_instance.__exit__ = mock.Mock(return_value=None)

    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(streaming=True, rate=1.0),
    ), mock.patch.object(
        cli_helper,
        "get_dashboard",
        side_effect=[
            mock.Mock(name="initial_dashboard"),
            RuntimeError("test error"),
        ],
    ), mock.patch.object(
        cli.live,
        "Live",
        return_value=mock_live_instance,
    ), mock.patch.object(
        cli_helper, "is_incompatible_python_version", return_value=False
    ):
      with contextlib.redirect_stdout(
          io.StringIO()
      ), contextlib.redirect_stderr(io.StringIO()) as captured_stderr:
        with self.assertRaises(RuntimeError):
          cli.print_chip_info()

      self.assertIn(
          "FATAL ERROR during streaming update cycle",
          captured_stderr.getvalue(),
      )

  def test_query_json_success(self):
    from tpu_info.registry import MetricRegistry  # pylint: disable=g-import-not-at-top
    mock_raw_data = {"hbm_usage": [{"device_id": 0, "usage": 123}]}
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="hbm_usage")],
            format="json",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        MetricRegistry, "get_raw_metric", return_value=mock_raw_data
    ) as mock_get_raw:
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      mock_get_raw.assert_called_once_with(
          "hbm_usage", device.TpuChip.V4, 4, None
      )
      self.assertIn('"hbm_usage":', output)
      self.assertIn("timestamp", output)

  def test_query_json_with_error(self):
    from tpu_info.registry import MetricRegistry  # pylint: disable=g-import-not-at-top
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="hbm_usage")],
            format="json",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        MetricRegistry,
        "get_raw_metric",
        side_effect=Exception("GRPC Connection failed"),
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.print_chip_info()
      stdout_output = f_stdout.getvalue()
      stderr_output = f_stderr.getvalue()

      # Verify warning is printed to stderr
      self.assertIn(
          "WARNING: Failed to fetch raw metric 'hbm_usage': GRPC Connection"
          " failed",
          stderr_output,
      )
      # Verify JSON output is still produced with error details
      self.assertIn(
          '"hbm_usage": "ERROR: GRPC Connection failed"', stdout_output
      )

  def test_query_csv(self):
    mock_table = mock.Mock(spec=rich.table.Table)
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="hbm_usage")],
            format="csv",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli_helper, "fetch_metric_tables", return_value=[mock_table]
    ), mock.patch.object(
        cli_helper, "serialize_tables_to_csv", return_value="csv,data\n"
    ) as mock_serialize:
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      mock_serialize.assert_called_once_with([mock_table])
      self.assertEqual(output, "csv,data\n")

  def test_query_csv_with_warnings(self):
    mock_table = mock.Mock(spec=rich.table.Table)
    mock_warning_panel = panel.Panel("Warning: Server Offline", title="Error")
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="hbm_usage")],
            format="csv",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli_helper,
        "fetch_metric_tables",
        return_value=[mock_table, mock_warning_panel],
    ), mock.patch.object(
        cli_helper, "serialize_tables_to_csv", return_value="csv,data\n"
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.print_chip_info()
      stdout_output = f_stdout.getvalue()
      stderr_output = f_stderr.getvalue()

      # Warning panel text printed to stderr
      self.assertIn("WARNING: Warning: Server Offline", stderr_output)
      # Tables printed to stdout
      self.assertEqual(stdout_output, "csv,data\n")

  def test_run_dashboard_streaming_json_and_csv(self):
    status_list = [{"device_id": 0}]
    with mock.patch.object(
        cli_helper,
        "fetch_consolidated_device_status",
        return_value=(status_list, "Sample Warning"),
    ), mock.patch(
        "time.sleep",
        side_effect=[None, KeyboardInterrupt, None, KeyboardInterrupt],
    ), mock.patch.object(
        cli_helper,
        "serialize_dashboard_to_json",
        return_value='{"status": "ok"}',
    ), mock.patch.object(
        cli_helper,
        "serialize_dashboard_to_csv",
        return_value="dev_id\n0\n",
    ):
      # JSON format
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.run_dashboard_streaming(
            1.0,
            device.TpuChip.V4,
            1,
            "json",
            console.Console(),
        )
      self.assertIn("Starting streaming mode", f_stderr.getvalue())
      self.assertIn("Exiting streaming mode.", f_stderr.getvalue())
      self.assertNotIn("Starting streaming mode", f_stdout.getvalue())
      self.assertIn('{"status": "ok"}', f_stdout.getvalue())
      self.assertIn("WARNING: Sample Warning", f_stderr.getvalue())

      # CSV format
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.run_dashboard_streaming(
            1.0,
            device.TpuChip.V4,
            1,
            "csv",
            console.Console(),
        )
      self.assertIn("Starting streaming mode", f_stderr.getvalue())
      self.assertIn("Exiting streaming mode.", f_stderr.getvalue())
      self.assertNotIn("Starting streaming mode", f_stdout.getvalue())
      self.assertIn("dev_id", f_stdout.getvalue())

  def test_run_dashboard_streaming_capping(self):
    with mock.patch(
        "time.sleep", side_effect=KeyboardInterrupt
    ), mock.patch.object(
        cli_helper, "get_dashboard", return_value="mock_dashboard"
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr, \
           contextlib.redirect_stdout(io.StringIO()):
        cli.run_dashboard_streaming(
            0.01,
            device.TpuChip.V4,
            1,
            "table",
            console.Console(),
        )
      self.assertIn("faster than the supported maximum", f_stderr.getvalue())

  def test_run_pmon_streaming_invalid_rate(self):
    with contextlib.redirect_stderr(io.StringIO()) as f_stderr:
      cli.run_pmon_streaming(
          0.0, device.TpuChip.V4, 1, "table", console.Console()
      )
    self.assertIn("Error: Refresh rate must be positive.", f_stderr.getvalue())

  def test_run_pmon_streaming_capping(self):
    with mock.patch(
        "time.sleep", side_effect=KeyboardInterrupt
    ), mock.patch.object(
        cli_helper, "fetch_consolidated_device_status", return_value=([], None)
    ), mock.patch.object(
        cli_helper,
        "render_process_monitor_table",
        return_value=rich_table.Table(),
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr, \
           contextlib.redirect_stdout(io.StringIO()):
        cli.run_pmon_streaming(
            0.01, device.TpuChip.V4, 1, "table", console.Console()
        )
      self.assertIn("faster than the supported maximum", f_stderr.getvalue())

  def test_run_pmon_streaming_json_and_csv(self):
    status_list = [{"device_id": 0}]
    with mock.patch.object(
        cli_helper,
        "fetch_consolidated_device_status",
        return_value=(status_list, None),
    ), mock.patch(
        "time.sleep",
        side_effect=[None, KeyboardInterrupt, None, KeyboardInterrupt],
    ), mock.patch.object(
        cli_helper,
        "serialize_pmon_to_json",
        return_value='{"pmon": "data"}',
    ), mock.patch.object(
        cli_helper,
        "serialize_pmon_to_csv",
        return_value="device_id,pid\n0,1234\n",
    ):
      # JSON format
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.run_pmon_streaming(
            1.0, device.TpuChip.V4, 1, "json", console.Console()
        )
      self.assertIn('{"pmon": "data"}', f_stdout.getvalue())
      self.assertIn("Starting streaming mode", f_stderr.getvalue())
      self.assertIn("Exiting streaming mode.", f_stderr.getvalue())
      self.assertNotIn("Starting streaming mode", f_stdout.getvalue())

      # CSV format
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.run_pmon_streaming(
            1.0, device.TpuChip.V4, 1, "csv", console.Console()
        )
      self.assertIn("device_id,pid", f_stdout.getvalue())
      self.assertIn("Starting streaming mode", f_stderr.getvalue())
      self.assertIn("Exiting streaming mode.", f_stderr.getvalue())
      self.assertNotIn("Starting streaming mode", f_stdout.getvalue())

  def test_run_pmon_streaming_table(self):
    table_mock = rich_table.Table(title="PMON")
    with mock.patch.object(
        cli_helper,
        "fetch_consolidated_device_status",
        return_value=([], None),
    ), mock.patch.object(
        cli_helper,
        "render_process_monitor_table",
        return_value=table_mock,
    ), mock.patch(
        "time.sleep", side_effect=[None, KeyboardInterrupt]
    ), mock.patch.object(
        Live, "update"
    ) as mock_update:
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout:
        cli.run_pmon_streaming(
            1.0, device.TpuChip.V4, 1, "table", console.Console()
        )
      self.assertIn("Starting streaming mode", f_stdout.getvalue())
      self.assertIn("Exiting streaming mode.", f_stdout.getvalue())
      mock_update.assert_called_once_with(table_mock)

  def test_run_pmon_streaming_dynamic_refresh_rate(self):
    """Verifies run_pmon_streaming uses dynamic FPS passed to Live."""
    test_cases = [
        ("slow_rate_uses_min_fps", 2.0, 4),
        ("medium_rate_is_dynamic", 0.2, 6),
        ("fast_rate_is_dynamic", 0.1, 12),
        ("very_fast_rate_uses_max_fps", 0.01, 30),
    ]
    for name, rate, expected_fps in test_cases:
      with self.subTest(name=name):
        mock_live_instance = mock.Mock(spec=Live)
        mock_live_instance.__enter__ = mock.Mock(
            return_value=mock_live_instance
        )
        mock_live_instance.__exit__ = mock.Mock(return_value=None)

        with mock.patch.object(
            cli_helper,
            "fetch_consolidated_device_status",
            return_value=([], None),
        ), mock.patch.object(
            cli_helper,
            "render_process_monitor_table",
            return_value=rich_table.Table(),
        ), mock.patch(
            "time.sleep", side_effect=KeyboardInterrupt
        ), mock.patch.object(
            cli.live,
            "Live",
            return_value=mock_live_instance,
        ) as mock_live_constructor:
          cli.run_pmon_streaming(
              rate, device.TpuChip.V4, 1, "table", console.Console()
          )
          mock_live_constructor.assert_called_once()
          self.assertEqual(
              mock_live_constructor.call_args[1]["refresh_per_second"],
              expected_fps,
          )

  def test_run_pmon_streaming_fatal_error(self):
    with mock.patch.object(
        cli_helper,
        "fetch_consolidated_device_status",
        side_effect=RuntimeError("Device read failed"),
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr, \
           contextlib.redirect_stdout(io.StringIO()):
        with self.assertRaises(RuntimeError):
          cli.run_pmon_streaming(
              1.0, device.TpuChip.V4, 1, "table", console.Console()
          )
      self.assertIn(
          "FATAL ERROR during streaming update cycle", f_stderr.getvalue()
      )

  def test_print_chip_info_parse_arguments_failure(self):
    with mock.patch.object(
        args, "parse_arguments", side_effect=Exception("Invalid CLI syntax")
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        with self.assertRaises(SystemExit) as cm:
          cli.print_chip_info()
        self.assertEqual(cm.exception.code, 1)
      self.assertIn(
          "Error parsing arguments: Invalid CLI syntax", f_stderr.getvalue()
      )

  def test_print_chip_info_no_chips_found(self):
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(subcommand="dashboard"),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(None, 0)
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.print_chip_info()
      self.assertIn("No TPU chips found.", f_stderr.getvalue())

  def test_print_chip_info_version_json_and_csv(self):
    with mock.patch.object(
        cli_helper, "fetch_cli_version", return_value="1.2.3"
    ), mock.patch.object(
        cli_helper, "fetch_libtpu_version", return_value="4.5.6"
    ), mock.patch.object(
        cli_helper, "fetch_accelerator_type", return_value="v4"
    ):
      # JSON Compatible
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(version=True, format="json"),
      ), mock.patch.object(
          cli_helper, "is_incompatible_python_version", return_value=False
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        data = json.loads(f.getvalue())
        self.assertEqual(data["tpu_info_version"], "1.2.3")
        self.assertEqual(data["libtpu_version"], "4.5.6")
        self.assertEqual(data["accelerator_type"], "v4")

      # JSON Incompatible
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(version=True, format="json"),
      ), mock.patch.object(
          cli_helper, "is_incompatible_python_version", return_value=True
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        data = json.loads(f.getvalue())
        self.assertEqual(data["libtpu_version"], "N/A")
        self.assertEqual(data["accelerator_type"], "N/A")

      # CSV Compatible
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(version=True, format="csv"),
      ), mock.patch.object(
          cli_helper, "is_incompatible_python_version", return_value=False
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertIn("1.2.3,4.5.6,v4", f.getvalue())

      # CSV Incompatible
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(version=True, format="csv"),
      ), mock.patch.object(
          cli_helper, "is_incompatible_python_version", return_value=True
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertIn("1.2.3,N/A,N/A", f.getvalue())

  def test_print_chip_info_query_list_metrics_json_and_csv(self):
    with mock.patch.object(
        cli_helper,
        "serialize_metrics_list_to_json",
        return_value='{"group": "metrics"}',
    ), mock.patch.object(
        cli_helper,
        "serialize_metrics_list_to_csv",
        return_value="group,name,desc\n",
    ):
      # JSON
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(
              subcommand="query", list_metrics=True, format="json"
          ),
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertEqual(f.getvalue().strip(), '{"group": "metrics"}')

      # CSV
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(
              subcommand="query", list_metrics=True, format="csv"
          ),
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertEqual(f.getvalue().strip(), "group,name,desc")

  def test_print_chip_info_dashboard_json_and_csv(self):
    status_list = [{"device_id": 0}]
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="dashboard", format="json"
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 1)
    ), mock.patch.object(
        cli_helper,
        "fetch_consolidated_device_status",
        return_value=(status_list, "RPC Issue"),
    ), mock.patch.object(
        cli_helper,
        "serialize_dashboard_to_json",
        return_value='{"dashboard": "data"}',
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.print_chip_info()
      self.assertEqual(f_stdout.getvalue().strip(), '{"dashboard": "data"}')
      self.assertIn("WARNING: RPC Issue", f_stderr.getvalue())

    # CSV
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="dashboard", format="csv"
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 1)
    ), mock.patch.object(
        cli_helper,
        "fetch_consolidated_device_status",
        return_value=(status_list, "RPC Issue"),
    ), mock.patch.object(
        cli_helper,
        "serialize_dashboard_to_csv",
        return_value="header\ndata\n",
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f_stdout, \
           contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        cli.print_chip_info()
      self.assertEqual(f_stdout.getvalue().strip(), "header\ndata")
      self.assertIn("WARNING: RPC Issue", f_stderr.getvalue())

  def test_print_chip_info_pmon_json_and_csv(self):
    status_list = [{"device_id": 0}]
    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 1)
    ), mock.patch.object(
        cli_helper,
        "fetch_consolidated_device_status",
        return_value=(status_list, None),
    ):
      # JSON
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(subcommand="pmon", format="json"),
      ), mock.patch.object(
          cli_helper,
          "serialize_pmon_to_json",
          return_value='{"pmon": "json"}',
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertEqual(f.getvalue().strip(), '{"pmon": "json"}')

      # CSV
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(subcommand="pmon", format="csv"),
      ), mock.patch.object(
          cli_helper,
          "serialize_pmon_to_csv",
          return_value="pmon,csv\n",
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertEqual(f.getvalue().strip(), "pmon,csv")

  def test_print_chip_info_query_with_group_and_invalid_group(self):
    # Valid group
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[],
            group=["runtime"],
            format="table",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli_helper, "fetch_metric_tables", return_value=[rich_table.Table()]
    ):
      with contextlib.redirect_stdout(io.StringIO()):
        cli.print_chip_info()

    # Invalid group
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[],
            group=["invalid_group_xyz"],
            format="table",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ):
      with contextlib.redirect_stdout(io.StringIO()):
        cli.print_chip_info()

  def test_print_chip_info_query_metric_parsing_error_non_table(self):
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[],
            group=["invalid_group_xyz"],
            format="json",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        with self.assertRaises(SystemExit) as cm:
          cli.print_chip_info()
        self.assertEqual(cm.exception.code, 1)
      self.assertIn("ERROR: ERROR: Invalid group", f_stderr.getvalue())

  def test_print_chip_info_query_csv_panel_warning_branches(self):
    mock_panel_rich_text = panel.Panel(text.Text("Text warning"))
    mock_panel_str = panel.Panel("Plain warning str")
    mock_panel_none = mock.MagicMock(spec=panel.Panel)
    del mock_panel_none.renderable
    mock_panel_none.__str__.return_value = "No renderable warning"

    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[args.MetricRequest(name="hbm_usage")],
            format="csv",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli_helper,
        "fetch_metric_tables",
        return_value=[mock_panel_rich_text, mock_panel_str, mock_panel_none],
    ), mock.patch.object(
        cli_helper, "serialize_tables_to_csv", return_value=""
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr, \
           contextlib.redirect_stdout(io.StringIO()):
        cli.print_chip_info()
      self.assertIn("WARNING: Text warning", f_stderr.getvalue())
      self.assertIn("WARNING: Plain warning str", f_stderr.getvalue())
      self.assertIn("WARNING: No renderable warning", f_stderr.getvalue())

  def test_print_chip_info_topo(self):
    table_mock = rich_table.Table(title="TPU Topology")
    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ):
      # Table
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(subcommand="topo", format="table"),
      ), mock.patch.object(
          cli_helper, "fetch_topo_table", return_value=table_mock
      ):
        with contextlib.redirect_stdout(io.StringIO()):
          cli.print_chip_info()

      # JSON
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(subcommand="topo", format="json"),
      ), mock.patch.object(
          cli_helper, "serialize_topo_to_json", return_value='[{"topo": 1}]'
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertEqual(f.getvalue().strip(), '[{"topo": 1}]')

      # CSV
      with mock.patch.object(
          args,
          "parse_arguments",
          return_value=self._set_parser_args(subcommand="topo", format="csv"),
      ), mock.patch.object(
          cli_helper, "serialize_topo_to_csv", return_value="pci_addr\n0\n"
      ):
        with contextlib.redirect_stdout(io.StringIO()) as f:
          cli.print_chip_info()
        self.assertEqual(f.getvalue().strip(), "pci_addr\n0")

  def test_print_chip_info_pmon_streaming(self):
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="pmon",
            streaming=True,
            rate=1.0,
            format="table",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        cli, "run_pmon_streaming"
    ) as mock_stream:
      cli.print_chip_info()
      mock_stream.assert_called_once()

  def test_query_without_metric_or_group_exits_with_error(self):
    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=None,
            group=None,
            list_metrics=False,
            format="table",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ):
      with contextlib.redirect_stderr(io.StringIO()) as f_stderr:
        with self.assertRaises(SystemExit) as cm:
          cli.print_chip_info()
        self.assertEqual(cm.exception.code, 1)
      self.assertIn(
          "ERROR: No metrics or groups specified. Use '--metric <name>',"
          " '-g/--group <group>', or '--list-metrics'.",
          f_stderr.getvalue(),
      )

  def test_query_json_multi_metric_with_filters(self):
    from tpu_info.registry import MetricRegistry  # pylint: disable=g-import-not-at-top
    mock_p50_data = {"latency": 50.0}
    mock_p90_data = {"latency": 90.0}

    def fake_get_raw(metric_name, chip_type, count, filters=None):
      del metric_name, chip_type, count
      if filters and filters.get("percentile") == "p50":
        return mock_p50_data
      if filters and filters.get("percentile") == "p90":
        return mock_p90_data
      return {"latency": 0.0}

    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[
                args.MetricRequest(
                    name="buffer_transfer_latency", filter_str="percentile:p50"
                ),
                args.MetricRequest(
                    name="buffer_transfer_latency", filter_str="percentile:p90"
                ),
            ],
            format="json",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        MetricRegistry, "get_raw_metric", side_effect=fake_get_raw
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      parsed = json.loads(output)
      self.assertIn(
          "buffer_transfer_latency[percentile=p50]", parsed["metrics"]
      )
      self.assertEqual(
          parsed["metrics"]["buffer_transfer_latency[percentile=p50]"],
          mock_p50_data,
      )
      self.assertIn(
          "buffer_transfer_latency[percentile=p90]", parsed["metrics"]
      )
      self.assertEqual(
          parsed["metrics"]["buffer_transfer_latency[percentile=p90]"],
          mock_p90_data,
      )

  def test_query_json_multi_metric_with_list_filters(self):
    from tpu_info.registry import MetricRegistry  # pylint: disable=g-import-not-at-top
    mock_dist_data = {"p50": 50.0, "p90": 90.0}

    def fake_get_raw(metric_name, chip_type, count, filters=None):
      del metric_name, chip_type, count, filters
      return mock_dist_data

    with mock.patch.object(
        args,
        "parse_arguments",
        return_value=self._set_parser_args(
            subcommand="query",
            metric=[
                args.MetricRequest(
                    name="buffer_transfer_latency",
                    filter_str="percentile:[p50,p90]",
                ),
            ],
            format="json",
        ),
    ), mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        MetricRegistry, "get_raw_metric", side_effect=fake_get_raw
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()
      output = f.getvalue()
      parsed = json.loads(output)
      expected_key = "buffer_transfer_latency[percentile=[p50,p90]]"
      self.assertIn(expected_key, parsed["metrics"])
      self.assertEqual(parsed["metrics"][expected_key], mock_dist_data)


if __name__ == "__main__":
  absltest.main()
