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

"""Unit tests for cli_helper utilities, rendering, and serialization."""

import collections
import importlib.metadata
import io
import json
import multiprocessing
import os
import signal
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tpu_info import cli_helper
from tpu_info import device
from tpu_info import metrics
from tpu_info import registry
import grpc
from prometheus_client.core import Metric
from prometheus_client.core import Sample
from rich import console
from rich import panel
from rich import table as rich_table
from rich.text import Text


class FakeRpcError(grpc.RpcError):
  """Helper class for faking gRPC errors."""

  def __init__(self, code):
    self._code = code

  def code(self):
    return self._code


class LibtpuSafetyTest(parameterized.TestCase):
  """Tests for the libtpu safe initialization logic."""

  def tearDown(self):
    """Ensures a clean state between tests."""
    super().tearDown()
    # Reset global variables modified by the function especially with modules
    cli_helper.libtpu = None
    cli_helper.libtpu_sdk = None
    # Reset lazy initialization state
    cli_helper._libtpu_initialized = False
    cli_helper._libtpu_init_message = "Not initialized"
    # Clean up any mocked modules from the system
    for module in ["libtpu", "libtpu.sdk"]:
      if module in sys.modules:
        del sys.modules[module]

  @mock.patch("sys.exit")
  def test_check_library_safety_success(self, mock_exit):
    """Tests the success path where the import works and exit is NOT called."""
    # --- Arrange ---
    # To make the 'import' succeed, we add a fake module to sys.modules.
    sys.modules["libtpu"] = mock.MagicMock()

    # --- Act ---
    cli_helper._check_library_safety()

    # --- Assert ---
    # The most important check: assert that exit() was never called.
    mock_exit.assert_not_called()

  @mock.patch("sys.exit")
  def test_check_library_safety_import_error(self, mock_exit):
    """Tests that exit(1) is called when the library can't be imported."""
    # --- Arrange ---
    # Setting a module to None in sys.modules makes `import` raise ImportError
    # even when libtpu is installed in the environment (e.g. on a TPU VM).
    self.enter_context(
        mock.patch.dict(sys.modules, {"libtpu": None, "libtpu.sdk": None})
    )

    # --- Act ---
    cli_helper._check_library_safety()

    # --- Assert ---
    # Verify that our mocked exit function was called exactly once.
    mock_exit.assert_called_once()
    # Verify it was called with the specific exit code for ImportError.
    mock_exit.assert_called_once_with(1)

  @mock.patch.object(multiprocessing, "get_context", autospec=True)
  def test_initialize_libtpu_safely_success(self, mock_get_context):
    """Tests the successful import and initialization of libtpu."""
    # Configure the mock context and process to simulate a successful check
    # (exitcode 0).
    mock_context = mock.MagicMock()
    mock_get_context.return_value = mock_context
    mock_process_instance = mock_context.Process.return_value
    mock_process_instance.exitcode = 0
    # Create fake libtpu modules and add them to sys.modules
    # This allows the 'import libtpu' in the main process to succeed
    mock_libtpu = mock.MagicMock()
    mock_sdk = mock.MagicMock()
    sys.modules["libtpu"] = mock_libtpu
    sys.modules["libtpu.sdk"] = mock_sdk

    result = cli_helper._initialize_libtpu_safely()

    # Message output
    self.assertEqual(result, "OK")
    # Verify the global variables were set correctly (only needed for libtpu)
    self.assertIs(cli_helper.libtpu, mock_libtpu)
    # Verify the canary process was set up correctly
    mock_get_context.assert_called_once_with("spawn")
    mock_context.Process.assert_called_once_with(
        target=cli_helper._check_library_safety
    )
    mock_process_instance.start.assert_called_once()
    mock_process_instance.join.assert_called_once()

  @mock.patch.object(multiprocessing, "get_context", autospec=True)
  def test_initialize_libtpu_safely_segfault(self, mock_get_context):
    """Tests the case where the canary process segfaults."""
    # Configure the mock context and process to simulate a segfault.
    mock_context = mock.MagicMock()
    mock_get_context.return_value = mock_context
    mock_process_instance = mock_context.Process.return_value
    mock_process_instance.exitcode = -signal.SIGSEGV

    result = cli_helper._initialize_libtpu_safely()

    # Message output
    self.assertIn("WARNING: libtpu segfaulted", result)
    # Verify the global variables were NOT set
    self.assertIsNone(cli_helper.libtpu)
    self.assertIsNone(cli_helper.libtpu_sdk)

  @mock.patch.object(multiprocessing, "get_context", autospec=True)
  def test_initialize_libtpu_safely_import_error_in_canary(
      self, mock_get_context
  ):
    """Tests when libtpu is not found in the canary process."""
    # Configure the mock context and process to simulate an ImportError
    # (exitcode 1).

    mock_context = mock.MagicMock()
    mock_get_context.return_value = mock_context
    mock_process_instance = mock_context.Process.return_value
    mock_process_instance.exitcode = 1

    result = cli_helper._initialize_libtpu_safely()

    # Message output
    self.assertEqual(result, "ERROR: libtpu not found.")
    # Verify the global variables were NOT set
    self.assertIsNone(cli_helper.libtpu)
    self.assertIsNone(cli_helper.libtpu_sdk)

  @mock.patch.object(multiprocessing, "get_context", autospec=True)
  def test_initialize_libtpu_safely_unknown_exit_code(self, mock_get_context):
    """Tests an unknown exit code from the canary process."""
    # Configure the mock context and process with an arbitrary, unexpected
    # exit code.
    mock_context = mock.MagicMock()
    mock_get_context.return_value = mock_context
    mock_process_instance = mock_context.Process.return_value
    mock_process_instance.exitcode = 123

    result = cli_helper._initialize_libtpu_safely()

    # Message output
    self.assertEqual(result, "Check failed with unknown exit code: 123.")
    # Verify the global variables were NOT set
    self.assertIsNone(cli_helper.libtpu)
    self.assertIsNone(cli_helper.libtpu_sdk)

  @mock.patch.object(multiprocessing, "get_context", autospec=True)
  def test_initialize_libtpu_safely_failsafe_import_error(
      self, mock_get_context
  ):
    """Tests the 'should never happen' case where the main import fails."""
    # Configure the mock context and process to simulate a successful check.
    mock_context = mock.MagicMock()
    mock_get_context.return_value = mock_context
    mock_process_instance = mock_context.Process.return_value
    mock_process_instance.exitcode = 0
    # Setting the modules to None in sys.modules makes 'import libtpu' in the
    # main process raise ImportError even when libtpu is installed.
    self.enter_context(
        mock.patch.dict(sys.modules, {"libtpu": None, "libtpu.sdk": None})
    )

    result = cli_helper._initialize_libtpu_safely()

    # Message output
    self.assertEqual(
        result, "ERROR: Import of libtpu failed despite safety check passing."
    )
    # Verify the global variables were NOT set
    self.assertIsNone(cli_helper.libtpu)
    self.assertIsNone(cli_helper.libtpu_sdk)

  def test_ensure_libtpu_initialized_caching(self):
    """Tests that ensure_libtpu_initialized caches the result and only runs once."""
    # Reset initialization state for test.
    cli_helper._libtpu_initialized = False
    cli_helper._libtpu_init_message = "Not initialized"

    with mock.patch.object(
        cli_helper, "_initialize_libtpu_safely", return_value="MOCK_OK"
    ) as mock_init:
      # First call should trigger initialization
      res1 = cli_helper.ensure_libtpu_initialized()
      self.assertEqual(res1, "MOCK_OK")
      mock_init.assert_called_once()

      # Reset mock to verify it's not called again
      mock_init.reset_mock()

      # Second call should return cached result without calling initializer
      res2 = cli_helper.ensure_libtpu_initialized()
      self.assertEqual(res2, "MOCK_OK")
      mock_init.assert_not_called()


class CliHelpersTest(parameterized.TestCase):

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

  def test_fetch_cli_version_success(self):
    with mock.patch.object(
        importlib.metadata,
        "version",
        return_value="1.2.3",
    ):
      self.assertEqual(cli_helper.fetch_cli_version(), "1.2.3")

  def test_bytes_to_gib(self):
    """Tests the _bytes_to_gib conversion."""
    one_gib_in_bytes = 1073741824
    self.assertAlmostEqual(cli_helper._bytes_to_gib(one_gib_in_bytes), 1.0)

    self.assertAlmostEqual(cli_helper._bytes_to_gib(0), 0.0)

    half_gib_in_bytes = 536870912
    self.assertAlmostEqual(cli_helper._bytes_to_gib(half_gib_in_bytes), 0.5)

  def test_fetch_cli_version_error(self):
    page_not_found_error = importlib.metadata.PackageNotFoundError
    with mock.patch.object(
        importlib.metadata,
        "version",
        side_effect=page_not_found_error,
    ):
      self.assertIn(
          "package metadata not found", cli_helper.fetch_cli_version()
      )

  def test_fetch_libtpu_version_from_metadata_success(self):
    with mock.patch.object(
        importlib.metadata,
        "version",
        return_value="1.2.3",
    ):
      self.assertEqual(cli_helper.fetch_libtpu_version(), "1.2.3")

  def _patch_libtpu_metadata_not_found(self):
    """Forces the `pip list` fallback even when libtpu is installed."""
    self.enter_context(
        mock.patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError,
        )
    )

  def test_fetch_libtpu_version_from_pip_success(self):
    self._patch_libtpu_metadata_not_found()
    with mock.patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess(
            args="pip list",
            stdout="libtpu 0.1.2",
            returncode=0,
        ),
    ):
      self.assertEqual(cli_helper.fetch_libtpu_version(), "0.1.2")

  def test_fetch_libtpu_version_error_no_libtpu_in_pip_list(self):
    self._patch_libtpu_metadata_not_found()
    with mock.patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess(
            args="pip list",
            stdout="pip list output",
            returncode=0,
        ),
    ):
      self.assertIn("libtpu not found", cli_helper.fetch_libtpu_version())

  def test_fetch_libtpu_version_error_running_pip_list(self):
    self._patch_libtpu_metadata_not_found()
    with mock.patch.object(
        subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(
            returncode=1,
            cmd="pip list",
            output="pip list output",
            stderr="pip list stderr",
        ),
    ):
      self.assertIn("error running pip list", cli_helper.fetch_libtpu_version())

  def test_fetch_libtpu_version_from_pip_unexpected_error(self):
    self._patch_libtpu_metadata_not_found()
    with mock.patch.object(
        subprocess,
        "run",
        side_effect=RuntimeError("unexpected error"),
    ):
      self.assertIn(
          "unexpected error getting libtpu version",
          cli_helper.fetch_libtpu_version(),
      )

  def test_fetch_accelerator_type_success(self):
    with mock.patch.object(
        device,
        "get_local_chips",
        return_value=(device.TpuChip.V4, 4),
    ):
      self.assertEqual(cli_helper.fetch_accelerator_type(), "v4")

  def test_fetch_accelerator_type_no_chips(self):
    with mock.patch.object(
        device,
        "get_local_chips",
        return_value=(None, 0),
    ):
      self.assertEqual(
          cli_helper.fetch_accelerator_type(), "unknown (no TPU chips found)"
      )

  def test_fetch_accelerator_type_error(self):
    with mock.patch.object(
        device,
        "get_local_chips",
        side_effect=RuntimeError("unexpected error"),
    ):
      self.assertIn(
          "unexpected error getting accelerator type",
          cli_helper.fetch_accelerator_type(),
      )

  def test_get_tpu_cli_info(self):
    with mock.patch.object(
        cli_helper,
        "fetch_libtpu_version",
        return_value="1.2.3",
    ), mock.patch.object(
        cli_helper,
        "fetch_accelerator_type",
        return_value="v4",
    ):
      tpu_cli_info = cli_helper.get_tpu_cli_info()
      output = io.StringIO()
      console_obj = console.Console(file=output)
      console_obj.print(tpu_cli_info)
      expected_strings = [
          "Libtpu version",
          "Accelerator type",
          "1.2.3",
          "v4",
      ]
      for s in expected_strings:
        self.assertIn(s, output.getvalue())

  def test_get_process_name(self):
    mock_open = mock.mock_open(read_data="python")
    with mock.patch("builtins.open", mock_open):
      self.assertEqual(cli_helper.get_process_name(123), "python")

  def test_get_process_name_error(self):
    with mock.patch("builtins.open", side_effect=FileNotFoundError):
      self.assertIsNone(cli_helper.get_process_name(123))

  def test_fetch_process_table_success(self):
    with mock.patch.object(
        device,
        "get_chip_owners",
        return_value={
            "/dev/accel0": 123,
            "/dev/accel1": 456,
        },
    ), mock.patch.object(
        cli_helper,
        "get_process_name",
        side_effect=["python", "python3"],
    ):
      table = cli_helper.fetch_process_table(device.TpuChip.V4, 2)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      console_obj.print(table)
      expected_strings = [
          "TPU Process Info",
          "Chip",
          "PID",
          "Process Name",
          "/dev/accel0",
          "123",
          "python",
          "/dev/accel1",
          "456",
          "python3",
      ]
      for s in expected_strings:
        self.assertIn(s, output.getvalue())

  def test_fetch_process_table_no_owners(self):
    with mock.patch.object(
        device,
        "get_chip_owners",
        return_value={},
    ):
      table = cli_helper.fetch_process_table(device.TpuChip.V4, 2)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      console_obj.print(table)
      self.assertIn("None", output.getvalue())

  def test_get_device_usage_success(self):
    with mock.patch.object(
        metrics,
        "get_chip_usage",
        return_value=[metrics.Usage(0, 0, 34088157184, 1.0)],
    ):
      device_usage = cli_helper.get_device_usage(device.TpuChip.V4)
      self.assertIsInstance(device_usage, list)
      self.assertLen(device_usage, 1)
      self.assertEqual(device_usage[0].memory_usage, 0)
      self.assertEqual(device_usage[0].total_memory, 34088157184)
      self.assertEqual(device_usage[0].duty_cycle_pct, 1.0)
      self.assertEqual(device_usage[0].device_id, 0)

  def test_get_device_usage_grpc_unavailable(self):
    with mock.patch.object(
        metrics,
        "get_chip_usage",
        side_effect=FakeRpcError(grpc.StatusCode.UNAVAILABLE),
    ):

      device_usage = cli_helper.get_device_usage(device.TpuChip.V4)
      self.assertIsInstance(device_usage, panel.Panel)
      self.assertIn("Runtime Utilization Status", device_usage.title)
      self.assertIn("Libtpu metrics unavailable.", device_usage.renderable)

  def test_get_device_usage_grpc_unknown_error(self):
    with mock.patch.object(
        metrics,
        "get_chip_usage",
        side_effect=FakeRpcError(grpc.StatusCode.UNKNOWN),
    ):
      device_usage = cli_helper.get_device_usage(device.TpuChip.V4)
      self.assertIsInstance(device_usage, panel.Panel)
      self.assertIn("Runtime Utilization Error", device_usage.title)
      self.assertIn(
          "ERROR fetching runtime utilization", device_usage.renderable
      )

  def test_get_libtpusdk_version(self):
    """Tests function for getting the libtpu SDK version."""
    mock_fetch_libtpu_version = mock.Mock()
    mock_fetch_libtpu_version.return_value = "1.2.3"
    with mock.patch.object(
        cli_helper,
        "fetch_libtpu_version",
        mock_fetch_libtpu_version,
    ):
      self.assertEqual(cli_helper._get_libtpusdk_version(), "1.2.3")

  def test_get_libtpusdk_version_error(self):
    """Tests function for getting the libtpu SDK version."""
    mock_fetch_libtpu_version = mock.Mock()
    mock_fetch_libtpu_version.return_value = "unknown (libtpu not found)"
    with mock.patch.object(
        cli_helper,
        "fetch_libtpu_version",
        mock_fetch_libtpu_version,
    ):
      self.assertIsNone(cli_helper._get_libtpusdk_version())

  def test_is_incompatible_python_version_libtpu_not_imported(self):
    """Tests function for checking Python 3.12 and libtpu not imported."""
    # Mock the libtpu SDK version
    mock_get_libtpusdk_version = mock.Mock()
    mock_get_libtpusdk_version.return_value = None
    # Make the version info closer to the actual sys.version_info output
    VersionInfo = collections.namedtuple(
        "VersionInfo",
        ["major", "minor", "micro", "releaselevel", "serial"],
    )
    mock_python_version = VersionInfo(
        major=3, minor=12, micro=5, releaselevel="final", serial=0
    )

    with mock.patch.object(
        cli_helper,
        "_get_libtpusdk_version",
        mock_get_libtpusdk_version,
    ):
      # Mock the Python version
      with mock.patch("sys.version_info", mock_python_version):
        self.assertTrue(cli_helper.is_incompatible_python_version())

  @parameterized.named_parameters(
      dict(
          testcase_name="3_10_5-and-libtpu_0_0_20_0",
          python_version_major=3,
          python_version_minor=10,
          python_version_micro=5,
          libtpu_version="0.0.20.0",
          expected_result=True,
      ),
      dict(
          testcase_name="3_10_5-and-libtpu_0_0_20_1",
          python_version_major=3,
          python_version_minor=11,
          python_version_micro=5,
          libtpu_version="0.0.20.1",
          expected_result=False,
      ),
      dict(
          testcase_name="3_11_0-and-libtpu_0_0_20_0",
          python_version_major=3,
          python_version_minor=11,
          python_version_micro=2,
          libtpu_version="0.0.20.0",
          expected_result=False,
      ),
      dict(
          testcase_name="3_12_7-and-libtpu_0_0_20_0",
          python_version_major=3,
          python_version_minor=12,
          python_version_micro=7,
          libtpu_version="0.0.20.0",
          expected_result=True,
      ),
      dict(
          testcase_name="3_13_5-and-libtpu_0_0_20_0",
          python_version_major=3,
          python_version_minor=13,
          python_version_micro=5,
          libtpu_version="0.0.20.0",
          expected_result=True,
      ),
      dict(
          testcase_name="3_11_0-and-libtpu_0_0_20_1",
          python_version_major=3,
          python_version_minor=11,
          python_version_micro=2,
          libtpu_version="0.0.20.1",
          expected_result=False,
      ),
      dict(
          testcase_name="3_11_1-and-libtpu_0_0_20_1",
          python_version_major=3,
          python_version_minor=11,
          python_version_micro=1,
          libtpu_version="0.0.20.1",
          expected_result=False,
      ),
      dict(
          testcase_name="3_13_2-and-libtpu_0_0_20_1",
          python_version_major=3,
          python_version_minor=11,
          python_version_micro=2,
          libtpu_version="0.0.20.1",
          expected_result=False,
      ),
      dict(
          testcase_name="3_13_5-and-libtpu_0_0_21_0",
          python_version_major=3,
          python_version_minor=13,
          python_version_micro=5,
          libtpu_version="0.0.21.0",
          expected_result=False,
      ),
      dict(
          testcase_name="3_11_5-and-libtpu_0_0_20_0",
          python_version_major=3,
          python_version_minor=11,
          python_version_micro=5,
          libtpu_version="0.0.20.0",
          expected_result=False,
      ),
  )
  def test_is_incompatible_python_version_with_libtpu_version(
      self,
      python_version_major,
      python_version_minor,
      python_version_micro,
      libtpu_version,
      expected_result,
  ):
    """Tests function for checking Python 3.11 and libtpu not imported."""
    # Mock the libtpu SDK version
    mock_get_libtpusdk_version = mock.Mock()
    mock_get_libtpusdk_version.return_value = libtpu_version
    # Make the version info closer to the actual sys.version_info output
    VersionInfo = collections.namedtuple(
        "VersionInfo",
        ["major", "minor", "micro", "releaselevel", "serial"],
    )
    mock_python_version = VersionInfo(
        major=python_version_major,
        minor=python_version_minor,
        micro=python_version_micro,
        releaselevel="final",
        serial=0,
    )
    with mock.patch.object(
        cli_helper,
        "_get_libtpusdk_version",
        mock_get_libtpusdk_version,
    ):
      # Mock the Python version
      with mock.patch("sys.version_info", mock_python_version):
        self.assertEqual(
            cli_helper.is_incompatible_python_version(), expected_result
        )

  def test_calculate_percentile(self):
    # Empty buckets
    self.assertEqual(cli_helper.calculate_percentile([], 0.5), 0.0)

    # Total count is 0
    self.assertEqual(
        cli_helper.calculate_percentile([(1.0, 0.0), (2.0, 0.0)], 0.5), 0.0
    )

    # Interpolation
    buckets = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
    # Target percentile 0.5 -> target count 15
    # lies between (1.0, 10.0) and (2.0, 20.0). ratio = (15-10)/(20-10) = 0.5.
    # Val = 1.0 + 0.5 * (2.0 - 1.0) = 1.5
    self.assertAlmostEqual(cli_helper.calculate_percentile(buckets, 0.5), 1.5)

    # Inf upper bound
    buckets_with_inf = [(1.0, 10.0), (2.0, 20.0), (float("inf"), 30.0)]
    # Target percentile 0.9 -> target count 27
    # lies between (2.0, 20.0) and (inf, 30.0) -> returns prev_upper_bound (2.0)
    self.assertEqual(
        cli_helper.calculate_percentile(buckets_with_inf, 0.9), 2.0
    )

    # Exact match
    self.assertEqual(cli_helper.calculate_percentile(buckets, 1.0), 3.0)

    # Zero percentile with zero count first bucket (covers count == prev_count)
    self.assertEqual(
        cli_helper.calculate_percentile([(1.0, 0.0), (2.0, 10.0)], 0.0), 1.0
    )

    # Percentile > 1.0 (covers default return at end of calculate_percentile)
    self.assertEqual(cli_helper.calculate_percentile(buckets, 1.5), 3.0)

  @parameterized.named_parameters(
      dict(
          testcase_name="gbytes_per_sec",
          metric_name="jax_orbax_write_gbytes_per_sec",
          expected="GiB/s",
      ),
      dict(
          testcase_name="gb_per_sec",
          metric_name="orbax_write_gb_per_sec",
          expected="GiB/s",
      ),
      dict(
          testcase_name="gbytes",
          metric_name="jax_orbax_write_gbytes",
          expected="GiB",
      ),
      dict(
          testcase_name="bytes",
          metric_name="grain_python_data_sources_bytes_read",
          expected="B",
      ),
      dict(
          testcase_name="generic_bytes",
          metric_name="test_bytes_metric",
          expected="B",
      ),
      dict(
          testcase_name="duration_secs",
          metric_name="orbax_write_duration_secs",
          expected="s",
      ),
      dict(
          testcase_name="duration_ns",
          metric_name="grain_python_dataset_next_duration_ns",
          expected="ns",
      ),
      dict(
          testcase_name="gb_s",
          metric_name="orbax_write_gb_s",
          expected="GiB/s",
      ),
      dict(
          testcase_name="gb",
          metric_name="orbax_write_gb",
          expected="GiB",
      ),
      dict(
          testcase_name="duration_sec",
          metric_name="orbax_write_duration_sec",
          expected="s",
      ),
      dict(
          testcase_name="seconds",
          metric_name="orbax_write_seconds",
          expected="s",
      ),
      dict(
          testcase_name="secs",
          metric_name="orbax_write_secs",
          expected="s",
      ),
      dict(
          testcase_name="sec",
          metric_name="orbax_write_sec",
          expected="s",
      ),
      dict(
          testcase_name="ns",
          metric_name="orbax_write_ns",
          expected="ns",
      ),
      dict(
          testcase_name="unknown",
          metric_name="some_random_metric_name",
          expected="",
      ),
  )
  def test_get_unit_from_name(self, metric_name, expected):
    self.assertEqual(cli_helper.get_unit_from_name(metric_name), expected)


class CliHelperMetricTablesTest(parameterized.TestCase):

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

  def test_fetch_metric_tables(self):
    validated_metrics = [
        ("hbm_usage", None),
        ("duty_cycle_percent", None),
        ("tensorcore_utilization", None),
    ]
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        cli_helper,
        "get_metric_table",
        return_value=["Metric Table"],
    ):
      renderables = cli_helper.fetch_metric_tables(
          validated_metrics, chip_type, count
      )
      self.assertLen(renderables, 3)

  @parameterized.named_parameters(
      (
          "duty_cycle",
          "duty_cycle_percent",
          [
              "TPU Duty Cycle",
              "1.00%",
              "2.05%",
              "3.06%",
              "4.00%",
              "Core ID",
              "Duty Cycle (%)",
          ],
      ),
      (
          "hbm_usage",
          "hbm_usage",
          [
              "TPU HBM Usage",
              "0.00 GiB",
              "31.75 GiB",
              "Device",
              "HBM Usage (GiB)",
          ],
      ),
      (
          "hlo_queue_size",
          "hlo_queue_size",
          [
              "HLO Queue Size",
              "0",
              "5",
              "Device",
              "Queue Size",
          ],
      ),
      (
          "hlo_exec_timing",
          "hlo_exec_timing",
          [
              "HLO Execution Timing",
              "0",
              "10.00",
              "32.00",
              "192.00",
              "224.00",
              "255.36",
              "Device",
              "Mean",
              "P50",
              "P90",
              "P95",
              "P999",
          ],
      ),
      (
          "runtime_hbm_utilization",
          "runtime_hbm_utilization",
          [
              "Runtime HBM Utilization",
              "Device",
              "Utilization (%)",
              "10.50%",
              "20.00%",
          ],
      ),
      (
          "tensorcore_idle_duration",
          "tensorcore_idle_duration",
          [
              "TensorCore Idle Duration",
              "Device",
              "Idle Duration (s)",
              "1.50",
              "2.00",
          ],
      ),
  )
  def test_get_metric_table(self, metric_name, expected_strings):
    chip_usage = [
        metrics.Usage(0, 0, 34088157184, 1.0),
        metrics.Usage(1, 0, 34088157184, 2.05),
        metrics.Usage(2, 0, 34088157184, 3.055),
        metrics.Usage(3, 0, 34088157184, 4.001),
    ]
    queue_sizes = [
        metrics.HloQueueSize(0, 0),
        metrics.HloQueueSize(1, 5),
        metrics.HloQueueSize(2, 10),
        metrics.HloQueueSize(3, 2),
    ]
    exec_timings = [
        metrics.HloExecutionTiming(0, 10.0, 32.0, 192.0, 224.0, 255.36),
    ]
    with mock.patch.object(
        metrics,
        "get_chip_usage",
        return_value=chip_usage,
    ), mock.patch.object(
        metrics,
        "get_hlo_queue_size",
        return_value=queue_sizes,
    ), mock.patch.object(
        metrics,
        "get_hlo_exec_timing",
        return_value=exec_timings,
    ), mock.patch.object(
        metrics,
        "get_runtime_hbm_utilization",
        return_value=[(0, 10.5), (1, 20.0)],
    ), mock.patch.object(
        metrics,
        "get_tensorcore_idle_duration",
        return_value=[(0, 1.5), (1, 2.0)],
    ):
      table = cli_helper.get_metric_table(
          (metric_name, None), device.TpuChip.V4, 4
      )
      output = io.StringIO()
      console_obj = console.Console(file=output)
      console_obj.print(*table)
      for s in expected_strings:
        self.assertIn(s, output.getvalue())

  def test_get_hlo_queue_size_table(self):
    chip_type = device.TpuChip.V4
    count = 4
    queue_sizes = [
        metrics.HloQueueSize(0, 0),
        metrics.HloQueueSize(1, 5),
        metrics.HloQueueSize(2, 10),
        metrics.HloQueueSize(3, 2),
    ]
    with mock.patch.object(
        metrics,
        "get_hlo_queue_size",
        return_value=queue_sizes,
    ):
      renderables = cli_helper.get_hlo_queue_size_table(chip_type, count)

      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HLO Queue Size", output_string)
      self.assertIn("Device", output_string)
      self.assertIn("Queue Size", output_string)
      self.assertIn("0", output_string)
      self.assertIn("5", output_string)
      self.assertIn("10", output_string)
      self.assertIn("2", output_string)

  def test_get_hlo_queue_size_table_unavailable(self):
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        metrics,
        "get_hlo_queue_size",
        side_effect=FakeRpcError(grpc.StatusCode.UNAVAILABLE),
    ):
      renderables = cli_helper.get_hlo_queue_size_table(chip_type, count)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HLO Queue Size Status", output_string)
      self.assertIn("HLO queue size metrics unavailable", output_string)

  def test_get_hlo_queue_size_table_error(self):
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        metrics,
        "get_hlo_queue_size",
        side_effect=FakeRpcError(grpc.StatusCode.UNKNOWN),
    ):
      renderables = cli_helper.get_hlo_queue_size_table(chip_type, count)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HLO Queue Size Error", output_string)
      self.assertIn("ERROR fetching HLO queue size", output_string)

  def test_get_runtime_hbm_utilization_table(self):
    chip_type = device.TpuChip.V4
    count = 4
    runtime_hbm_utilization = [(0, 10.5), (1, 20.0)]
    with mock.patch.object(
        metrics,
        "get_runtime_hbm_utilization",
        return_value=runtime_hbm_utilization,
    ):
      renderables = cli_helper.get_runtime_hbm_utilization_table(
          chip_type, count
      )

      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("Runtime HBM Utilization", output_string)
      self.assertIn("Device", output_string)
      self.assertIn("Utilization (%)", output_string)
      self.assertIn("10.50%", output_string)
      self.assertIn("20.00%", output_string)

  def test_get_runtime_hbm_utilization_table_error(self):
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        metrics,
        "get_runtime_hbm_utilization",
        side_effect=FakeRpcError(grpc.StatusCode.UNKNOWN),
    ):
      renderables = cli_helper.get_runtime_hbm_utilization_table(
          chip_type, count
      )

      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HBM BW Util Error", output_string)
      self.assertIn("ERROR fetching HBM bandwidth utilization", output_string)
      for i in range(count):
        self.assertIn(str(i), output_string)
      self.assertEqual(output_string.count("N/A"), count)

  def test_get_tensorcore_idle_duration_table(self):
    chip_type = device.TpuChip.V4
    count = 4
    tensorcore_idle_duration = [(0, 1.5), (1, 2.0)]
    with mock.patch.object(
        metrics,
        "get_tensorcore_idle_duration",
        return_value=tensorcore_idle_duration,
    ):
      renderables = cli_helper.get_tensorcore_idle_duration_table(
          chip_type, count
      )

      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("TensorCore Idle Duration", output_string)
      self.assertIn("Device", output_string)
      self.assertIn("Idle Duration (s)", output_string)
      self.assertIn("1.50", output_string)
      self.assertIn("2.00", output_string)

  def test_get_tensorcore_idle_duration_table_error(self):
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        metrics,
        "get_tensorcore_idle_duration",
        side_effect=FakeRpcError(grpc.StatusCode.UNKNOWN),
    ):
      renderables = cli_helper.get_tensorcore_idle_duration_table(
          chip_type, count
      )

      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("TensorCore Idle Duration Error", output_string)
      self.assertIn("ERROR fetching TensorCore idle duration", output_string)
      for i in range(count):
        self.assertIn(str(i), output_string)
      self.assertEqual(output_string.count("N/A"), count)

  def test_get_hlo_exec_timing_table(self):
    chip_type = device.TpuChip.V4
    count = 4
    exec_timings = [
        metrics.HloExecutionTiming(0, 10.0, 32.0, 192.0, 224.0, 255.36),
    ]
    with mock.patch.object(
        metrics,
        "get_hlo_exec_timing",
        return_value=exec_timings,
    ):
      renderables = cli_helper.get_hlo_exec_timing_table(chip_type, count)

      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HLO Execution Timing", output_string)
      self.assertIn("Device", output_string)
      self.assertIn("Mean", output_string)
      self.assertIn("P50", output_string)
      self.assertIn("P90", output_string)
      self.assertIn("P95", output_string)
      self.assertIn("P999", output_string)
      self.assertIn("0", output_string)
      self.assertIn("10.00", output_string)
      self.assertIn("32.00", output_string)
      self.assertIn("192.00", output_string)
      self.assertIn("224.00", output_string)
      self.assertIn("255.36", output_string)

  def test_get_hlo_exec_timing_table_empty(self):
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        metrics,
        "get_hlo_exec_timing",
        return_value=[],
    ):
      renderables = cli_helper.get_hlo_exec_timing_table(chip_type, count)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HLO Execution Timing", output_string)
      for i in range(count):
        self.assertIn(str(i), output_string)
      # 5 columns of N/A per row * 4 rows = 20 N/A
      self.assertEqual(output_string.count("N/A"), 20)

  def test_get_hlo_exec_timing_table_unavailable(self):
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        metrics,
        "get_hlo_exec_timing",
        side_effect=FakeRpcError(grpc.StatusCode.UNAVAILABLE),
    ):
      renderables = cli_helper.get_hlo_exec_timing_table(chip_type, count)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HLO Execution Timing Status", output_string)
      self.assertIn("HLO execution timing metrics unavailable", output_string)

  def test_get_hlo_exec_timing_table_error(self):
    chip_type = device.TpuChip.V4
    count = 4
    with mock.patch.object(
        metrics,
        "get_hlo_exec_timing",
        side_effect=FakeRpcError(grpc.StatusCode.UNKNOWN),
    ):
      renderables = cli_helper.get_hlo_exec_timing_table(chip_type, count)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

      self.assertIn("HLO Execution Timing Error", output_string)
      self.assertIn("ERROR fetching HLO execution timing", output_string)

  @parameterized.named_parameters(
      (
          "buffer_transfer_latency",
          "buffer_transfer_latency",
          "Buffer Transfer Latency",
      ),
      (
          "inbound_buffer_transfer_latency",
          "inbound_buffer_transfer_latency",
          "Inbound Buffer Transfer Latency",
      ),
      (
          "host_to_device_transfer_latency",
          "host_to_device_transfer_latency",
          "Host to Device Transfer Latency",
      ),
      (
          "device_to_host_transfer_latency",
          "device_to_host_transfer_latency",
          "Device to Host Transfer Latency",
      ),
      (
          "collective_e2e_latency",
          "collective_e2e_latency",
          "Collective E2E Latency",
      ),
      (
          "host_compute_latency",
          "host_compute_latency",
          "Host Compute Latency",
      ),
  )
  def test_get_metric_table_latency(self, metric_name, expected_title):
    """Tests that the correct render method is called for the latency metric."""
    mock_table = rich_table.Table(title=expected_title)
    with mock.patch.object(
        cli_helper.TransferLatencyTables, "render", return_value=mock_table
    ) as mock_render:
      renderables = cli_helper.get_metric_table(
          (metric_name, None), device.TpuChip.V4, 4
      )
      mock_render.assert_called_once_with(metric_name, None)
      self.assertIn(mock_table, renderables)

  @parameterized.named_parameters(
      (
          "grpc_unavailable",
          grpc.StatusCode.UNAVAILABLE,
          "Libtpu metrics unavailable",
      ),
      (
          "grpc_error",
          grpc.StatusCode.INTERNAL,
          "ERROR fetching runtime utilization",
      ),
  )
  def test_get_metric_table_with_exception(self, grpc_code, expected_string):
    with mock.patch.object(
        metrics,
        "get_chip_usage",
        side_effect=FakeRpcError(grpc_code),
    ):
      renderables = cli_helper.fetch_metric_tables(
          [
              ("hbm_usage", None),
              ("duty_cycle_percent", None),
          ],
          device.TpuChip.V4,
          4,
      )
      output = io.StringIO()
      console_obj = console.Console(file=output)
      console_obj.print(*renderables)
      self.assertIn(expected_string, output.getvalue())
      self.assertIn("N/A", output.getvalue())

  def test_get_metric_table_tensorcore_utilization(self):
    with mock.patch.object(
        cli_helper.TensorCoreUtilizationTable,
        "render",
        return_value=rich_table.Table(
            title="Mocked TensorCore Utilization Table"
        ),
    ):
      table = cli_helper.get_metric_table(
          ("tensorcore_utilization", None), device.TpuChip.V4, 4
      )
      self.assertLen(table, 1)
      self.assertIsInstance(table[0], rich_table.Table)
      self.assertEqual(table[0].title, "Mocked TensorCore Utilization Table")

  def test_get_metric_table_for_prometheus_metric_orbax(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "jax_orbax_write_gbytes"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "jax_orbax_write_gbytes_total"
    mock_sample.value = 10.5
    mock_sample.labels = {"host": "host1", "storage_type": "gcs"}
    mock_family.samples = [mock_sample]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[mock_family],
    ):
      table = cli_helper.get_metric_table(
          ("orbax_write_size", None), device.TpuChip.V4, 4
      )
      self.assertLen(table, 2)
      self.assertIsInstance(table[0], Text)
      self.assertIsInstance(table[1], rich_table.Table)
      output = io.StringIO()
      console_obj = console.Console(file=output, width=1000)
      for item in table:
        console_obj.print(item)
      self.assertIn("Orbax Write Size", output.getvalue())
      self.assertIn("10.50 GiB", output.getvalue())

  def test_get_metric_table_for_individual_prometheus_metric_pygrain(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "grain_python_data_sources_bytes_read"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "grain_python_data_sources_bytes_read_total"
    mock_sample.value = 1024.0
    mock_sample.labels = {"source": "source1"}
    mock_family.samples = [mock_sample]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[mock_family],
    ):
      table = cli_helper.get_metric_table(
          ("pygrain_data_sources_read_size", None), device.TpuChip.V4, 4
      )
      self.assertLen(table, 2)
      self.assertIsInstance(table[0], Text)
      self.assertIsInstance(table[1], rich_table.Table)
      output = io.StringIO()
      console_obj = console.Console(file=output, width=1000)
      console_obj.print(table[0])
      console_obj.print(table[1])
      self.assertIn("Pygrain Data Sources Read Size", output.getvalue())
      self.assertIn("Read", output.getvalue())
      self.assertIn("1024 B", output.getvalue())
      self.assertIn("source1", output.getvalue())


class TpuRuntimeUtilizationTableTest(absltest.TestCase):
  """Tests for the TpuRuntimeUtilizationTable helper class."""

  def test_render_success(self):
    """Tests that the table renders correctly with valid data."""
    chip_usage = [
        metrics.Usage(
            device_id=0,
            memory_usage=1073741824,
            total_memory=34088157184,
            duty_cycle_pct=95.5,
        ),
    ]
    with mock.patch.object(
        cli_helper, "get_device_usage", return_value=chip_usage
    ):
      table_helper = cli_helper.TpuRuntimeUtilizationTable()
      renderables = table_helper.render(chip_type=device.TpuChip.V4, count=1)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

    self.assertIn("1.00 GiB / 31.75 GiB", output_string)
    self.assertIn("95.50%", output_string)

  def test_render_unavailable_grpc_error(self):
    """Tests that a WARNING panel is shown for a gRPC UNAVAILABLE error."""
    error_panel = panel.Panel("Libtpu metrics unavailable.")
    with mock.patch.object(
        cli_helper, "get_device_usage", return_value=error_panel
    ):
      table_helper = cli_helper.TpuRuntimeUtilizationTable()
      renderables = table_helper.render(chip_type=device.TpuChip.V4, count=1)
      output = io.StringIO()
      console_obj = console.Console(file=output)
      for item in renderables:
        console_obj.print(item)
      output_string = output.getvalue()

    self.assertIn("Libtpu metrics unavailable", output_string)
    self.assertIn("N/A", output_string)


class TensorCoreUtilizationTableTest(absltest.TestCase):
  """Tests for the TensorCoreUtilizationTable helper class."""

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

  def test_render_success(self):
    """Tests that the table renders correctly when the libtpu SDK is available."""
    mock_sdk = mock.Mock()
    mock_sdk.tpumonitoring.get_metric.return_value.data.return_value = [
        "10.0",
        "20.5",
    ]
    with mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk,
    ):
      table_helper = cli_helper.TensorCoreUtilizationTable()
      table = table_helper.render(count=2)

      output = io.StringIO()
      console_obj = console.Console(file=output)
      console_obj.print(table)
      output_string = output.getvalue()

      self.assertIsInstance(table, rich_table.Table)
      self.assertEqual(table.columns[0].header, "Core ID")
      self.assertIn("10.0%", output_string)
      self.assertIn("20.5%", output_string)

  @mock.patch.dict(sys.modules, {"libtpu": None})
  def test_render_import_error(self):
    """Tests that a WARNING panel is shown for an ImportError."""
    table_helper = cli_helper.TensorCoreUtilizationTable()
    result_panel = table_helper.render(count=1)

    self.assertIsInstance(result_panel, panel.Panel)
    self.assertIn("WARNING: ImportError", str(result_panel.renderable))

  def test_render_attribute_error(self):
    """Tests that a WARNING panel is shown for an AttributeError."""
    mock_sdk = mock.Mock()
    mock_sdk.tpumonitoring.get_metric.side_effect = AttributeError("mock error")
    with mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk,
    ):
      table_helper = cli_helper.TensorCoreUtilizationTable()
      result_panel = table_helper.render(count=1)

      self.assertIsInstance(result_panel, panel.Panel)
      self.assertIn("WARNING: AttributeError", str(result_panel.renderable))

  def test_render_runtime_error(self):
    """Tests that a WARNING panel is shown for a RuntimeError."""
    mock_sdk = mock.Mock()
    mock_sdk.tpumonitoring.get_metric.side_effect = RuntimeError("mock error")
    with mock.patch.object(
        cli_helper,
        "libtpu_sdk",
        mock_sdk,
    ):
      table_helper = cli_helper.TensorCoreUtilizationTable()
      result_group = table_helper.render(count=2)

    self.assertIsInstance(result_group, console.Group)
    renderables = list(result_group.renderables)
    self.assertLen(renderables, 2)

    warning_panel = renderables[0]
    self.assertIsInstance(warning_panel, panel.Panel)
    self.assertIn("WARNING: RuntimeError", str(warning_panel.renderable))

    na_table = renderables[1]
    self.assertIsInstance(na_table, rich_table.Table)
    output = io.StringIO()
    console_obj = console.Console(file=output)
    console_obj.print(na_table)
    output_string = output.getvalue()
    self.assertIn("N/A", output_string)
    self.assertEqual(output_string.count("N/A"), 2)


class LatencyTablesTest(absltest.TestCase):
  """Tests for the TransferLatencyTables helper class."""

  def test_render_with_filter(self):
    """Tests that the table renders correctly with a percentile filter."""
    mock_distributions = [
        metrics.TransferLatencyDistribution(
            transfer_size="1024", p50=100.1, p90=200.2, p95=300.3, p999=400.4
        )
    ]
    with mock.patch.object(
        metrics, "get_transfer_latency", return_value=mock_distributions
    ):
      table_helper = cli_helper.TransferLatencyTables()
      table = table_helper.render(
          "buffer_transfer_latency", filters={"percentile": ["p50", "p95"]}
      )

    self.assertIsInstance(table, rich_table.Table)
    output = io.StringIO()
    console_obj = console.Console(file=output)
    console_obj.print(table)
    output_string = output.getvalue()

    self.assertIn("P50", output_string)
    self.assertIn("P95", output_string)
    self.assertNotIn("P90", output_string)
    self.assertNotIn("P999", output_string)
    self.assertIn("100.10 us", output_string)
    self.assertIn("300.30 us", output_string)

  def test_render_success(self):
    """Tests that the table renders correctly with valid data."""
    mock_distributions = [
        metrics.TransferLatencyDistribution(
            transfer_size="1024", p50=100.1, p90=200.2, p95=300.3, p999=400.4
        )
    ]
    with mock.patch.object(
        metrics, "get_transfer_latency", return_value=mock_distributions
    ):
      table_helper = cli_helper.TransferLatencyTables()
      table = table_helper.render("buffer_transfer_latency")

    self.assertIsInstance(table, rich_table.Table)
    output = io.StringIO()
    console_obj = console.Console(file=output)
    console_obj.print(table)
    output_string = output.getvalue()
    print(f"output_string: {output_string}")
    self.assertIn("Buffer Size", output_string)
    self.assertIn("P50", output_string)
    self.assertIn("P90", output_string)
    self.assertIn("P95", output_string)
    self.assertIn("P999", output_string)
    self.assertIn("1024", output_string)
    self.assertIn("100.10 us", output_string)
    self.assertIn("200.20 us", output_string)
    self.assertIn("300.30 us", output_string)
    self.assertIn("400.40 us", output_string)

  def test_render_unavailable_rpc_error(self):
    """Tests the panel for an UNAVAILABLE gRPC error."""
    with mock.patch.object(
        metrics,
        "get_transfer_latency",
        side_effect=FakeRpcError(grpc.StatusCode.UNAVAILABLE),
    ):
      table_helper = cli_helper.TransferLatencyTables()
      result_panel = table_helper.render("host_to_device_transfer_latency")

    self.assertIsInstance(result_panel, panel.Panel)
    self.assertIn("WARNING", str(result_panel.renderable))
    self.assertIn(
        "Host to Device Transfer Latency Status", str(result_panel.title)
    )

  def test_render_generic_rpc_error(self):
    """Tests the panel for a generic gRPC error."""
    with mock.patch.object(
        metrics,
        "get_transfer_latency",
        side_effect=FakeRpcError(grpc.StatusCode.UNKNOWN),
    ):
      table_helper = cli_helper.TransferLatencyTables()
      result_panel = table_helper.render("device_to_host_transfer_latency")

    self.assertIsInstance(result_panel, panel.Panel)
    self.assertIn("ERROR", str(result_panel.renderable))
    self.assertIn(
        "Device to Host Transfer Latency Error", str(result_panel.title)
    )


class CliHelperGrpcMetricTests(parameterized.TestCase):
  """Tests for the metric table rendering functions in cli_helper."""

  def test_fetch_metric_tables_latency(self):
    """Tests that fetch_metric_tables calls the correct render method."""
    mock_table = rich_table.Table(title="Mocked Table")
    with mock.patch.object(
        cli_helper.TransferLatencyTables, "render", return_value=mock_table
    ) as mock_render:
      renderables = cli_helper.fetch_metric_tables(
          [("buffer_transfer_latency", None)],
          device.TpuChip.V4,
          4,
      )
      mock_render.assert_called_once_with("buffer_transfer_latency", None)
      self.assertIn(mock_table, renderables)

  @parameterized.named_parameters(
      (
          "grpc_unavailable",
          grpc.StatusCode.UNAVAILABLE,
          panel.Panel,
          "[b]Runtime Utilization Status[/b]",
      ),
      (
          "generic_grpc_error",
          grpc.StatusCode.UNKNOWN,
          panel.Panel,
          "[b]Runtime Utilization Error[/b]",
      ),
  )
  def test_get_device_usage_error(
      self, error_code, expected_type, expected_title
  ):
    """Tests get_device_usage with different gRPC error scenarios."""
    with mock.patch.object(
        metrics, "get_chip_usage", side_effect=FakeRpcError(error_code)
    ):
      result = cli_helper.get_device_usage(device.TpuChip.V4)

    self.assertIsInstance(result, expected_type)
    self.assertEqual(result.title, expected_title)


class CliHelperPrometheusMetricsTest(parameterized.TestCase):

  def test_get_prometheus_metric_table_scalar_success_orbax(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "jax_orbax_write_gbytes"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "jax_orbax_write_gbytes_total"
    mock_sample.value = 10.5
    mock_sample.labels = {"host": "host1", "storage_type": "gcs"}
    mock_family.samples = [mock_sample]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[mock_family],
    ):
      tables = cli_helper.get_prometheus_metric_table("orbax_write_size")
      self.assertLen(tables, 2)
      self.assertIsInstance(tables[0], Text)
      self.assertIsInstance(tables[1], rich_table.Table)
      output = io.StringIO()
      console_obj = console.Console(file=output, width=1000)
      for item in tables:
        console_obj.print(item)
      self.assertIn("Orbax Write Size", output.getvalue())
      self.assertIn("10.50 GiB", output.getvalue())

  def test_get_prometheus_metric_table_scalar_success_pygrain(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "grain_python_data_sources_bytes_read"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "grain_python_data_sources_bytes_read_total"
    mock_sample.value = 1024.0
    mock_sample.labels = {"source": "source1"}
    mock_family.samples = [mock_sample]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[mock_family],
    ):
      tables = cli_helper.get_prometheus_metric_table(
          "pygrain_data_sources_read_size"
      )
      self.assertLen(tables, 2)
      self.assertIsInstance(tables[0], Text)
      self.assertIsInstance(tables[1], rich_table.Table)
      output = io.StringIO()
      console_obj = console.Console(file=output, width=1000)
      console_obj.print(tables[0])
      console_obj.print(tables[1])
      self.assertIn("Pygrain Data Sources Read Size", output.getvalue())
      self.assertIn("Read", output.getvalue())
      self.assertIn("1024 B", output.getvalue())
      self.assertIn("source1", output.getvalue())

  def test_get_prometheus_metric_table_hist_success_orbax(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "jax_checkpoint_write_async_blocking_duration_secs"
    mock_family.type = "histogram"
    mock_sample1 = mock.create_autospec(Sample, instance=True)
    mock_sample1.name = (
        "jax_checkpoint_write_async_blocking_duration_secs_bucket"
    )
    mock_sample1.value = 5.0
    mock_sample1.labels = {"le": "0.1", "host": "host1"}
    mock_sample2 = mock.create_autospec(Sample, instance=True)
    mock_sample2.name = (
        "jax_checkpoint_write_async_blocking_duration_secs_bucket"
    )
    mock_sample2.value = 10.0
    mock_sample2.labels = {"le": "+Inf", "host": "host1"}
    mock_family.samples = [mock_sample1, mock_sample2]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[mock_family],
    ):
      tables = cli_helper.get_prometheus_metric_table(
          "checkpoint_write_async_blocking_duration"
      )
      self.assertLen(tables, 2)
      self.assertIsInstance(tables[0], Text)
      self.assertIsInstance(tables[1], rich_table.Table)
      output = io.StringIO()
      console_obj = console.Console(file=output, width=1000)
      for item in tables:
        console_obj.print(item)
      self.assertIn(
          "Checkpoint Write Async Blocking Duration",
          output.getvalue(),
      )
      self.assertIn("P5", output.getvalue())
      self.assertIn("0.01 s", output.getvalue())
      self.assertIn("P99", output.getvalue())
      self.assertIn("0.10 s", output.getvalue())

  def test_get_prometheus_metric_table_hist_success_pygrain(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "grain_python_dataset_next_duration_ns"
    mock_family.type = "histogram"
    mock_sample1 = mock.create_autospec(Sample, instance=True)
    mock_sample1.name = "grain_python_dataset_next_duration_ns_bucket"
    mock_sample1.value = 5.0
    mock_sample1.labels = {"le": "100.0"}
    mock_sample2 = mock.create_autospec(Sample, instance=True)
    mock_sample2.name = "grain_python_dataset_next_duration_ns_bucket"
    mock_sample2.value = 10.0
    mock_sample2.labels = {"le": "+Inf"}
    mock_family.samples = [mock_sample1, mock_sample2]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[mock_family],
    ):
      tables = cli_helper.get_prometheus_metric_table(
          "pygrain_dataset_next_duration"
      )
      self.assertLen(tables, 2)
      self.assertIsInstance(tables[0], Text)
      self.assertIsInstance(tables[1], rich_table.Table)
      output = io.StringIO()
      console_obj = console.Console(file=output, width=1000)
      console_obj.print(tables[0])
      console_obj.print(tables[1])
      self.assertIn(
          "Pygrain Dataset Next Duration",
          output.getvalue(),
      )
      self.assertIn("P5", output.getvalue())
      self.assertIn("10.00 ns", output.getvalue())
      self.assertIn("P99", output.getvalue())
      self.assertIn("100.00 ns", output.getvalue())

  def test_get_prometheus_metric_table_not_found(self):
    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[],
    ):
      tables = cli_helper.get_prometheus_metric_table(
          "pygrain_dataset_next_duration"
      )
      self.assertLen(tables, 1)
      self.assertIsInstance(tables[0], panel.Panel)
      self.assertIn(
          "Metric pygrain_dataset_next_duration not found on Prometheus"
          " server.",
          tables[0].renderable,
      )

  def test_get_prometheus_metric_table_connection_error(self):
    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        side_effect=metrics.PrometheusConnectionError("Connection refused"),
    ):
      tables = cli_helper.get_prometheus_metric_table(
          "pygrain_dataset_next_duration"
      )
      self.assertLen(tables, 1)
      self.assertIsInstance(tables[0], panel.Panel)
      self.assertIn(
          "Could not connect to Pygrain Prometheus server on port 9431",
          tables[0].renderable,
      )
      self.assertIn(
          "ENABLE_PYGRAIN_PROMETHEUS_TELEMETRY=true", tables[0].renderable
      )

  def test_get_prometheus_metric_table_unknown_metric(self):
    tables = cli_helper.get_prometheus_metric_table("unknown_metric_name")
    self.assertLen(tables, 1)
    self.assertIsInstance(tables[0], panel.Panel)
    self.assertIn("Unknown metric: unknown_metric_name", tables[0].renderable)
    self.assertEqual(tables[0].border_style, "red")

  def test_get_prometheus_metric_table_from_families_unknown_metric(self):
    tables = cli_helper.get_prometheus_metric_table_from_families(
        [], "unknown_metric_name", "Orbax"
    )
    self.assertLen(tables, 1)
    self.assertIsInstance(tables[0], panel.Panel)
    self.assertIn("Unknown metric: unknown_metric_name", tables[0].renderable)
    self.assertEqual(tables[0].border_style, "red")

  def test_get_prometheus_metric_table_custom_port(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "jax_orbax_write_gbytes"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "jax_orbax_write_gbytes_total"
    mock_sample.value = 10.5
    mock_sample.labels = {"host": "host1", "storage_type": "gcs"}
    mock_family.samples = [mock_sample]

    with mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "9999"}):
      with mock.patch.object(
          metrics,
          "scrape_prometheus",
          autospec=True,
          return_value=[mock_family],
      ) as mock_scrape:
        cli_helper.get_prometheus_metric_table("orbax_write_size")
        mock_scrape.assert_called_once_with(9999)

  def test_get_prometheus_metric_table_invalid_port(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "jax_orbax_write_gbytes"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "jax_orbax_write_gbytes_total"
    mock_sample.value = 10.5
    mock_sample.labels = {"host": "host1", "storage_type": "gcs"}
    mock_family.samples = [mock_sample]

    with mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "invalid"}):
      with mock.patch.object(
          metrics,
          "scrape_prometheus",
          autospec=True,
          return_value=[mock_family],
      ) as mock_scrape:
        cli_helper.get_prometheus_metric_table("orbax_write_size")
        # Should fallback to default port 9431
        mock_scrape.assert_called_once_with(9431)

  def test_fetch_metric_tables_batch_prometheus_success(self):
    mock_family1 = mock.create_autospec(Metric, instance=True)
    mock_family1.name = "jax_orbax_write_gbytes"
    mock_family1.type = "counter"
    mock_sample1 = mock.create_autospec(Sample, instance=True)
    mock_sample1.name = "jax_orbax_write_gbytes_total"
    mock_sample1.value = 10.5
    mock_sample1.labels = {"host": "host1", "storage_type": "gcs"}
    mock_family1.samples = [mock_sample1]

    mock_family2 = mock.create_autospec(Metric, instance=True)
    mock_family2.name = "jax_orbax_write_start"
    mock_family2.type = "counter"
    mock_sample2 = mock.create_autospec(Sample, instance=True)
    mock_sample2.name = "jax_orbax_write_start_total"
    mock_sample2.value = 3.0
    mock_sample2.labels = {"host": "host1"}
    mock_family2.samples = [mock_sample2]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[mock_family1, mock_family2],
    ) as mock_scrape:
      tables = cli_helper.fetch_metric_tables(
          [("orbax_write_size", None), ("orbax_write_start_count", None)],
          device.TpuChip.V4,
          4,
      )
      self.assertLen(tables, 4)
      self.assertIsInstance(tables[0], Text)
      self.assertIsInstance(tables[1], rich_table.Table)
      self.assertIsInstance(tables[2], Text)
      self.assertIsInstance(tables[3], rich_table.Table)
      mock_scrape.assert_called_once_with(9431)

  def test_fetch_metric_tables_batch_prometheus_custom_port(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "jax_orbax_write_gbytes"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "jax_orbax_write_gbytes_total"
    mock_sample.value = 10.5
    mock_sample.labels = {"host": "host1", "storage_type": "gcs"}
    mock_family.samples = [mock_sample]

    with mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "9999"}):
      with mock.patch.object(
          metrics,
          "scrape_prometheus",
          autospec=True,
          return_value=[mock_family],
      ) as mock_scrape:
        cli_helper.fetch_metric_tables(
            [("orbax_write_size", None)],
            device.TpuChip.V4,
            4,
        )
        mock_scrape.assert_called_once_with(9999)

  def test_fetch_metric_tables_batch_prometheus_invalid_port(self):
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "jax_orbax_write_gbytes"
    mock_family.type = "counter"
    mock_sample = mock.create_autospec(Sample, instance=True)
    mock_sample.name = "jax_orbax_write_gbytes_total"
    mock_sample.value = 10.5
    mock_sample.labels = {"host": "host1", "storage_type": "gcs"}
    mock_family.samples = [mock_sample]

    with mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "invalid"}):
      with mock.patch.object(
          metrics,
          "scrape_prometheus",
          autospec=True,
          return_value=[mock_family],
      ) as mock_scrape:
        cli_helper.fetch_metric_tables(
            [("orbax_write_size", None)],
            device.TpuChip.V4,
            4,
        )
        # Should fallback to default port 9431
        mock_scrape.assert_called_once_with(9431)

  def test_fetch_metric_tables_batch_prometheus_missing_consolidated(self):
    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        return_value=[],
    ) as mock_scrape:
      tables = cli_helper.fetch_metric_tables(
          [("orbax_write_size", None), ("orbax_write_start_count", None)],
          device.TpuChip.V4,
          4,
      )
      self.assertLen(tables, 1)
      self.assertIsInstance(tables[0], panel.Panel)
      self.assertIn("Orbax Metrics Not Found", tables[0].title)
      self.assertIn("orbax_write_size", tables[0].renderable)
      self.assertIn("orbax_write_start_count", tables[0].renderable)
      mock_scrape.assert_called_once_with(9431)

  def test_fetch_metric_tables_batch_prometheus_offline_consolidated(self):
    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        side_effect=metrics.PrometheusConnectionError("Connection refused"),
    ) as mock_scrape:
      tables = cli_helper.fetch_metric_tables(
          [("orbax_write_size", None), ("orbax_write_start_count", None)],
          device.TpuChip.V4,
          4,
      )
      self.assertLen(tables, 1)
      self.assertIsInstance(tables[0], panel.Panel)
      self.assertIn("Orbax Telemetry Offline", tables[0].title)
      mock_scrape.assert_called_once_with(9431)


class CliHelperMarkupInjectionTest(parameterized.TestCase):

  def test_get_tpuz_sequencer_state_handles_markup(self):
    # Mock CoreState with malicious error_message and hlo_detailed_info
    malicious_error = "[/red]Fatal Error"
    malicious_hlo_details = "[/red]HLO Details"

    mock_sequencer_state = metrics.SequencerState(
        sequencer_index=0,
        pc=123,
        tag=0,
        tracemark=0,
        program_id=1,
        run_id=1,
        sequencer_type="KERNEL",
        hlo_location="some_loc",
        hlo_detailed_info=malicious_hlo_details,
    )

    mock_core_state = metrics.CoreState(
        global_core_id=0,
        chip_id=0,
        core_on_chip_index=0,
        core_type="V4",
        xdb_server=False,
        sequencer_states=[mock_sequencer_state],
        program_fingerprint="fp",
        error_message=malicious_error,
    )

    with mock.patch.object(
        metrics,
        "get_tpuz_info",
        return_value=[mock_core_state],
    ):
      # We don't expect this to raise MarkupError now.
      renderables = cli_helper.get_tpuz_sequencer_state(detailed_info=True)
      output = io.StringIO()
      test_console = console.Console(file=output, width=1000)
      for r in renderables:
        test_console.print(r)

      output_str = output.getvalue()
      # Assert that raw markup is present in the output (escaped, not parsed)
      self.assertIn(malicious_error, output_str)
      self.assertIn(malicious_hlo_details, output_str)

  def test_get_tpuz_sequencer_state_grpc_error_markup(self):
    malicious_grpc_error = "[/red]gRPC Error Details"

    class FakeRpcErrorWithDetails(FakeRpcError):

      def __str__(self):
        return malicious_grpc_error

    mock_error = FakeRpcErrorWithDetails(grpc.StatusCode.UNKNOWN)

    with mock.patch.object(
        metrics,
        "get_tpuz_info",
        side_effect=mock_error,
    ):
      renderables = cli_helper.get_tpuz_sequencer_state(detailed_info=True)
      output = io.StringIO()
      test_console = console.Console(file=output, width=1000)
      for r in renderables:
        test_console.print(r)

      output_str = output.getvalue()
      self.assertIn(malicious_grpc_error, output_str)

  def test_get_py_compat_warning_panel_handles_markup(self):
    malicious_version = "[/red]Malicious Version"
    with mock.patch.object(
        cli_helper,
        "_get_libtpusdk_version",
        return_value=malicious_version,
    ):
      panel_obj = cli_helper.get_py_compat_warning_panel()
      output = io.StringIO()
      test_console = console.Console(file=output, width=1000)
      test_console.print(panel_obj)
      output_str = output.getvalue()
      self.assertIn(malicious_version, output_str)

  def test_fetch_metric_tables_exception_markup(self):

    class MaliciousError(Exception):

      def __repr__(self):
        return "[/red]Malicious Exception"

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        side_effect=MaliciousError(),
    ):
      tables = cli_helper.fetch_metric_tables(
          [("orbax_write_size", None)], device.TpuChip.V4, 1
      )
      output = io.StringIO()
      test_console = console.Console(file=output, width=1000)
      for t in tables:
        test_console.print(t)
      output_str = output.getvalue()
      self.assertIn("[/red]Malicious Exception", output_str)

  def test_get_tensorcore_utilization_import_error_markup(self):
    mock_sdk = mock.MagicMock()
    type(mock_sdk).tpumonitoring = mock.PropertyMock(
        side_effect=ImportError("[/red]Malicious ImportError")
    )

    with mock.patch.object(
        cli_helper,
        "ensure_libtpu_initialized",
        return_value="OK",
    ):
      with mock.patch.object(
          cli_helper,
          "libtpu_sdk",
          mock_sdk,
      ):
        table_generator = cli_helper.TensorCoreUtilizationTable()
        panel_obj = table_generator.render(1)
        output = io.StringIO()
        test_console = console.Console(file=output, width=1000)
        test_console.print(panel_obj)
        output_str = output.getvalue()
        self.assertIn("[/red]Malicious ImportError", output_str)

  def test_fetch_metric_tables_prometheus_scalar_label_markup(self):
    mock_family = Metric("jax_orbax_write_storage_type", "help", "gauge")
    malicious_label_val = "[/red]Malicious Storage"
    mock_sample = Sample(
        "jax_orbax_write_storage_type_total",
        {"storage_type": malicious_label_val},
        1.0,
    )
    mock_family.samples = [mock_sample]

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        return_value=[mock_family],
    ):
      tables = cli_helper.fetch_metric_tables(
          [("orbax_write_storage_type", None)],
          device.TpuChip.V4,
          1,
      )
      output = io.StringIO()
      test_console = console.Console(file=output, width=1000)
      for t in tables:
        test_console.print(t)
      output_str = output.getvalue()
      self.assertIn(malicious_label_val, output_str)

  def test_fetch_metric_tables_prometheus_histogram_label_markup(self):
    mock_family = Metric(
        "jax_checkpoint_write_async_blocking_duration_secs", "help", "histogram"
    )
    malicious_label_val = "[/red]Malicious Storage"

    mock_samples = [
        Sample(
            "jax_checkpoint_write_async_blocking_duration_secs_bucket",
            {"storage_type": malicious_label_val, "le": "1.0"},
            1.0,
        ),
        Sample(
            "jax_checkpoint_write_async_blocking_duration_secs_bucket",
            {"storage_type": malicious_label_val, "le": "2.0"},
            2.0,
        ),
        Sample(
            "jax_checkpoint_write_async_blocking_duration_secs_bucket",
            {"storage_type": malicious_label_val, "le": "+Inf"},
            2.0,
        ),
    ]
    mock_family.samples = mock_samples

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        return_value=[mock_family],
    ):
      tables = cli_helper.fetch_metric_tables(
          [("checkpoint_write_async_blocking_duration", None)],
          device.TpuChip.V4,
          1,
      )
      output = io.StringIO()
      test_console = console.Console(file=output, width=1000)
      for t in tables:
        test_console.print(t)
      output_str = output.getvalue()
      self.assertIn(malicious_label_val, output_str)


class SerializationAndUtilityTest(parameterized.TestCase):
  """Tests for CLI helper export serializations, table parsing, and utility functions."""

  def test_to_snake_case(self):
    self.assertEqual(cli_helper.to_snake_case("Memory (GiB)"), "memory")
    self.assertEqual(
        cli_helper.to_snake_case("TPU Device ID #1"), "tpu_device_id_1"
    )
    self.assertEqual(
        cli_helper.to_snake_case("   leading and trailing   "),
        "leading_and_trailing",
    )

  def test_table_to_raw_data(self):
    class CustomCell:
      def __rich__(self):
        return "custom_val"

      def __str__(self):
        return "custom_val"

    table = rich_table.Table(title="Sample Table")
    table.add_column("Col A")
    table.add_column(Text("Col B", style="bold"))
    table.add_column(None)

    table.add_row("val1", Text("val2", style="green"), "val3")
    table.add_row(None, "val5", Text("val6"))
    table.add_row(CustomCell(), None, "val9")

    headers, rows = cli_helper.table_to_raw_data(table)
    self.assertEqual(headers, ["Col A", "Col B", ""])
    self.assertEqual(
        rows,
        [
            ["val1", "val2", "val3"],
            ["", "val5", "val6"],
            ["custom_val", "", "val9"],
        ],
    )

  def test_serialize_tables_to_csv_single_and_multiple(self):
    table1 = rich_table.Table(title="Table One")
    table1.add_column("Col1")
    table1.add_column("Col2")
    table1.add_row("A", "B")

    table2 = rich_table.Table(title="Table Two")
    table2.add_column("ColX")
    table2.add_column("ColY")
    table2.add_row("1", "2")

    # Single table (no section title header comment)
    csv_single = cli_helper.serialize_tables_to_csv([table1])
    self.assertIn("Col1,Col2", csv_single)
    self.assertIn("A,B", csv_single)
    self.assertNotIn("# Table: Table One", csv_single)

    # Multiple tables (includes section title comment)
    csv_multi = cli_helper.serialize_tables_to_csv([table1, table2])
    self.assertIn("# Table: Table One", csv_multi)
    self.assertIn("# Table: Table Two", csv_multi)
    self.assertIn("Col1,Col2", csv_multi)
    self.assertIn("ColX,ColY", csv_multi)

  def test_serialize_tables_to_json(self):
    table1 = rich_table.Table(title="Metrics Summary")
    table1.add_column("Device ID")
    table1.add_column("Usage (GB)")
    table1.add_row("0", "16")

    # Single table
    json_single = json.loads(cli_helper.serialize_tables_to_json([table1]))
    self.assertIn("metrics_summary", json_single)
    self.assertEqual(
        json_single["metrics_summary"],
        [{"device_id": "0", "usage": "16"}],
    )

    # Table without title (defaults to 'data')
    table_no_title = rich_table.Table()
    table_no_title.add_column("Name")
    table_no_title.add_row("Alice")
    json_no_title = json.loads(
        cli_helper.serialize_tables_to_json([table_no_title])
    )
    self.assertEqual(json_no_title["data"], [{"name": "Alice"}])

    # Multiple tables with duplicate titles (converts to list of lists)
    table_dup1 = rich_table.Table(title="Duplicate")
    table_dup1.add_column("Key")
    table_dup1.add_row("k1")

    table_dup2 = rich_table.Table(title="Duplicate")
    table_dup2.add_column("Key")
    table_dup2.add_row("k2")

    table_dup3 = rich_table.Table(title="Duplicate")
    table_dup3.add_column("Key")
    table_dup3.add_row("k3")

    json_dup = json.loads(
        cli_helper.serialize_tables_to_json(
            [table_dup1, table_dup2, table_dup3]
        )
    )
    self.assertIsInstance(json_dup["duplicate"], list)
    self.assertLen(json_dup["duplicate"], 3)
    self.assertEqual(json_dup["duplicate"][0], [{"key": "k1"}])
    self.assertEqual(json_dup["duplicate"][1], [{"key": "k2"}])
    self.assertEqual(json_dup["duplicate"][2], [{"key": "k3"}])

    # Multiple tables with duplicate titles where first table is empty
    table_empty1 = rich_table.Table(title="Empty First")
    table_empty1.add_column("Col")

    table_nonempty2 = rich_table.Table(title="Empty First")
    table_nonempty2.add_column("Col")
    table_nonempty2.add_row("val")

    json_empty = json.loads(
        cli_helper.serialize_tables_to_json([table_empty1, table_nonempty2])
    )
    self.assertIsInstance(json_empty["empty_first"], list)
    self.assertLen(json_empty["empty_first"], 2)
    self.assertEqual(json_empty["empty_first"][0], [])
    self.assertEqual(json_empty["empty_first"][1], [{"col": "val"}])

  @mock.patch.object(cli_helper, "fetch_cli_version", return_value="1.0.0")
  @mock.patch.object(cli_helper, "fetch_libtpu_version", return_value="2.0.0")
  @mock.patch.object(
      cli_helper, "fetch_accelerator_topology", return_value="4xv4"
  )
  @mock.patch.object(device, "get_chip_owners")
  @mock.patch.object(device, "get_chips")
  @mock.patch.object(cli_helper, "get_process_name")
  def test_serialize_dashboard_to_json_and_csv(
      self,
      mock_proc_name,
      mock_get_chips,
      mock_get_owners,
      mock_topo,
      mock_libtpu_ver,
      mock_cli_ver,
  ):
    del mock_topo, mock_libtpu_ver, mock_cli_ver  # Unused mock injections.
    mock_chip = device.ChipInfo(
        base_addr="0000:00:04",
        cores={
            0: device.CoreInfo(
                core_index=0,
                device_id="0x005e",
                vfio_path="/dev/vfio/1",
                full_addr="0000:00:04.0",
            )
        },
    )
    mock_get_chips.return_value = [mock_chip]
    mock_get_owners.return_value = {"/dev/vfio/1": 1234}
    mock_proc_name.return_value = "python_train"

    status_list = [
        {
            "device_id": 0,
            "memory_used": 1024,
            "memory_total": 2048,
            "duty_cycle": 90.0,
            "hbm_bw_util": 55.0,
        },
        {
            "device_id": 1,
            "memory_used": None,
            "memory_total": None,
            "duty_cycle": None,
            "hbm_bw_util": None,
        },
    ]

    # JSON Serialization
    json_out = json.loads(
        cli_helper.serialize_dashboard_to_json(
            status_list, device.TpuChip.V4, 2
        )
    )
    self.assertEqual(json_out["tpu_info_version"], "1.0.0")
    self.assertEqual(json_out["libtpu_version"], "2.0.0")
    self.assertEqual(json_out["topology"], "4xv4")
    self.assertLen(json_out["devices"], 2)
    self.assertEqual(json_out["devices"][0]["process"]["pid"], 1234)
    self.assertEqual(
        json_out["devices"][0]["process"]["name"], "python_train"
    )
    self.assertIsNone(json_out["devices"][1]["process"])

    # CSV Serialization
    csv_out = cli_helper.serialize_dashboard_to_csv(
        status_list, device.TpuChip.V4
    )
    self.assertIn("device_id,device_path,memory_used_bytes", csv_out)
    self.assertIn("1234,python_train", csv_out)

  @mock.patch.object(device, "get_chip_owners")
  @mock.patch.object(device, "get_chips")
  @mock.patch.object(cli_helper, "get_process_name")
  def test_serialize_pmon_to_json_and_csv(
      self, mock_proc_name, mock_get_chips, mock_get_owners
  ):
    mock_chip = device.ChipInfo(
        base_addr="0000:00:04",
        cores={
            0: device.CoreInfo(
                core_index=0,
                device_id="0x005e",
                vfio_path="/dev/vfio/1",
                full_addr="0000:00:04.0",
            )
        },
    )
    mock_get_chips.return_value = [mock_chip]
    mock_get_owners.return_value = {"/dev/vfio/1": 5678}
    mock_proc_name.return_value = "jax_workload"

    status_list = [
        {"device_id": 0, "memory_used": 4096},
        {"device_id": 1, "memory_used": None},
    ]

    # JSON
    json_out = json.loads(
        cli_helper.serialize_pmon_to_json(status_list, device.TpuChip.V4)
    )
    self.assertLen(json_out, 2)
    self.assertEqual(json_out[0]["pid"], 5678)
    self.assertEqual(json_out[0]["process_name"], "jax_workload")
    self.assertIsNone(json_out[1]["pid"])

    # CSV
    csv_out = cli_helper.serialize_pmon_to_csv(status_list, device.TpuChip.V4)
    self.assertIn("5678,jax_workload,4096", csv_out)

  @mock.patch.object(device, "get_chips")
  def test_fetch_and_serialize_topo(self, mock_get_chips):
    mock_chip = device.ChipInfo(
        base_addr="0000:00:04",
        cores={
            0: device.CoreInfo(
                core_index=0,
                device_id="0x005e",
                vfio_path="/dev/vfio/1",
                full_addr="0000:00:04.0",
            )
        },
    )
    mock_get_chips.return_value = [mock_chip]

    table = cli_helper.fetch_topo_table()
    self.assertEqual(table.title, "TPU Topology & PCI/IOMMU Mappings")
    self.assertLen(table.rows, 1)

    json_out = json.loads(cli_helper.serialize_topo_to_json())
    self.assertEqual(
        json_out,
        [{
            "pci_addr_full": "0000:00:04.0",
            "pci_base_addr": "0000:00:04",
            "core_index": 0,
            "device_id": "0x005e",
            "vfio_path": "/dev/vfio/1",
        }],
    )

    csv_out = cli_helper.serialize_topo_to_csv()
    self.assertIn(
        "pci_addr_full,pci_base_addr,core_index,device_id,vfio_path", csv_out
    )
    self.assertIn("0000:00:04.0,0000:00:04,0,0x005e,/dev/vfio/1", csv_out)

  def test_serialize_metrics_list_to_json_and_csv(self):
    json_out = json.loads(cli_helper.serialize_metrics_list_to_json())
    self.assertIn("runtime", json_out)
    self.assertTrue(
        any(m["name"] == "hbm_usage" for m in json_out["runtime"])
    )

    csv_out = cli_helper.serialize_metrics_list_to_csv()
    self.assertIn("group,name,description", csv_out)
    self.assertIn("runtime,hbm_usage,", csv_out)

  @mock.patch.object(
      device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
  )
  def test_batch_scrape_prometheus_and_render_tables(self, mock_local_chips):
    del mock_local_chips  # Unused mock parameter.
    with mock.patch.object(
        registry.MetricRegistry,
        "_batch_scrape_prometheus",
        return_value=["mock_rendered_table"],
    ) as mock_batch:
      res = cli_helper._fetch_prometheus_metrics_batch(
          ["orbax_write_size"],
          "Orbax",
          9431,
          "ORBAX_PROMETHEUS_PORT",
          "ORBAX_METRICS",
      )
      self.assertEqual(res, ["mock_rendered_table"])
      mock_batch.assert_called_once()

  def test_fetch_accelerator_topology_variations(self):
    # Case 1: TPU_CHIPS_PER_HOST_BOUNDS in os.environ
    with mock.patch.dict(
        os.environ, {"TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1"}
    ):
      self.assertEqual(cli_helper.fetch_accelerator_topology(), "2,2,1")

    # Case 2: chip_type is None
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(
          device, "get_local_chips", return_value=(None, 0)
      ):
        self.assertEqual(cli_helper.fetch_accelerator_topology(), "N/A")

    # Case 3: count == 0
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(
          device, "get_local_chips", return_value=(device.TpuChip.V4, 0)
      ):
        self.assertEqual(cli_helper.fetch_accelerator_topology(), "N/A")

    # Case 4: Normal
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(
          device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
      ):
        self.assertEqual(cli_helper.fetch_accelerator_topology(), "4xv4")

  def test_ascii_bar_gauge_clamping(self):
    self.assertEqual(
        cli_helper.ascii_bar_gauge(-10.0, width=10), "[..........] 0.0%"
    )
    self.assertEqual(
        cli_helper.ascii_bar_gauge(150.0, width=10), "[||||||||||] 100.0%"
    )
    self.assertEqual(
        cli_helper.ascii_bar_gauge(50.0, width=10), "[|||||.....] 50.0%"
    )

  def test_fetch_consolidated_device_status_panel_variations(self):
    # Panel without .plain on renderable
    mock_panel1 = mock.MagicMock(spec=panel.Panel)
    mock_panel1.renderable = "String error message"
    with mock.patch.object(
        cli_helper, "get_device_usage", return_value=mock_panel1
    ):
      status_list, error_msg = cli_helper.fetch_consolidated_device_status(
          device.TpuChip.V4, 2
      )
      self.assertEqual(error_msg, "String error message")
      self.assertLen(status_list, 2)

    # Panel without renderable attribute
    mock_panel2 = mock.MagicMock(spec=panel.Panel)
    del mock_panel2.renderable
    with mock.patch.object(
        cli_helper, "get_device_usage", return_value=mock_panel2
    ):
      status_list, error_msg = cli_helper.fetch_consolidated_device_status(
          device.TpuChip.V4, 1
      )
      self.assertEqual(error_msg, "Failed to fetch device usage.")
      self.assertLen(status_list, 1)

  def test_get_owner_for_device_fallbacks(self):
    mock_chip = device.ChipInfo(
        base_addr="0000:00:04",
        cores={
            0: device.CoreInfo(
                core_index=0,
                device_id="0x005e",
                vfio_path="/dev/vfio/1",
                full_addr="0000:00:04.0",
            ),
            1: device.CoreInfo(
                core_index=1,
                device_id="0x005e",
                vfio_path="/dev/vfio/2",
                full_addr="0000:00:04.1",
            ),
        },
    )

    # Match on full_addr
    pid, path = cli_helper.get_owner_for_device(
        0, device.TpuChip.V4, [mock_chip], {"0000:00:04.0": 999}
    )
    self.assertEqual(pid, 999)
    self.assertEqual(path, "0000:00:04.0")

    # Match on another core in the chip
    pid, path = cli_helper.get_owner_for_device(
        0, device.TpuChip.V4, [mock_chip], {"/dev/vfio/2": 888}
    )
    self.assertEqual(pid, 888)
    self.assertEqual(path, "/dev/vfio/2")

    # No match
    pid, path = cli_helper.get_owner_for_device(
        0, device.TpuChip.V4, [mock_chip], {}
    )
    self.assertIsNone(pid)

  def test_render_consolidated_device_status_table_zero_memory(self):
    """Verifies total memory of 0 or None renders N/A without division error."""
    status_list = [
        {
            "device_id": 0,
            "memory_used": 0,
            "memory_total": 0,
            "hbm_bw_util": None,
            "duty_cycle": None,
        },
        {
            "device_id": 1,
            "memory_used": 1024 * 1024 * 1024,
            "memory_total": 4 * 1024 * 1024 * 1024,
            "hbm_bw_util": 50.0,
            "duty_cycle": 25.0,
        },
    ]
    table = cli_helper.render_consolidated_device_status_table(status_list)
    self.assertIsInstance(table, rich_table.Table)
    output = io.StringIO()
    console_obj = console.Console(file=output, width=120)
    console_obj.print(table)
    text_out = output.getvalue()
    self.assertIn("N/A", text_out)
    self.assertIn("1.00 GiB / 4.00 GiB", text_out)

  def test_render_consolidated_device_status_table_negative_and_invalid_values(
      self,
  ):
    """Verifies negative or non-numeric values render N/A safely without crashing."""
    status_list = [
        {
            "device_id": 0,
            "memory_used": -1,
            "memory_total": 4 * 1024 * 1024 * 1024,
            "hbm_bw_util": -5.0,
            "duty_cycle": -10.0,
        },
        {
            "device_id": 1,
            "memory_used": "bad_value",
            "memory_total": 4 * 1024 * 1024 * 1024,
            "hbm_bw_util": "bad_bw",
            "duty_cycle": "bad_dc",
        },
    ]
    table = cli_helper.render_consolidated_device_status_table(status_list)
    self.assertIsInstance(table, rich_table.Table)
    output = io.StringIO()
    console_obj = console.Console(file=output, width=120)
    console_obj.print(table)
    text_out = output.getvalue()
    self.assertIn("N/A", text_out)
    self.assertNotIn("-0.00 GiB", text_out)


if __name__ == "__main__":
  absltest.main()
