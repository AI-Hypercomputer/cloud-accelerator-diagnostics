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

"""Unit tests for domain telemetry metric providers."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tpu_info import cli_helper
from tpu_info import device
from tpu_info import metrics
from tpu_info import registry
from tpu_info.providers import execution_provider
from tpu_info.providers import network_provider
from tpu_info.providers import orbax_provider
from tpu_info.providers import pygrain_provider
from tpu_info.providers import runtime_provider
from prometheus_client.core import Metric
from rich import table as rich_table


class ProvidersTest(parameterized.TestCase):
  """Tests for registered domain metric providers metadata and descriptors."""

  def setUp(self):
    """Sets up provider registry and registers cleanup."""
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()
    orig_descriptors = dict(reg._descriptors)

    def cleanup():
      """Restores original metric descriptors in the registry."""
      registry.MetricRegistry()._descriptors = dict(orig_descriptors)

    self.addCleanup(cleanup)

  def test_total_metrics_count(self):
    """Tests that all domain providers register 57 telemetry metrics."""
    reg = registry.MetricRegistry()
    self.assertLen(reg.get_all_descriptors(), 57)
    self.assertLen(reg.get_all_metric_names(), 57)

  def test_all_groups(self):
    """Tests that all expected provider domain groups are registered."""
    reg = registry.MetricRegistry()
    self.assertEqual(
        set(reg.get_all_groups()),
        {"runtime", "execution", "network", "orbax", "pygrain"},
    )

  @parameterized.named_parameters(
      dict(testcase_name="runtime", group="runtime", expected_count=5),
      dict(testcase_name="execution", group="execution", expected_count=6),
      dict(testcase_name="network", group="network", expected_count=8),
      dict(testcase_name="orbax", group="orbax", expected_count=33),
      dict(testcase_name="pygrain", group="pygrain", expected_count=5),
  )
  def test_group_counts(self, group, expected_count):
    """Tests descriptor count for each domain metric provider group."""
    reg = registry.MetricRegistry()
    self.assertLen(reg.get_descriptors_by_group(group), expected_count)

  def test_descriptor_attributes(self):
    """Tests descriptor metadata and attributes for specific metrics."""
    reg = registry.MetricRegistry()
    desc = reg.get_descriptor("hbm_usage")
    self.assertIsNotNone(desc)
    self.assertEqual(desc.name, "hbm_usage")
    self.assertEqual(desc.group, "runtime")
    self.assertEqual(
        desc.description, "TPU HBM memory usage per device (GiB)"
    )
    self.assertTrue(callable(desc.handler))

    net_desc = reg.get_descriptor("buffer_transfer_latency")
    self.assertIsNotNone(net_desc)
    self.assertEqual(net_desc.allowed_filters, {"percentile"})

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  def test_batch_scraping_orbax_providers(self, mock_scrape):
    """Tests batch scraping execution for registered Orbax provider metrics."""
    mock_scrape.return_value = []
    reg = registry.MetricRegistry()
    reg.render_metrics(
        [("orbax_write_size", None), ("orbax_read_start_count", None)],
        chip_type=None,
        count=1,
    )
    mock_scrape.assert_called_once_with(metrics.ORBAX_PROMETHEUS_DEFAULT_PORT)


class RuntimeProviderTest(parameterized.TestCase):
  """Tests for runtime domain metric provider handlers."""

  def setUp(self):
    """Ensures provider modules are loaded before running tests."""
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @mock.patch.object(cli_helper, "get_hbm_usage_table", autospec=True)
  def test_hbm_usage_handler(self, mock_fn):
    """Tests hbm_usage handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="HBM Usage")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("hbm_usage")
    self.assertIsNotNone(desc)
    result = desc.handler(chip_type=device.TpuChip.V4, count=4)
    mock_fn.assert_called_once_with(device.TpuChip.V4, 4)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_duty_cycle_table", autospec=True)
  def test_duty_cycle_percent_handler(self, mock_fn):
    """Tests duty_cycle_percent handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="Duty Cycle")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("duty_cycle_percent")
    self.assertIsNotNone(desc)
    result = desc.handler(chip_type=device.TpuChip.V4, count=4)
    mock_fn.assert_called_once_with(device.TpuChip.V4, 4)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "TensorCoreUtilizationTable", autospec=True)
  def test_tensorcore_utilization_handler(self, mock_cls):
    """Tests tensorcore_utilization handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="TC Util")
    mock_instance = mock_cls.return_value
    mock_instance.render.return_value = mock_table
    desc = registry.MetricRegistry().get_descriptor("tensorcore_utilization")
    self.assertIsNotNone(desc)
    result = desc.handler(count=8)
    mock_instance.render.assert_called_once_with(8)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(
      cli_helper, "get_runtime_hbm_utilization_table", autospec=True
  )
  def test_runtime_hbm_utilization_handler(self, mock_fn):
    """Tests runtime_hbm_utilization handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="Runtime HBM Util")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("runtime_hbm_utilization")
    self.assertIsNotNone(desc)
    result = desc.handler(chip_type=device.TpuChip.V4, count=2)
    mock_fn.assert_called_once_with(device.TpuChip.V4, 2)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(
      cli_helper, "get_tensorcore_idle_duration_table", autospec=True
  )
  def test_tensorcore_idle_duration_handler(self, mock_fn):
    """Tests tensorcore_idle_duration handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="TC Idle")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("tensorcore_idle_duration")
    self.assertIsNotNone(desc)
    result = desc.handler(chip_type=device.TpuChip.V4, count=2)
    mock_fn.assert_called_once_with(device.TpuChip.V4, 2)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_hbm_usage_table", autospec=True)
  def test_runtime_render_metrics_dispatch(self, mock_fn):
    """Tests render_metrics routes runtime metrics through handlers."""
    mock_table = rich_table.Table(title="HBM Table")
    mock_fn.return_value = [mock_table]
    renderables = registry.MetricRegistry().render_metrics(
        [("hbm_usage", None)],
        chip_type=device.TpuChip.V4,
        count=4,
    )
    mock_fn.assert_called_once_with(device.TpuChip.V4, 4)
    self.assertEqual(renderables, [mock_table])


class ExecutionProviderTest(parameterized.TestCase):
  """Tests for execution domain metric provider handlers."""

  def setUp(self):
    """Ensures provider modules are loaded before running tests."""
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @mock.patch.object(cli_helper, "get_hlo_queue_size_table", autospec=True)
  def test_hlo_queue_size_handler(self, mock_fn):
    """Tests hlo_queue_size handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="HLO Queue")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("hlo_queue_size")
    self.assertIsNotNone(desc)
    result = desc.handler(chip_type=device.TpuChip.V4, count=4)
    mock_fn.assert_called_once_with(device.TpuChip.V4, 4)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_hlo_exec_timing_table", autospec=True)
  def test_hlo_exec_timing_handler(self, mock_fn):
    """Tests hlo_exec_timing handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="HLO Timing")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("hlo_exec_timing")
    self.assertIsNotNone(desc)
    result = desc.handler(chip_type=device.TpuChip.V4, count=4)
    mock_fn.assert_called_once_with(device.TpuChip.V4, 4)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_tpuz_core_state", autospec=True)
  def test_core_state_handler(self, mock_fn):
    """Tests core_state handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="Core State")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("core_state")
    self.assertIsNotNone(desc)
    result = desc.handler()
    mock_fn.assert_called_once()
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_tpuz_sequencer_state", autospec=True)
  def test_sequencer_state_handler(self, mock_fn):
    """Tests sequencer_state handler dispatches with detailed_info=False."""
    mock_table = rich_table.Table(title="Sequencer State")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("sequencer_state")
    self.assertIsNotNone(desc)
    result = desc.handler()
    mock_fn.assert_called_once_with(detailed_info=False)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_tpuz_sequencer_state", autospec=True)
  def test_sequencer_state_detailed_handler(self, mock_fn):
    """Tests sequencer_state_detailed handler dispatches with detailed_info=True."""
    mock_table = rich_table.Table(title="Detailed Sequencer")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("sequencer_state_detailed")
    self.assertIsNotNone(desc)
    result = desc.handler()
    mock_fn.assert_called_once_with(detailed_info=True)
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_tpuz_queued_programs", autospec=True)
  def test_queued_programs_handler(self, mock_fn):
    """Tests queued_programs handler dispatches to cli_helper."""
    mock_table = rich_table.Table(title="Queued Programs")
    mock_fn.return_value = [mock_table]
    desc = registry.MetricRegistry().get_descriptor("queued_programs")
    self.assertIsNotNone(desc)
    result = desc.handler()
    mock_fn.assert_called_once()
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "get_tpuz_core_state", autospec=True)
  def test_execution_render_metrics_dispatch(self, mock_fn):
    """Tests render_metrics routes execution metrics through handlers."""
    mock_table = rich_table.Table(title="Core State")
    mock_fn.return_value = [mock_table]
    renderables = registry.MetricRegistry().render_metrics(
        [("core_state", None)],
        chip_type=None,
        count=0,
    )
    mock_fn.assert_called_once()
    self.assertEqual(renderables, [mock_table])


class NetworkProviderTest(parameterized.TestCase):
  """Tests for network domain metric provider handlers."""

  def setUp(self):
    """Ensures provider modules are loaded before running tests."""
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @parameterized.named_parameters(
      (m, m) for m in network_provider._NETWORK_METRICS
  )
  @mock.patch.object(cli_helper, "TransferLatencyTables", autospec=True)
  def test_network_handlers_dispatch(self, metric_name, mock_cls):
    """Tests all 8 network metric handlers dispatch to TransferLatencyTables."""
    mock_table = rich_table.Table(title=f"Table {metric_name}")
    mock_instance = mock_cls.return_value
    mock_instance.render.return_value = mock_table

    desc = registry.MetricRegistry().get_descriptor(metric_name)
    self.assertIsNotNone(desc)
    result = desc.handler(filters={"percentile": "p99"})
    mock_instance.render.assert_called_once_with(
        metric_name, {"percentile": "p99"}
    )
    self.assertEqual(result, [mock_table])

  @mock.patch.object(cli_helper, "TransferLatencyTables", autospec=True)
  def test_network_render_metrics_dispatch(self, mock_cls):
    """Tests render_metrics routes network metrics with filters."""
    mock_table = rich_table.Table(title="Latency Table")
    mock_instance = mock_cls.return_value
    mock_instance.render.return_value = mock_table

    renderables = registry.MetricRegistry().render_metrics(
        [("buffer_transfer_latency", {"percentile": "p50"})],
        chip_type=None,
        count=0,
    )
    mock_instance.render.assert_called_once_with(
        "buffer_transfer_latency", {"percentile": "p50"}
    )
    self.assertEqual(renderables, [mock_table])


class OrbaxProviderTest(parameterized.TestCase):
  """Tests for Orbax domain metric provider handlers."""

  def setUp(self):
    """Ensures provider modules are loaded before running tests."""
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @parameterized.named_parameters(
      (m, m) for m in metrics.ORBAX_SHORT_TO_LONG_MAP
  )
  @mock.patch.object(
      cli_helper, "get_prometheus_metric_table_from_families", autospec=True
  )
  def test_orbax_fetcher_with_pre_scraped_metrics(self, metric_name, mock_fn):
    """Tests Orbax metric handler with pre_scraped_metrics."""
    mock_table = rich_table.Table(title=metric_name)
    mock_fn.return_value = [mock_table]
    mock_family = mock.create_autospec(Metric, instance=True)

    desc = registry.MetricRegistry().get_descriptor(metric_name)
    self.assertIsNotNone(desc)
    result = desc.handler(
        pre_scraped_metrics=[mock_family],
        skip_if_missing=True,
    )
    mock_fn.assert_called_once_with(
        [mock_family],
        metric_name,
        "Orbax",
        skip_if_missing=True,
    )
    self.assertEqual(result, [mock_table])

  @parameterized.named_parameters(
      (m, m) for m in metrics.ORBAX_SHORT_TO_LONG_MAP
  )
  @mock.patch.object(
      cli_helper, "get_prometheus_metric_table", autospec=True
  )
  def test_orbax_fetcher_without_pre_scraped_metrics(
      self, metric_name, mock_fn
  ):
    """Tests Orbax metric handler without pre_scraped_metrics."""
    mock_table = rich_table.Table(title=metric_name)
    mock_fn.return_value = [mock_table]

    desc = registry.MetricRegistry().get_descriptor(metric_name)
    self.assertIsNotNone(desc)
    result = desc.handler(pre_scraped_metrics=None)
    mock_fn.assert_called_once_with(metric_name)
    self.assertEqual(result, [mock_table])

  @parameterized.named_parameters(
      (m, m) for m in metrics.ORBAX_SHORT_TO_LONG_MAP
  )
  @mock.patch.object(
      cli_helper, "get_prometheus_metric_table", autospec=True
  )
  def test_orbax_module_getter_functions(self, metric_name, mock_fn):
    """Tests exported get_<metric_name> functions in orbax_provider."""
    mock_table = rich_table.Table(title=metric_name)
    mock_fn.return_value = [mock_table]
    getter_fn = getattr(orbax_provider, f"get_{metric_name}", None)
    self.assertIsNotNone(getter_fn)
    self.assertTrue(callable(getter_fn))
    result = getter_fn()
    mock_fn.assert_called_once_with(metric_name)
    self.assertEqual(result, [mock_table])


class PyGrainProviderTest(parameterized.TestCase):
  """Tests for PyGrain domain metric provider handlers."""

  def setUp(self):
    """Ensures provider modules are loaded before running tests."""
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @parameterized.named_parameters(
      (m, m) for m in metrics.PYGRAIN_SHORT_TO_LONG_MAP
  )
  @mock.patch.object(
      cli_helper, "get_prometheus_metric_table_from_families", autospec=True
  )
  def test_pygrain_fetcher_with_pre_scraped_metrics(
      self, metric_name, mock_fn
  ):
    """Tests PyGrain metric handler with pre_scraped_metrics."""
    mock_table = rich_table.Table(title=metric_name)
    mock_fn.return_value = [mock_table]
    mock_family = mock.create_autospec(Metric, instance=True)

    desc = registry.MetricRegistry().get_descriptor(metric_name)
    self.assertIsNotNone(desc)
    result = desc.handler(
        pre_scraped_metrics=[mock_family],
        skip_if_missing=True,
    )
    mock_fn.assert_called_once_with(
        [mock_family],
        metric_name,
        "Pygrain",
        skip_if_missing=True,
    )
    self.assertEqual(result, [mock_table])

  @parameterized.named_parameters(
      (m, m) for m in metrics.PYGRAIN_SHORT_TO_LONG_MAP
  )
  @mock.patch.object(
      cli_helper, "get_prometheus_metric_table", autospec=True
  )
  def test_pygrain_fetcher_without_pre_scraped_metrics(
      self, metric_name, mock_fn
  ):
    """Tests PyGrain metric handler without pre_scraped_metrics."""
    mock_table = rich_table.Table(title=metric_name)
    mock_fn.return_value = [mock_table]

    desc = registry.MetricRegistry().get_descriptor(metric_name)
    self.assertIsNotNone(desc)
    result = desc.handler(pre_scraped_metrics=None)
    mock_fn.assert_called_once_with(metric_name)
    self.assertEqual(result, [mock_table])

  @parameterized.named_parameters(
      (m, m) for m in metrics.PYGRAIN_SHORT_TO_LONG_MAP
  )
  @mock.patch.object(
      cli_helper, "get_prometheus_metric_table", autospec=True
  )
  def test_pygrain_module_getter_functions(self, metric_name, mock_fn):
    """Tests exported get_<metric_name> functions in pygrain_provider."""
    mock_table = rich_table.Table(title=metric_name)
    mock_fn.return_value = [mock_table]
    getter_fn = getattr(pygrain_provider, f"get_{metric_name}", None)
    self.assertIsNotNone(getter_fn)
    self.assertTrue(callable(getter_fn))
    result = getter_fn()
    mock_fn.assert_called_once_with(metric_name)
    self.assertEqual(result, [mock_table])


class RuntimeProviderRawTest(parameterized.TestCase):
  """Tests for runtime provider raw data handlers."""

  def setUp(self):
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @mock.patch.object(metrics, "get_chip_usage", autospec=True)
  @mock.patch.object(metrics, "get_chip_usage_new", autospec=True)
  def test_hbm_usage_raw(self, mock_usage_new, mock_usage):
    usage_mock = metrics.Usage(
        device_id=0,
        memory_usage=1024,
        total_memory=2048,
        duty_cycle_pct=50.0,
    )
    mock_usage.return_value = [usage_mock]
    mock_usage_new.return_value = [usage_mock]

    # Non-V7X
    res = runtime_provider._hbm_usage_raw(chip_type=device.TpuChip.V4)
    mock_usage.assert_called_once_with(device.TpuChip.V4)
    self.assertEqual(res, [usage_mock])

    # V7X
    res_v7x = runtime_provider._hbm_usage_raw(chip_type=device.TpuChip.V7X)
    mock_usage_new.assert_called_once_with(device.TpuChip.V7X)
    self.assertEqual(res_v7x, [usage_mock])

  @mock.patch.object(metrics, "get_chip_usage", autospec=True)
  @mock.patch.object(metrics, "get_chip_usage_new", autospec=True)
  def test_duty_cycle_raw(self, mock_usage_new, mock_usage):
    usage_mock_v4 = metrics.Usage(
        device_id=0,
        memory_usage=1024,
        total_memory=2048,
        duty_cycle_pct=50.0,
    )
    usage_mock_v7x = metrics.Usage(
        device_id=0,
        memory_usage=2048,
        total_memory=4096,
        duty_cycle_pct=85.0,
    )
    mock_usage.return_value = [usage_mock_v4]
    mock_usage_new.return_value = [usage_mock_v7x]

    # Non-V7X
    res = runtime_provider._duty_cycle_raw(chip_type=device.TpuChip.V4)
    mock_usage.assert_called_once_with(device.TpuChip.V4)
    mock_usage_new.assert_not_called()
    self.assertEqual(res, [{"device_id": 0, "duty_cycle_pct": 50.0}])

    # V7X
    res_v7x = runtime_provider._duty_cycle_raw(chip_type=device.TpuChip.V7X)
    mock_usage_new.assert_called_once_with(device.TpuChip.V7X)
    self.assertEqual(res_v7x, [{"device_id": 0, "duty_cycle_pct": 85.0}])

  def test_tc_util_raw_no_sdk(self):
    with mock.patch.object(
        cli_helper, "ensure_libtpu_initialized"
    ) as mock_init:
      with mock.patch.object(cli_helper, "libtpu_sdk", None):
        self.assertEqual(runtime_provider._tc_util_raw(), [])
        mock_init.assert_called_once()

  def test_tc_util_raw_with_sdk(self):
    mock_sdk = mock.MagicMock(spec=["tpumonitoring", "monitoring"])
    mock_metric = mock.MagicMock()
    mock_metric.data.return_value = ["10.0", "20.5"]
    mock_sdk.tpumonitoring.get_metric.return_value = mock_metric
    mock_sdk.monitoring = None
    with mock.patch.object(
        cli_helper, "ensure_libtpu_initialized"
    ) as mock_init:
      with mock.patch.object(cli_helper, "libtpu_sdk", mock_sdk):
        res = runtime_provider._tc_util_raw()
        self.assertEqual(
            res,
            [
                {"core_id": 0, "utilization": 10.0},
                {"core_id": 1, "utilization": 20.5},
            ],
        )
        mock_init.assert_called_once()

  def test_tc_util_raw_sdk_without_monitoring(self):
    mock_sdk = mock.MagicMock(spec=[])
    with mock.patch.object(
        cli_helper, "ensure_libtpu_initialized"
    ) as mock_init:
      with mock.patch.object(cli_helper, "libtpu_sdk", mock_sdk):
        res = runtime_provider._tc_util_raw()
        self.assertEqual(res, [])
        mock_init.assert_called_once()

  @mock.patch.object(metrics, "get_runtime_hbm_utilization", autospec=True)
  def test_runtime_hbm_util_raw(self, mock_fn):
    mock_fn.return_value = [(0, 42.0), (1, 88.5)]
    res = runtime_provider._runtime_hbm_util_raw()
    self.assertEqual(
        res,
        [
            {"device_id": 0, "utilization": 42.0},
            {"device_id": 1, "utilization": 88.5},
        ],
    )

  @mock.patch.object(metrics, "get_tensorcore_idle_duration", autospec=True)
  def test_tc_idle_raw(self, mock_fn):
    mock_fn.return_value = [(0, 12.3)]
    res = runtime_provider._tc_idle_raw()
    self.assertEqual(
        res,
        [{"device_id": 0, "idle_duration_seconds": 12.3}],
    )


class ExecutionProviderRawTest(parameterized.TestCase):
  """Tests for execution provider raw data handlers."""

  def setUp(self):
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @mock.patch.object(metrics, "get_hlo_queue_size", autospec=True)
  def test_hlo_queue_size_raw(self, mock_fn):
    mock_fn.return_value = [1, 2, 3]
    res = execution_provider._hlo_queue_size_raw(chip_type=device.TpuChip.V4)
    mock_fn.assert_called_once_with(device.TpuChip.V4)
    self.assertEqual(res, [1, 2, 3])

  @mock.patch.object(metrics, "get_hlo_exec_timing", autospec=True)
  def test_hlo_exec_timing_raw(self, mock_fn):
    mock_fn.return_value = [0.1, 0.2]
    res = execution_provider._hlo_exec_timing_raw(chip_type=device.TpuChip.V4)
    mock_fn.assert_called_once_with(device.TpuChip.V4)
    self.assertEqual(res, [0.1, 0.2])

  @mock.patch.object(metrics, "get_tpuz_info", autospec=True)
  def test_core_state_raw(self, mock_fn):
    mock_core = metrics.CoreState(
        global_core_id=1,
        chip_id=0,
        core_on_chip_index=0,
        core_type="TPU_CORE",
        xdb_server=True,
        sequencer_states=[],
        program_fingerprint="fp123",
        error_message="",
        queued_programs=[],
    )
    mock_fn.return_value = [mock_core]
    res = execution_provider._core_state_raw()
    mock_fn.assert_called_once_with(include_hlo_info=False)
    self.assertEqual(
        res,
        [{
            "chip_id": 0,
            "global_core_id": 1,
            "core_type": "TPU_CORE",
            "xdb_server": True,
        }],
    )

  @mock.patch.object(metrics, "get_tpuz_info", autospec=True)
  def test_sequencer_state_raw(self, mock_fn):
    mock_seq = metrics.SequencerState(
        sequencer_type="SEQ_MAIN",
        sequencer_index=0,
        pc=100,
        tag=0,
        tracemark=200,
        program_id=300,
        run_id=400,
    )
    mock_core = metrics.CoreState(
        global_core_id=1,
        chip_id=0,
        core_on_chip_index=0,
        core_type="TPU_CORE",
        xdb_server=True,
        sequencer_states=[mock_seq],
        program_fingerprint="fp123",
        error_message="",
        queued_programs=[],
    )
    mock_fn.return_value = [mock_core]
    res = execution_provider._sequencer_state_raw()
    mock_fn.assert_called_once_with(include_hlo_info=False)
    self.assertEqual(
        res,
        [{
            "chip_id": 0,
            "global_core_id": 1,
            "program_counter": 100,
            "tracemark": 200,
            "program_id": 300,
            "run_id": 400,
            "sequence_type": "SEQ_MAIN",
        }],
    )

  @mock.patch.object(metrics, "get_tpuz_info", autospec=True)
  def test_sequencer_state_detailed_raw(self, mock_fn):
    mock_seq = metrics.SequencerState(
        sequencer_type="SEQ_MAIN",
        sequencer_index=0,
        pc=100,
        tag=0,
        tracemark=200,
        program_id=300,
        run_id=400,
        hlo_location="location_foo",
        hlo_detailed_info="detailed_bar",
    )
    mock_core = metrics.CoreState(
        global_core_id=1,
        chip_id=0,
        core_on_chip_index=0,
        core_type="TPU_CORE",
        xdb_server=True,
        sequencer_states=[mock_seq],
        program_fingerprint="fp123",
        error_message="core error msg",
        queued_programs=[],
    )
    mock_fn.return_value = [mock_core]
    res = execution_provider._sequencer_state_detailed_raw()
    mock_fn.assert_called_once_with(include_hlo_info=True)
    self.assertEqual(
        res,
        [{
            "chip_id": 0,
            "global_core_id": 1,
            "program_counter": 100,
            "tracemark": 200,
            "program_id": 300,
            "run_id": 400,
            "sequence_type": "SEQ_MAIN",
            "core_error": "core error msg",
            "hlo_location": "location_foo",
            "hlo_details": "detailed_bar",
        }],
    )

  @mock.patch.object(metrics, "get_tpuz_info", autospec=True)
  def test_queued_programs_raw(self, mock_fn):
    mock_prog = metrics.QueuedProgram(
        run_id=10,
        launch_id=20,
        program_fingerprint="fp999",
    )
    mock_core = metrics.CoreState(
        global_core_id=1,
        chip_id=0,
        core_on_chip_index=0,
        core_type="TPU_CORE",
        xdb_server=True,
        sequencer_states=[],
        program_fingerprint="fp123",
        error_message="",
        queued_programs=[mock_prog],
    )
    mock_fn.return_value = [mock_core]
    res = execution_provider._queued_programs_raw()
    mock_fn.assert_called_once_with(include_hlo_info=False)
    self.assertEqual(
        res,
        [{
            "chip_id": 0,
            "global_core_id": 1,
            "run_id": 10,
            "launch_id": 20,
            "program_fingerprint": "fp999",
        }],
    )


class NetworkProviderRawTest(parameterized.TestCase):
  """Tests for network provider raw data handlers."""

  def setUp(self):
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()

  @mock.patch.object(metrics, "get_transfer_latency", autospec=True)
  def test_latency_raw_handler(self, mock_fn):
    mock_fn.return_value = [{"latency": 10.5}]
    desc = registry.MetricRegistry().get_descriptor("buffer_transfer_latency")
    self.assertIsNotNone(desc)
    self.assertIsNotNone(desc.raw_handler)
    res = desc.raw_handler()
    mock_fn.assert_called_with("buffer_transfer_latency", filters=None)
    self.assertEqual(res, [{"latency": 10.5}])

    res_with_filters = desc.raw_handler(filters={"percentile": "p99"})
    mock_fn.assert_called_with(
        "buffer_transfer_latency", filters={"percentile": "p99"}
    )
    self.assertEqual(res_with_filters, [{"latency": 10.5}])


if __name__ == "__main__":
  absltest.main()
