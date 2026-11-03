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

"""Unit tests for the central telemetry metric registry."""

import collections
import dataclasses
import enum
import importlib
import os
import sys
import threading
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tpu_info import device
from tpu_info import metrics
from tpu_info import registry
from prometheus_client.core import Metric
from prometheus_client.core import Sample
from rich import panel
from rich import table as rich_table


class MetricRegistryTest(parameterized.TestCase):
  """Tests for the central metric registry."""

  def setUp(self):
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()
    orig_descriptors = dict(reg._descriptors)
    orig_loaded = reg._loaded

    def cleanup():
      active_reg = registry.MetricRegistry()
      active_reg._descriptors = dict(orig_descriptors)
      active_reg._loaded = orig_loaded

    self.addCleanup(cleanup)

  def test_singleton_instance(self):
    """Tests that MetricRegistry implements a singleton pattern."""
    reg1 = registry.MetricRegistry()
    reg2 = registry.MetricRegistry()

    self.assertIs(reg1, reg2)

  def test_singleton_reset(self):
    """Tests that resetting the singleton re-creates the registry."""
    reg1 = registry.MetricRegistry()
    registry.MetricRegistry._reset()
    reg2 = registry.MetricRegistry()

    self.assertIsNot(reg1, reg2)

  def test_ensure_providers_loaded_clears_cached_modules(self):
    """Tests that provider submodules are deleted from sys.modules on load."""
    metric_registry = registry.MetricRegistry()
    metric_registry._loaded = False
    package_name = (
        registry.__package__
        or "cloud_accelerator_diagnostics.tpu_info.tpu_info"
    )
    provider_prefix = f"{package_name}.providers"
    fake_mod_parent = provider_prefix
    fake_mod_sub = f"{provider_prefix}.runtime"
    unrelated_mod = f"{package_name}.device"

    saved_modules = {
        k: v
        for k, v in sys.modules.items()
        if k == provider_prefix or k.startswith(provider_prefix + ".")
    }

    sys.modules[fake_mod_parent] = mock.MagicMock()
    sys.modules[fake_mod_sub] = mock.MagicMock()

    def cleanup_sys_modules():
      for k in list(sys.modules):
        if k == provider_prefix or k.startswith(provider_prefix + "."):
          sys.modules.pop(k, None)
      sys.modules.update(saved_modules)

    self.addCleanup(cleanup_sys_modules)

    with mock.patch.object(
        importlib, "import_module", autospec=True
    ) as mock_import:
      metric_registry._ensure_providers_loaded()
      mock_import.assert_called_once_with(provider_prefix)

    self.assertNotIn(fake_mod_parent, sys.modules)
    self.assertNotIn(fake_mod_sub, sys.modules)
    self.assertIn(unrelated_mod, sys.modules)
    self.assertTrue(metric_registry._loaded)

  def test_ensure_providers_loaded_open_source_package(self):
    """Tests that provider loading uses __package__ when in open-source layout."""
    metric_registry = registry.MetricRegistry()
    metric_registry._loaded = False

    fake_os_provider = "tpu_info.providers"
    fake_os_sub = "tpu_info.providers.runtime"
    sys.modules[fake_os_provider] = mock.MagicMock()
    sys.modules[fake_os_sub] = mock.MagicMock()

    def cleanup_sys_modules():
      sys.modules.pop(fake_os_provider, None)
      sys.modules.pop(fake_os_sub, None)

    self.addCleanup(cleanup_sys_modules)

    with mock.patch.object(registry, "__package__", "tpu_info"):
      with mock.patch.object(
          importlib, "import_module", autospec=True
      ) as mock_import:
        metric_registry._ensure_providers_loaded()
        mock_import.assert_called_once_with("tpu_info.providers")

    self.assertNotIn(fake_os_provider, sys.modules)
    self.assertNotIn(fake_os_sub, sys.modules)
    self.assertTrue(metric_registry._loaded)

  def test_ensure_providers_loaded_already_loaded(self):
    """Tests that provider loading is skipped if already loaded."""
    metric_registry = registry.MetricRegistry()
    metric_registry._loaded = True

    with mock.patch.object(
        importlib, "import_module", autospec=True
    ) as mock_import:
      metric_registry._ensure_providers_loaded()
      mock_import.assert_not_called()

  def test_ensure_providers_loaded_package_none_fallback(self):
    """Tests that provider loading falls back to default package when __package__ is None."""
    metric_registry = registry.MetricRegistry()
    metric_registry._loaded = False

    with mock.patch.object(registry, "__package__", None):
      with mock.patch.object(
          importlib, "import_module", autospec=True
      ) as mock_import:
        metric_registry._ensure_providers_loaded()
        mock_import.assert_called_once_with("tpu_info.providers")

    self.assertTrue(metric_registry._loaded)

  @parameterized.named_parameters(
      dict(
          testcase_name="module_not_found",
          error_cls=ModuleNotFoundError,
      ),
      dict(
          testcase_name="import_error",
          error_cls=ImportError,
      ),
  )
  def test_ensure_providers_loaded_import_error(self, error_cls):
    """Tests that missing providers module does not raise an exception."""
    metric_registry = registry.MetricRegistry()
    metric_registry._loaded = False
    package_name = (
        registry.__package__
        or "cloud_accelerator_diagnostics.tpu_info.tpu_info"
    )
    provider_prefix = f"{package_name}.providers"

    with mock.patch.object(
        importlib,
        "import_module",
        autospec=True,
        side_effect=error_cls("Module not found", name=provider_prefix),
    ):
      metric_registry._ensure_providers_loaded()

    self.assertTrue(metric_registry._loaded)

  @parameterized.named_parameters(
      dict(
          testcase_name="downstream_submodule_not_found",
          error_cls=ModuleNotFoundError,
          error_name="cloud_accelerator_diagnostics.tpu_info.tpu_info.providers.runtime",
      ),
      dict(
          testcase_name="downstream_missing_dependency",
          error_cls=ModuleNotFoundError,
          error_name="nonexistent_pkg",
      ),
      dict(
          testcase_name="downstream_import_error_unnamed",
          error_cls=ImportError,
          error_name=None,
      ),
  )
  def test_ensure_providers_loaded_downstream_import_error_raised(
      self, error_cls, error_name
  ):
    """Tests that downstream import errors inside providers are re-raised."""
    metric_registry = registry.MetricRegistry()
    metric_registry._loaded = False

    if error_name is not None:
      err = error_cls("Downstream import error", name=error_name)
    else:
      err = error_cls("cannot import name 'something'")

    with mock.patch.object(
        importlib,
        "import_module",
        autospec=True,
        side_effect=err,
    ):
      with self.assertRaises(error_cls):
        metric_registry._ensure_providers_loaded()

    self.assertFalse(metric_registry._loaded)

  def test_ensure_providers_loaded_thread_safety_concurrent_calls(self):
    """Tests concurrent calls to an unloaded registry load providers safely."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    errors = []

    def worker(worker_id: int):
      try:
        if worker_id % 4 == 0:
          reg.get_all_descriptors()
        elif worker_id % 4 == 1:
          reg.get_descriptor("test_metric")
        elif worker_id % 4 == 2:
          reg.get_all_groups()
        else:
          reg.get_all_metric_names()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEmpty(errors)
    self.assertTrue(reg._loaded)

  def test_get_descriptor_triggers_ensure_providers_loaded(self):
    """Tests that get_descriptor() on an unloaded registry triggers provider loading."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    reg.get_descriptor("non_existent_metric")
    self.assertTrue(reg._loaded)

  def test_get_all_descriptors_triggers_ensure_providers_loaded(self):
    """Tests that get_all_descriptors() on an unloaded registry triggers provider loading."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    descriptors = reg.get_all_descriptors()
    self.assertTrue(reg._loaded)
    self.assertIsInstance(descriptors, dict)

  def test_get_all_metric_names_triggers_ensure_providers_loaded(self):
    """Tests that get_all_metric_names() on an unloaded registry triggers provider loading."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    metric_names = reg.get_all_metric_names()
    self.assertTrue(reg._loaded)
    self.assertIsInstance(metric_names, frozenset)

  def test_get_all_filter_schemas_triggers_ensure_providers_loaded(self):
    """Tests that get_all_filter_schemas() on an unloaded registry triggers provider loading."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    schemas = reg.get_all_filter_schemas()
    self.assertTrue(reg._loaded)
    self.assertIsInstance(schemas, dict)

  def test_get_descriptors_by_group_triggers_ensure_providers_loaded(self):
    """Tests that get_descriptors_by_group() on an unloaded registry triggers provider loading."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    descs = reg.get_descriptors_by_group("runtime")
    self.assertTrue(reg._loaded)
    self.assertIsInstance(descs, list)

  def test_get_all_groups_triggers_ensure_providers_loaded(self):
    """Tests that get_all_groups() on an unloaded registry triggers provider loading."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    groups = reg.get_all_groups()
    self.assertTrue(reg._loaded)
    self.assertIsInstance(groups, list)

  def test_render_metrics_triggers_ensure_providers_loaded(self):
    """Tests that render_metrics() on an unloaded registry triggers provider loading."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()
    self.assertFalse(reg._loaded)

    renderables = reg.render_metrics([], chip_type=None, count=0)
    self.assertTrue(reg._loaded)
    self.assertEqual(renderables, [])

  @parameterized.named_parameters(
      dict(
          testcase_name="get_descriptor",
          action=lambda reg: reg.get_descriptor("test_m"),
      ),
      dict(
          testcase_name="get_all_descriptors",
          action=lambda reg: reg.get_all_descriptors(),
      ),
      dict(
          testcase_name="get_all_metric_names",
          action=lambda reg: reg.get_all_metric_names(),
      ),
      dict(
          testcase_name="get_all_filter_schemas",
          action=lambda reg: reg.get_all_filter_schemas(),
      ),
      dict(
          testcase_name="get_descriptors_by_group",
          action=lambda reg: reg.get_descriptors_by_group("test_g"),
      ),
      dict(
          testcase_name="get_all_groups",
          action=lambda reg: reg.get_all_groups(),
      ),
      dict(
          testcase_name="render_metrics",
          action=lambda reg: reg.render_metrics([], chip_type=None, count=0),
      ),
  )
  def test_public_methods_explicitly_call_ensure_providers_loaded(self, action):
    """Tests that every public method invokes _ensure_providers_loaded."""
    reg = registry.MetricRegistry()
    with mock.patch.object(
        reg, "_ensure_providers_loaded", autospec=True
    ) as mock_ensure:
      action(reg)
      mock_ensure.assert_called_once()

  def test_register_before_getters_preserves_custom_metric(self):
    """Tests registering a custom metric before getter calls preserves it."""
    registry.MetricRegistry._reset()
    reg = registry.MetricRegistry()

    @registry.register_metric("pre_getter_metric", "custom_group", "desc")
    def dummy_handler(**_kwargs):
      return []

    self.assertIsNotNone(reg.get_descriptor("pre_getter_metric"))
    self.assertIn("pre_getter_metric", reg.get_all_metric_names())

  def test_register_and_get_descriptor(self):
    """Tests registering a metric handler and retrieving its descriptor."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric(
        "test_metric",
        "test_group",
        "test description",
        allowed_filters={"f1"},
    )
    def dummy_handler(**_kwargs):
      return []

    desc = metric_registry.get_descriptor("test_metric")
    self.assertIsNotNone(desc)
    self.assertEqual(desc.name, "test_metric")
    self.assertEqual(desc.group, "test_group")
    self.assertEqual(desc.description, "test description")
    self.assertEqual(desc.allowed_filters, {"f1"})
    self.assertEqual(desc.handler, dummy_handler)

    self.assertIn("test_metric", metric_registry.get_all_metric_names())
    self.assertIn("test_group", metric_registry.get_all_groups())
    self.assertLen(metric_registry.get_descriptors_by_group("test_group"), 1)
    self.assertEqual(
        metric_registry.get_all_filter_schemas()["test_metric"], {"f1"}
    )

  def test_get_descriptor_nonexistent(self):
    """Tests that getting an unregistered metric returns None."""
    metric_registry = registry.MetricRegistry()
    self.assertIsNone(metric_registry.get_descriptor("nonexistent_metric"))

  def test_get_all_descriptors(self):
    """Tests getting all registered metric descriptors."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("m1", "group_a", "desc 1")
    def handler1(**_kwargs):
      return []

    @registry.register_metric("m2", "group_b", "desc 2")
    def handler2(**_kwargs):
      return []

    descriptors = metric_registry.get_all_descriptors()
    self.assertIn("m1", descriptors)
    self.assertIn("m2", descriptors)
    self.assertEqual(descriptors["m1"].description, "desc 1")
    self.assertEqual(descriptors["m2"].description, "desc 2")

  def test_get_all_groups(self):
    """Tests retrieving unique sorted group names."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("m_beta", "group_b", "desc b")
    def handler_b(**_kwargs):
      return []

    @registry.register_metric("m_alpha", "group_a", "desc a")
    def handler_a(**_kwargs):
      return []

    @registry.register_metric("m_gamma", "group_a", "desc a2")
    def handler_a2(**_kwargs):
      return []

    groups = metric_registry.get_all_groups()
    self.assertIn("group_a", groups)
    self.assertIn("group_b", groups)
    self.assertEqual(groups, sorted(list(set(groups))))

  def test_get_descriptors_by_group(self):
    """Tests retrieving metric descriptors filtered by group."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("m1", "group_x", "desc 1")
    def handler1(**_kwargs):
      return []

    @registry.register_metric("m2", "group_x", "desc 2")
    def handler2(**_kwargs):
      return []

    @registry.register_metric("m3", "group_y", "desc 3")
    def handler3(**_kwargs):
      return []

    group_x_descs = metric_registry.get_descriptors_by_group("group_x")
    self.assertLen(group_x_descs, 2)
    group_x_names = {d.name for d in group_x_descs}
    self.assertEqual(group_x_names, {"m1", "m2"})

    group_y_descs = metric_registry.get_descriptors_by_group("group_y")
    self.assertLen(group_y_descs, 1)
    self.assertEqual(group_y_descs[0].name, "m3")

  def test_get_all_filter_schemas(self):
    """Tests retrieving filter schemas for metrics with allowed filters."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric(
        "filtered_metric", "group_f", "desc f", allowed_filters={"p50", "p90"}
    )
    def handler_filtered(**_kwargs):
      return []

    @registry.register_metric(
        "unfiltered_metric", "group_f", "desc u", allowed_filters=None
    )
    def handler_unfiltered(**_kwargs):
      return []

    schemas = metric_registry.get_all_filter_schemas()
    self.assertEqual(schemas.get("filtered_metric"), {"p50", "p90"})
    self.assertNotIn("unfiltered_metric", schemas)


class BatchScrapePrometheusTest(parameterized.TestCase):
  """Tests for batch Prometheus scraping in registry."""

  def setUp(self):
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()
    orig_descriptors = dict(reg._descriptors)
    orig_loaded = reg._loaded

    def cleanup():
      active_reg = registry.MetricRegistry()
      active_reg._descriptors = dict(orig_descriptors)
      active_reg._loaded = orig_loaded

    self.addCleanup(cleanup)

  @parameterized.named_parameters(
      dict(
          testcase_name="custom_port_valid",
          env_port="9876",
          expected_port=9876,
      ),
      dict(
          testcase_name="custom_port_invalid_fallback",
          env_port="not_a_number",
          expected_port=8080,
      ),
  )
  def test_batch_scrape_prometheus_port_handling(
      self, env_port, expected_port
  ):
    """Tests Prometheus port parsing from environment variables."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("m_port_test", "orbax", "desc")
    def dummy_handler(**_kwargs):
      return []

    with mock.patch.dict(os.environ, {"TEST_PROMETHEUS_PORT": env_port}):
      with mock.patch.object(
          metrics, "scrape_prometheus", return_value=[], autospec=True
      ) as mock_scrape:
        metric_registry._batch_scrape_prometheus(
            batch_metrics=[("m_port_test", None)],
            telemetry_name="Orbax",
            default_port=8080,
            port_env_var="TEST_PROMETHEUS_PORT",
            env_var_name="ENABLE_ORBAX=true",
        )
        mock_scrape.assert_called_once_with(expected_port)

  def test_batch_scrape_prometheus_connection_error(self):
    """Tests offline Prometheus server returns a warning Panel."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("offline_metric", "orbax", "desc")
    def dummy_handler(**_kwargs):
      return []

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        side_effect=metrics.PrometheusConnectionError("Connection refused"),
    ):
      renderables = metric_registry._batch_scrape_prometheus(
          batch_metrics=[("offline_metric", None)],
          telemetry_name="Orbax",
          default_port=metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
          port_env_var="ORBAX_PROMETHEUS_PORT",
          env_var_name="ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
      )

    self.assertLen(renderables, 1)
    self.assertIsInstance(renderables[0], panel.Panel)
    self.assertIn("Orbax Telemetry Offline", str(renderables[0].title))
    self.assertIn(
        "ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
        str(renderables[0].renderable),
    )

  def test_batch_scrape_prometheus_unexpected_exception(self):
    """Tests generic exception during scraping returns an error Panel."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("error_metric", "orbax", "desc")
    def dummy_handler(**_kwargs):
      return []

    with mock.patch.object(
        metrics,
        "scrape_prometheus",
        autospec=True,
        side_effect=RuntimeError("Scraper crash"),
    ):
      renderables = metric_registry._batch_scrape_prometheus(
          batch_metrics=[("error_metric", None)],
          telemetry_name="Orbax",
          default_port=metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
          port_env_var="ORBAX_PROMETHEUS_PORT",
          env_var_name="ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
      )

    self.assertLen(renderables, 1)
    self.assertIsInstance(renderables[0], panel.Panel)
    self.assertIn("Error", str(renderables[0].title))
    self.assertIn("Scraper crash", str(renderables[0].renderable))

  def test_batch_scrape_prometheus_missing_metrics_warning(self):
    """Tests missing metrics warning when batch has missing metrics."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("found_metric", "orbax", "desc found")
    def found_handler(**_kwargs):
      return [rich_table.Table(title="Found Table")]

    @registry.register_metric("missing_metric", "orbax", "desc missing")
    def missing_handler(**_kwargs):
      return []

    with mock.patch.object(
        metrics, "scrape_prometheus", return_value=[], autospec=True
    ):
      renderables = metric_registry._batch_scrape_prometheus(
          batch_metrics=[("found_metric", None), ("missing_metric", None)],
          telemetry_name="Orbax",
          default_port=metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
          port_env_var="ORBAX_PROMETHEUS_PORT",
          env_var_name="ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
      )

    self.assertLen(renderables, 2)
    self.assertIsInstance(renderables[0], rich_table.Table)
    self.assertIsInstance(renderables[1], panel.Panel)
    self.assertIn("Orbax Metrics Not Found", str(renderables[1].title))
    self.assertIn("missing_metric", str(renderables[1].renderable))

  def test_batch_scrape_prometheus_skips_unregistered_or_no_handler_metrics(
      self,
  ):
    """Tests skipping unregistered metrics or metrics without a handler."""
    metric_registry = registry.MetricRegistry()
    mock_table = rich_table.Table(title="Valid Metric Table")

    @registry.register_metric("valid_metric", "orbax", "desc valid")
    def valid_handler(**_kwargs):
      return [mock_table]

    metric_registry._descriptors["no_handler_metric"] = (
        registry.MetricDescriptor(
            name="no_handler_metric",
            group="orbax",
            description="desc no handler",
            handler=cast(Any, None),
        )
    )

    with mock.patch.object(
        metrics, "scrape_prometheus", return_value=[], autospec=True
    ):
      renderables = metric_registry._batch_scrape_prometheus(
          batch_metrics=[
              ("valid_metric", None),
              ("unregistered_metric", None),
              ("no_handler_metric", None),
          ],
          telemetry_name="Orbax",
          default_port=metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
          port_env_var="ORBAX_PROMETHEUS_PORT",
          env_var_name="ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
      )

    self.assertEqual(renderables, [mock_table])

  def test_batch_scrape_prometheus_single_metric_missing(self):
    """Tests that a single missing metric does not add batch missing warnings."""
    metric_registry = registry.MetricRegistry()

    @registry.register_metric("single_missing", "orbax", "desc")
    def dummy_handler(**_kwargs):
      return []

    with mock.patch.object(
        metrics, "scrape_prometheus", return_value=[], autospec=True
    ):
      renderables = metric_registry._batch_scrape_prometheus(
          batch_metrics=[("single_missing", None)],
          telemetry_name="Orbax",
          default_port=metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
          port_env_var="ORBAX_PROMETHEUS_PORT",
          env_var_name="ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
      )

    self.assertEmpty(renderables)

  def test_batch_scrape_prometheus_handler_exception(self):
    """Tests that a handler exception in batch scraping yields an error panel and continues."""
    metric_registry = registry.MetricRegistry()
    mock_table = rich_table.Table(title="Valid Table")

    @registry.register_metric("faulty_metric", "orbax", "desc faulty")
    def faulty_handler(**_kwargs):
      raise RuntimeError("Handler failure")

    @registry.register_metric("valid_metric", "orbax", "desc valid")
    def valid_handler(**_kwargs):
      return [mock_table]

    with mock.patch.object(
        metrics, "scrape_prometheus", return_value=[], autospec=True
    ):
      renderables = metric_registry._batch_scrape_prometheus(
          batch_metrics=[("faulty_metric", None), ("valid_metric", None)],
          telemetry_name="Orbax",
          default_port=metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
          port_env_var="ORBAX_PROMETHEUS_PORT",
          env_var_name="ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
      )

    self.assertLen(renderables, 2)
    self.assertIsInstance(renderables[0], panel.Panel)
    self.assertIn(
        "Error rendering metric 'faulty_metric'",
        str(renderables[0].renderable),
    )
    self.assertEqual(renderables[1], mock_table)


class RenderMetricsTest(parameterized.TestCase):
  """Tests for render_metrics routing and execution."""

  def setUp(self):
    super().setUp()
    reg = registry.MetricRegistry()
    reg._ensure_providers_loaded()
    orig_descriptors = dict(reg._descriptors)
    orig_loaded = reg._loaded

    def cleanup():
      active_reg = registry.MetricRegistry()
      active_reg._descriptors = dict(orig_descriptors)
      active_reg._loaded = orig_loaded

    self.addCleanup(cleanup)

  def test_render_metrics_unknown_metric(self):
    """Tests that requesting an unknown metric produces an error Panel."""
    metric_registry = registry.MetricRegistry()
    renderables = metric_registry.render_metrics(
        [("unknown_metric", None)],
        chip_type=device.TpuChip.V4,
        count=1,
    )
    self.assertLen(renderables, 1)
    self.assertIsInstance(renderables[0], panel.Panel)
    self.assertIn(
        "Unknown metric: unknown_metric",
        str(renderables[0].renderable),
    )

  def test_render_metrics_other_group_success(self):
    """Tests rendering non-Prometheus metrics calls handlers directly."""
    metric_registry = registry.MetricRegistry()
    mock_table = rich_table.Table(title="Custom Metric Table")

    mock_handler = mock.MagicMock(return_value=[mock_table])
    metric_registry.register(
        name="custom_metric",
        group="runtime",
        description="Custom runtime metric",
    )(mock_handler)

    renderables = metric_registry.render_metrics(
        [("custom_metric", {"filter_key": "val"})],
        chip_type=device.TpuChip.V4,
        count=4,
    )

    mock_handler.assert_called_once_with(
        chip_type=device.TpuChip.V4,
        count=4,
        filters={"filter_key": "val"},
    )
    self.assertEqual(renderables, [mock_table])

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  def test_render_metrics_batch_scrape_orbax(self, mock_scrape):
    """Tests rendering Orbax metrics uses batch Prometheus scraping."""
    metric_registry = registry.MetricRegistry()
    mock_family = mock.create_autospec(Metric, instance=True)
    mock_family.name = "test_metric"
    mock_family.samples = []
    mock_scrape.return_value = [mock_family]

    mock_orbax_table1 = rich_table.Table(title="Orbax Table 1")
    mock_orbax_table2 = rich_table.Table(title="Orbax Table 2")

    mock_handler1 = mock.MagicMock(return_value=[mock_orbax_table1])
    mock_handler2 = mock.MagicMock(return_value=[mock_orbax_table2])

    metric_registry.register(name="orbax_m1", group="orbax", description="d1")(
        mock_handler1
    )
    metric_registry.register(name="orbax_m2", group="orbax", description="d2")(
        mock_handler2
    )

    renderables = metric_registry.render_metrics(
        [("orbax_m1", None), ("orbax_m2", None)],
        chip_type=device.TpuChip.V4,
        count=4,
    )

    mock_scrape.assert_called_once_with(metrics.ORBAX_PROMETHEUS_DEFAULT_PORT)
    mock_handler1.assert_called_once_with(
        pre_scraped_metrics=[mock_family],
        skip_if_missing=True,
        filters=None,
        chip_type=device.TpuChip.V4,
        count=4,
    )
    mock_handler2.assert_called_once_with(
        pre_scraped_metrics=[mock_family],
        skip_if_missing=True,
        filters=None,
        chip_type=device.TpuChip.V4,
        count=4,
    )
    self.assertEqual(renderables, [mock_orbax_table1, mock_orbax_table2])

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  def test_render_metrics_batch_scrape_pygrain(self, mock_scrape):
    """Tests rendering Pygrain metrics uses batch Prometheus scraping."""
    metric_registry = registry.MetricRegistry()
    mock_scrape.return_value = []
    mock_pygrain_table = rich_table.Table(title="Pygrain Table")
    mock_handler = mock.MagicMock(return_value=[mock_pygrain_table])

    metric_registry.register(
        name="pygrain_m1", group="pygrain", description="desc pygrain"
    )(mock_handler)

    renderables = metric_registry.render_metrics(
        [("pygrain_m1", None)],
        chip_type=device.TpuChip.V4,
        count=2,
    )

    mock_scrape.assert_called_once_with(metrics.PYGRAIN_PROMETHEUS_DEFAULT_PORT)
    mock_handler.assert_called_once()
    self.assertEqual(renderables, [mock_pygrain_table])

  def test_render_metrics_other_group_no_handler(self):
    """Tests that metrics without a handler are skipped in other groups."""
    metric_registry = registry.MetricRegistry()
    metric_registry._descriptors["runtime_no_handler"] = (
        registry.MetricDescriptor(
            name="runtime_no_handler",
            group="runtime",
            description="desc",
            handler=cast(Any, None),
        )
    )
    renderables = metric_registry.render_metrics(
        [("runtime_no_handler", None)],
        chip_type=device.TpuChip.V4,
        count=1,
    )
    self.assertEmpty(renderables)

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  def test_render_metrics_mixed_groups(self, mock_scrape):
    """Tests rendering a batch containing mixed metric groups and unknown metrics."""
    metric_registry = registry.MetricRegistry()
    mock_scrape.return_value = []

    runtime_table = rich_table.Table(title="Runtime Table")
    orbax_table = rich_table.Table(title="Orbax Table")
    pygrain_table = rich_table.Table(title="Pygrain Table")

    metric_registry.register("runtime_m", "runtime", "desc runtime")(
        mock.MagicMock(return_value=[runtime_table])
    )
    metric_registry.register("orbax_m", "orbax", "desc orbax")(
        mock.MagicMock(return_value=[orbax_table])
    )
    metric_registry.register("pygrain_m", "pygrain", "desc pygrain")(
        mock.MagicMock(return_value=[pygrain_table])
    )

    renderables = metric_registry.render_metrics(
        [
            ("unknown_metric", None),
            ("runtime_m", {"filter1": "val"}),
            ("orbax_m", None),
            ("pygrain_m", None),
        ],
        chip_type=device.TpuChip.V4,
        count=2,
    )

    self.assertLen(renderables, 4)
    self.assertIsInstance(renderables[0], panel.Panel)
    self.assertIn(
        "Unknown metric: unknown_metric", str(renderables[0].renderable)
    )
    self.assertEqual(renderables[1], orbax_table)
    self.assertEqual(renderables[2], pygrain_table)
    self.assertEqual(renderables[3], runtime_table)

  def test_render_metrics_other_group_handler_exception(self):
    """Tests that a handler exception in non-Prometheus metrics yields an error panel and continues."""
    metric_registry = registry.MetricRegistry()
    mock_table = rich_table.Table(title="Valid Table")

    @registry.register_metric("faulty_runtime", "runtime", "desc faulty")
    def faulty_handler(**_kwargs):
      raise RuntimeError("Runtime handler failure")

    @registry.register_metric("valid_runtime", "runtime", "desc valid")
    def valid_handler(**_kwargs):
      return [mock_table]

    renderables = metric_registry.render_metrics(
        [("faulty_runtime", None), ("valid_runtime", None)],
        chip_type=device.TpuChip.V4,
        count=1,
    )

    self.assertLen(renderables, 2)
    self.assertIsInstance(renderables[0], panel.Panel)
    self.assertIn(
        "Error rendering metric 'faulty_runtime'",
        str(renderables[0].renderable),
    )
    self.assertEqual(renderables[1], mock_table)

  def test_to_json_serializable_types(self):
    """Tests _to_json_serializable across namedtuples, dataclasses, sets, dicts, and primitives."""
    Point = collections.namedtuple("Point", ["x", "y"])

    @dataclasses.dataclass
    class SampleData:
      name: str
      values: list[int]

    class SampleEnum(enum.Enum):
      FOO = "foo_val"

    data_structure = {
        "point": Point(10, 20),
        "data": SampleData("test", [1, 2, 3]),
        "unique_items": {4, 5},
        "tuple_items": (6, 7),
        "nested_dict": {"k": "v"},
        "primitive": 42,
        "enum_item": SampleEnum.FOO,
    }

    serialized = registry._to_json_serializable(data_structure)
    self.assertEqual(serialized["point"], {"x": 10, "y": 20})
    self.assertEqual(serialized["data"], {"name": "test", "values": [1, 2, 3]})
    self.assertEqual(sorted(serialized["unique_items"]), [4, 5])
    self.assertEqual(serialized["tuple_items"], [6, 7])
    self.assertEqual(serialized["nested_dict"], {"k": "v"})
    self.assertEqual(serialized["primitive"], 42)
    self.assertEqual(serialized["enum_item"], "foo_val")
    self.assertIs(registry._to_json_serializable(Point), Point)
    self.assertIs(registry._to_json_serializable(SampleData), SampleData)
    self.assertIs(registry._to_json_serializable(SampleEnum), SampleEnum)

  def test_get_raw_metric_unknown_metric(self):
    """Tests that get_raw_metric raises ValueError for unregistered metrics."""
    reg = registry.MetricRegistry()
    with self.assertRaisesRegex(ValueError, "Unknown metric: non_existent"):
      reg.get_raw_metric("non_existent", chip_type=None, count=1)

  def test_get_raw_metric_with_custom_raw_handler(self):
    """Tests get_raw_metric with a registered custom raw handler."""
    reg = registry.MetricRegistry()

    def custom_raw(chip_type, count, filters):
      return {"chip": str(chip_type), "count": count, "filters": filters}

    @registry.register_metric(
        "test_custom_raw",
        "runtime",
        "custom raw desc",
        raw_handler=custom_raw,
    )
    def dummy_handler(**_kwargs):
      return []

    result = reg.get_raw_metric(
        "test_custom_raw",
        chip_type=device.TpuChip.V4,
        count=4,
        filters={"k": "v"},
    )
    self.assertEqual(
        result,
        {
            "chip": "TPU v4 chip",
            "count": 4,
            "filters": {"k": "v"},
        },
    )

  def test_get_raw_metric_ensures_providers_loaded(self):
    """Tests that get_raw_metric explicitly invokes _ensure_providers_loaded."""
    reg = registry.MetricRegistry()
    mock_desc = mock.MagicMock(raw_handler=lambda **_: {"k": "v"})
    with mock.patch.object(
        reg, "_ensure_providers_loaded"
    ) as mock_ensure, mock.patch.object(
        reg, "get_descriptor", return_value=mock_desc
    ):
      result = reg.get_raw_metric("test_metric", chip_type=None, count=1)
      self.assertEqual(result, {"k": "v"})
      mock_ensure.assert_called_once()

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  @mock.patch.object(metrics, "ORBAX_PROMETHEUS_DEFAULT_PORT", 9091)
  @mock.patch.object(metrics, "PYGRAIN_PROMETHEUS_DEFAULT_PORT", 9092)
  def test_get_raw_metric_orbax_default_and_custom_port(self, mock_scrape):
    """Tests get_raw_metric for Orbax Prometheus metrics with default and custom ports."""
    reg = registry.MetricRegistry()
    reg.register(
        name="orbax_write_storage_type",
        group="orbax",
        description="Orbax storage type",
    )(lambda **_kwargs: [])

    mock_sample = Sample(
        "jax_orbax_write_storage_type",
        {"storage_type": "gcs"},
        42.0,
        123456789,
    )
    mock_metric = Metric(
        "jax_orbax_write_storage_type", "help", "gauge"
    )
    mock_metric.samples = [mock_sample]
    mock_scrape.return_value = [mock_metric]

    # Test with default port
    samples = reg.get_raw_metric(
        "orbax_write_storage_type", chip_type=None, count=1
    )
    mock_scrape.assert_called_with(9091)
    self.assertEqual(
        samples,
        [
            {
                "name": "jax_orbax_write_storage_type",
                "labels": {"storage_type": "gcs"},
                "value": 42.0,
                "timestamp": 123456789,
            }
        ],
    )

    # Test with custom port in env
    with mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "9555"}):
      _ = reg.get_raw_metric(
          "orbax_write_storage_type", chip_type=None, count=1
      )
      mock_scrape.assert_called_with(9555)

    # Test with invalid port in env (fallback to default)
    with mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "invalid"}):
      _ = reg.get_raw_metric(
          "orbax_write_storage_type", chip_type=None, count=1
      )
      mock_scrape.assert_called_with(9091)

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  def test_get_raw_metric_orbax_no_matching_metric(self, mock_scrape):
    """Tests get_raw_metric when scraped metrics do not match the prom name."""
    reg = registry.MetricRegistry()
    reg.register(
        name="orbax_write_storage_type",
        group="orbax",
        description="Orbax storage type",
    )(lambda **_kwargs: [])

    mock_scrape.return_value = []
    samples = reg.get_raw_metric(
        "orbax_write_storage_type", chip_type=None, count=1
    )
    self.assertEqual(samples, [])

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  @mock.patch.object(metrics, "ORBAX_PROMETHEUS_DEFAULT_PORT", 9091)
  @mock.patch.object(metrics, "PYGRAIN_PROMETHEUS_DEFAULT_PORT", 9092)
  def test_get_raw_metric_pygrain_default_port(self, mock_scrape):
    """Tests get_raw_metric for PyGrain metrics with default port."""
    reg = registry.MetricRegistry()
    reg.register(
        name="pygrain_data_sources_read_size",
        group="pygrain",
        description="PyGrain read size",
    )(lambda **_kwargs: [])

    mock_sample = Sample(
        "grain_python_data_sources_bytes_read",
        {"dataset": "imagenet"},
        1024.0,
        123456789,
    )
    mock_metric = Metric(
        "grain_python_data_sources_bytes_read", "help", "gauge"
    )
    mock_metric.samples = [mock_sample]
    mock_scrape.return_value = [mock_metric]

    # Default port with no environment variable
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("PYGRAIN_PROMETHEUS_PORT", None)
      samples = reg.get_raw_metric(
          "pygrain_data_sources_read_size", chip_type=None, count=1
      )
      mock_scrape.assert_called_with(9092)
      self.assertEqual(
          samples,
          [
              {
                  "name": "grain_python_data_sources_bytes_read",
                  "labels": {"dataset": "imagenet"},
                  "value": 1024.0,
                  "timestamp": 123456789,
              }
          ],
      )

    # Fallback to default port on invalid environment variable
    with mock.patch.dict(os.environ, {"PYGRAIN_PROMETHEUS_PORT": "invalid"}):
      _ = reg.get_raw_metric(
          "pygrain_data_sources_read_size", chip_type=None, count=1
      )
      mock_scrape.assert_called_with(9092)

  @mock.patch.object(metrics, "scrape_prometheus", autospec=True)
  def test_get_raw_metric_pygrain_custom_port(self, mock_scrape):
    """Tests get_raw_metric for PyGrain metrics with PYGRAIN_PROMETHEUS_PORT."""
    reg = registry.MetricRegistry()
    reg.register(
        name="pygrain_data_sources_read_size",
        group="pygrain",
        description="PyGrain read size",
    )(lambda **_kwargs: [])

    mock_sample = Sample(
        "grain_python_data_sources_bytes_read",
        {"dataset": "imagenet"},
        1024.0,
        123456789,
    )
    mock_metric = Metric(
        "grain_python_data_sources_bytes_read", "help", "gauge"
    )
    mock_metric.samples = [mock_sample]
    mock_scrape.return_value = [mock_metric]

    with mock.patch.dict(os.environ, {"PYGRAIN_PROMETHEUS_PORT": "9666"}):
      samples = reg.get_raw_metric(
          "pygrain_data_sources_read_size", chip_type=None, count=1
      )
      mock_scrape.assert_called_with(9666)
      self.assertEqual(
          samples,
          [
              {
                  "name": "grain_python_data_sources_bytes_read",
                  "labels": {"dataset": "imagenet"},
                  "value": 1024.0,
                  "timestamp": 123456789,
              }
          ],
      )

  def test_get_raw_metric_not_implemented(self):
    """Tests that get_raw_metric raises NotImplementedError for metrics without raw support."""
    reg = registry.MetricRegistry()

    @registry.register_metric("unsupported_raw", "unknown_group", "desc")
    def dummy_handler(**_kwargs):
      return []

    with self.assertRaisesRegex(
        NotImplementedError,
        "Raw data collection not implemented for metric 'unsupported_raw'",
    ):
      reg.get_raw_metric("unsupported_raw", chip_type=None, count=1)

  @mock.patch.object(metrics, "scrape_prometheus", return_value=[])
  def test_get_raw_metric_orbax_unmapped_metric(self, mock_scrape):
    del mock_scrape  # Unused mock parameter.
    reg = registry.MetricRegistry()
    reg.register(
        name="custom_unmapped_orbax",
        group="orbax",
        description="Unmapped orbax metric",
    )(lambda **_kwargs: [])
    result = reg.get_raw_metric(
        "custom_unmapped_orbax", chip_type=None, count=1
    )
    self.assertEqual(result, [])


if __name__ == "__main__":
  absltest.main()

