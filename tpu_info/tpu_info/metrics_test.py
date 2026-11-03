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

"""Tests for metrics.py."""
from concurrent import futures
import contextlib
import functools
import http.client
import itertools
import os
import textwrap
from unittest import mock
import urllib.request

from absl.testing import absltest
from absl.testing import parameterized
from tpu_info import device
from tpu_info import metrics
import grpc

from tpu_info.proto import tpu_metric_service_pb2
from tpu_info.proto import tpu_metric_service_pb2_grpc
from tpu_info.proto import tpu_telemetry_pb2


class FakeGaugeTpuMetrics(
    tpu_metric_service_pb2_grpc.RuntimeMetricServiceServicer
):

  def __init__(
      self,
      responses: dict[metrics.MetricName, list[int | float]],
      device_ordinal_type: str = "string",
  ):
    self._responses = responses
    self._device_ordinal_type = device_ordinal_type
    super().__init__()

  @functools.singledispatchmethod
  def _gauge(self, val):
    raise NotImplementedError(f"Cannot create Gauge from {val}")

  @_gauge.register
  def _(self, val: int):
    return tpu_metric_service_pb2.Gauge(as_int=val)

  @_gauge.register
  def _(self, val: float):
    return tpu_metric_service_pb2.Gauge(as_double=val)

  def GetRuntimeMetric(
      self, request: tpu_metric_service_pb2.MetricRequest, context
  ):
    metric_name = metrics.MetricName(request.metric_name)
    resp = self._responses[metric_name]

    metric_list = []
    for i, val in enumerate(resp):
      if metric_name == metrics.MetricName.HLO_QUEUE_SIZE:
        if self._device_ordinal_type == "int":
          device_ordinal_attr = tpu_metric_service_pb2.AttrValue(int_attr=i)
        elif self._device_ordinal_type == "string":
          device_ordinal_attr = tpu_metric_service_pb2.AttrValue(
              string_attr=str(i)
          )
        elif self._device_ordinal_type == "string_non_digit":
          device_ordinal_attr = tpu_metric_service_pb2.AttrValue(
              string_attr="NOT_A_DIGIT"
          )
        elif self._device_ordinal_type == "unsupported_type":
          device_ordinal_attr = tpu_metric_service_pb2.AttrValue(
              double_attr=float(i)
          )
        else:
          raise ValueError(
              f"Unknown device_ordinal_type: {self._device_ordinal_type}"
          )
        attribute = tpu_metric_service_pb2.Attribute(
            key="attributes",
            value=tpu_metric_service_pb2.AttrValue(
                kvlist_attr=tpu_metric_service_pb2.KeyValueList(
                    attributes=[
                        tpu_metric_service_pb2.Attribute(
                            key="device_ordinal",
                            value=device_ordinal_attr,
                        )
                    ]
                )
            ),
        )
      else:
        attribute = tpu_metric_service_pb2.Attribute(
            key="device-id",
            value=tpu_metric_service_pb2.AttrValue(int_attr=i),
        )

      metric_list.append(
          tpu_metric_service_pb2.Metric(
              attribute=attribute,
              gauge=self._gauge(val),
          )
      )

    return tpu_metric_service_pb2.MetricResponse(
        metric=tpu_metric_service_pb2.TPUMetric(
            name=metric_name.value,
            metrics=metric_list,
        )
    )


class FakeTpuStatusMetrics(
    tpu_metric_service_pb2_grpc.RuntimeMetricServiceServicer
):
  """Helper class for faking TPU status responses."""

  def __init__(
      self, response: tpu_metric_service_pb2.GetTpuRuntimeStatusResponse
  ):
    self._response = response
    super().__init__()

  def GetTpuRuntimeStatus(
      self,
      request: tpu_metric_service_pb2.GetTpuRuntimeStatusRequest,
      context,
  ):
    return self._response


# This class is used to mock the response from the libtpu metrics server for
# various latency metrics, where the response is a distribution, and
# buffer size is 1M, the bucket options are fixed exponential with 10 buckets,
# scale 1.0, and growth factor 2.0. The bucket counts are get from the
# responses.
class FakeTransferLatencyTpuMetrics(
    tpu_metric_service_pb2_grpc.RuntimeMetricServiceServicer
):

  def __init__(
      self,
      responses: dict[metrics.MetricName, list[int] | list[list[int]]],
      device_ordinal_type: str = "string",
  ):
    self._responses = responses
    self._device_ordinal_type = device_ordinal_type
    super().__init__()

  def GetRuntimeMetric(
      self, request: tpu_metric_service_pb2.MetricRequest, context
  ):
    """Mocks the GetRuntimeMetric RPC for transfer latency distributions.

    Args:
      request: The MetricRequest containing the metric name.
      context: The gRPC context.

    Returns:
      A MetricResponse containing a Distribution metric with pre-defined bucket
      counts.
    """
    metric_name = metrics.MetricName(request.metric_name)
    resp = self._responses[metric_name]

    # Support both single list of counts and list of lists of counts
    if resp and isinstance(resp[0], list):
      metrics_data = resp
    else:
      metrics_data = [resp]

    if (
        metric_name
        == metrics.MetricName.HLO_EXECUTION_TIMING_DISTRIBUTION_MICROSECONDS
    ):
      if self._device_ordinal_type == "int":
        device_ordinal_attr = tpu_metric_service_pb2.AttrValue(int_attr=0)
      elif self._device_ordinal_type == "string":
        device_ordinal_attr = tpu_metric_service_pb2.AttrValue(string_attr="0")
      elif self._device_ordinal_type == "string_non_digit":
        device_ordinal_attr = tpu_metric_service_pb2.AttrValue(
            string_attr="NOT_A_DIGIT"
        )
      elif self._device_ordinal_type == "unsupported_type":
        device_ordinal_attr = tpu_metric_service_pb2.AttrValue(double_attr=0.0)
      else:
        raise ValueError(
            f"Unknown device_ordinal_type: {self._device_ordinal_type}"
        )

      metric_list = []
      for bucket_counts in metrics_data:
        count = sum(bucket_counts)
        metric_list.append(
            tpu_metric_service_pb2.Metric(
                attribute=tpu_metric_service_pb2.Attribute(
                    key="attributes",
                    value=tpu_metric_service_pb2.AttrValue(
                        kvlist_attr=tpu_metric_service_pb2.KeyValueList(
                            attributes=[
                                tpu_metric_service_pb2.Attribute(
                                    key="device_ordinal",
                                    value=device_ordinal_attr,
                                )
                            ]
                        )
                    ),
                ),
                distribution=tpu_metric_service_pb2.Distribution(
                    bucket_options=tpu_metric_service_pb2.Distribution.BucketOptions(
                        exponential_buckets=tpu_metric_service_pb2.Distribution.BucketOptions.Exponential(
                            num_finite_buckets=10,
                            scale=1.0,
                            growth_factor=2.0,
                        )
                    ),
                    count=count,
                    mean=10.0,
                    bucket_counts=bucket_counts,
                ),
            )
        )
      return tpu_metric_service_pb2.MetricResponse(
          metric=tpu_metric_service_pb2.TPUMetric(
              name=metric_name.value,
              metrics=metric_list,
          )
      )

    metric_list = []
    for bucket_counts in metrics_data:
      count = sum(bucket_counts)
      metric_list.append(
          tpu_metric_service_pb2.Metric(
              attribute=tpu_metric_service_pb2.Attribute(
                  key="buffer-size",
                  value=tpu_metric_service_pb2.AttrValue(
                      kvlist_attr=tpu_metric_service_pb2.KeyValueList(
                          attributes=[
                              tpu_metric_service_pb2.Attribute(
                                  key="buffer-size",
                                  value=tpu_metric_service_pb2.AttrValue(
                                      string_attr="1M"
                                  ),
                              )
                          ]
                      )
                  ),
              ),
              distribution=tpu_metric_service_pb2.Distribution(
                  bucket_options=tpu_metric_service_pb2.Distribution.BucketOptions(
                      exponential_buckets=tpu_metric_service_pb2.Distribution.BucketOptions.Exponential(
                          num_finite_buckets=10,
                          scale=1.0,
                          growth_factor=2.0,
                      )
                  ),
                  count=count,
                  bucket_counts=bucket_counts,
              ),
          )
      )

    return tpu_metric_service_pb2.MetricResponse(
        metric=tpu_metric_service_pb2.TPUMetric(
            name=metric_name.value,
            metrics=metric_list,
        )
    )


class TestMetrics(parameterized.TestCase):

  @staticmethod
  @contextlib.contextmanager
  def _fake_gauge_tpu_metrics(
      responses,
      device_ordinal_type="string",
  ) -> str:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tpu_metric_service_pb2_grpc.add_RuntimeMetricServiceServicer_to_server(
        FakeGaugeTpuMetrics(responses, device_ordinal_type),
        server,
    )
    port = server.add_secure_port(
        "localhost:0", grpc.local_server_credentials()
    )
    try:
      server.start()
      yield f"localhost:{port}"
    finally:
      server.stop(None)

  @staticmethod
  @contextlib.contextmanager
  def _fake_transfer_latency_tpu_metrics(
      responses,
      device_ordinal_type="string",
  ) -> str:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tpu_metric_service_pb2_grpc.add_RuntimeMetricServiceServicer_to_server(
        FakeTransferLatencyTpuMetrics(responses, device_ordinal_type),
        server,
    )
    port = server.add_secure_port(
        "localhost:0", grpc.local_server_credentials()
    )
    try:
      server.start()
      yield f"localhost:{port}"
    finally:
      server.stop(None)

  @staticmethod
  @contextlib.contextmanager
  def _fake_tpu_status_metrics(response):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tpu_metric_service_pb2_grpc.add_RuntimeMetricServiceServicer_to_server(
        FakeTpuStatusMetrics(response), server
    )
    port = server.add_secure_port(
        "localhost:0", grpc.local_server_credentials()
    )
    try:
      server.start()
      yield f"localhost:{port}"
    finally:
      server.stop(None)

  @parameterized.named_parameters(
      [
          (
              "v4_chip",
              device.TpuChip.V4,
              {
                  metrics.MetricName.TOTAL_MEMORY: [34088157184],
                  metrics.MetricName.MEMORY_USAGE: [100],
                  metrics.MetricName.DUTY_CYCLE_PCT: [50.01],
              },
          ),
          (
              "v4_8",
              device.TpuChip.V4,
              {
                  metrics.MetricName.TOTAL_MEMORY: [34088157184] * 4,
                  metrics.MetricName.MEMORY_USAGE: [
                      0,
                      8522039296,
                      17044078592,
                      34088157184,
                  ],
                  metrics.MetricName.DUTY_CYCLE_PCT: [0.0, 25.0, 50.0, 100.0],
              },
          ),
          (
              "v5litepod_8",
              device.TpuChip.V5E,
              {
                  metrics.MetricName.TOTAL_MEMORY: [17044078592] * 8,
                  metrics.MetricName.MEMORY_USAGE: list(range(8)),
                  metrics.MetricName.DUTY_CYCLE_PCT: [
                      100.0 / (i + 1) for i in range(8)
                  ],
              },
          ),
          (
              "v3_8",
              device.TpuChip.V3,
              {
                  metrics.MetricName.TOTAL_MEMORY: [17044078592] * 8,
                  metrics.MetricName.MEMORY_USAGE: list(range(8)),
                  # Duty cycle is reported per-chip
                  metrics.MetricName.DUTY_CYCLE_PCT: [
                      100.0 / (i + 1) for i in range(4)
                  ],
              },
          ),
      ],
  )
  def test_metrics(self, chip_type: device.TpuChip, responses):
    """Tests that chip usage metrics are parsed correctly."""
    with self._fake_gauge_tpu_metrics(responses) as fake_libtpu_addr:
      expected = [
          metrics.Usage(i, m, t, d)
          for i, (m, t, d) in enumerate(
              zip(
                  responses[metrics.MetricName.MEMORY_USAGE],
                  responses[metrics.MetricName.TOTAL_MEMORY],
                  itertools.chain.from_iterable(
                      itertools.repeat(d, chip_type.value.devices_per_chip)
                      for d in responses[metrics.MetricName.DUTY_CYCLE_PCT]
                  ),
              )
          )
      ]

      self.assertListEqual(
          metrics.get_chip_usage(chip_type, fake_libtpu_addr), expected
      )

  @parameterized.named_parameters(
      [
          (
              "buffer_transfer_latency",
              "buffer_transfer_latency",
              {
                  metrics.MetricName.BUFFER_TRANSFER_LATENCY_US: [
                      0,
                      0,
                      100,
                      0,
                      400,
                      0,
                      300,
                      0,
                      200,
                      0,
                  ],
              },
              metrics.TransferLatencyDistribution(
                  "1M", 32.0, 192.0, 224.0, 255.36
              ),
          ),
          (
              "buffer_transfer_latency_empty",
              "buffer_transfer_latency",
              {
                  metrics.MetricName.BUFFER_TRANSFER_LATENCY_US: [],
              },
              metrics.TransferLatencyDistribution("1M", 1.0, 1.0, 1.0, 1.0),
          ),
          (
              "inbound_buffer_transfer_latency",
              "inbound_buffer_transfer_latency",
              {
                  metrics.MetricName.INBOUND_BUFFER_TRANSFER_LATENCY_US: [
                      0,
                      0,
                      100,
                      0,
                      200,
                      0,
                  ],
              },
              metrics.TransferLatencyDistribution(
                  "1M", 10.0, 14.8, 15.4, 15.96
              ),
          ),
          (
              "host_to_device_transfer_latency",
              "host_to_device_transfer_latency",
              {
                  metrics.MetricName.HOST_TO_DEVICE_TRANSFER_LATENCY_US: [
                      1,
                      1,
                      10,
                      20,
                      40,
                      50,
                      30,
                      60,
                      20,
                      10,
                  ],
              },
              metrics.TransferLatencyDistribution(
                  "1M", 31.68, 160.0, 236.8, 486.4
              ),
          ),
          (
              "device_to_host_transfer_latency",
              "device_to_host_transfer_latency",
              {
                  metrics.MetricName.DEVICE_TO_HOST_TRANSFER_LATENCY_US: [
                      0,
                      50,
                      0,
                      200,
                  ],
              },
              metrics.TransferLatencyDistribution("1M", 5.5, 7.5, 7.74, 7.98),
          ),
          (
              "collective_e2e_latency",
              "collective_e2e_latency",
              {
                  metrics.MetricName.COLLECTIVE_E2E_LATENCY_US: [
                      0,
                      0,
                      100,
                      0,
                      200,
                      0,
                  ],
              },
              metrics.TransferLatencyDistribution(
                  "1M", 10.0, 14.8, 15.4, 15.96
              ),
          ),
          (
              "host_compute_latency",
              "host_compute_latency",
              {
                  metrics.MetricName.HOST_COMPUTE_LATENCY_US: [
                      0,
                      0,
                      100,
                      0,
                      200,
                      0,
                  ],
              },
              metrics.TransferLatencyDistribution(
                  "1M", 10.0, 14.8, 15.4, 15.96
              ),
          ),
          (
              "grpc_tcp_min_rtt",
              "grpc_tcp_min_rtt",
              {
                  metrics.MetricName.GRPC_TCP_MIN_RTT_US: [
                      0,
                      0,
                      100,
                      0,
                      200,
                      0,
                  ],
              },
              metrics.TransferLatencyDistribution(
                  "1M", 10.0, 14.8, 15.4, 15.96
              ),
          ),
          (
              "grpc_tcp_delivery_rate",
              "grpc_tcp_delivery_rate",
              {
                  metrics.MetricName.GRPC_TCP_DELIVERY_RATE_MBPS: [
                      0,
                      0,
                      100,
                      0,
                      200,
                      0,
                  ],
              },
              metrics.TransferLatencyDistribution(
                  "1M", 10.0, 14.8, 15.4, 15.96
              ),
          ),
      ],
  )
  def test_get_transfer_latency(self, metric_arg, responses, expected):
    """Tests that transfer latency metrics are parsed correctly."""
    with self._fake_transfer_latency_tpu_metrics(responses) as fake_libtpu_addr:

      self.assertListEqual(
          metrics.get_transfer_latency(
              metric_arg, fake_libtpu_addr, filters=None
          ),
          [expected],
      )

  def test_get_transfer_latency_with_filters(self):
    """Tests that filters are accepted but not yet used."""
    responses = {
        metrics.MetricName.BUFFER_TRANSFER_LATENCY_US: [
            0,
            0,
            100,
            0,
            400,
            0,
            300,
            0,
            200,
            0,
        ],
    }
    expected = metrics.TransferLatencyDistribution(
        "1M", 32.0, 192.0, 224.0, 255.36
    )
    with self._fake_transfer_latency_tpu_metrics(responses) as fake_libtpu_addr:
      result = metrics.get_transfer_latency(
          "buffer_transfer_latency",
          fake_libtpu_addr,
          filters={"percentile": ["p50"]},
      )
      self.assertListEqual(result, [expected])

  def test_get_transfer_latency_aggregation(self):
    """Tests that multiple metrics for the same buffer size are aggregated."""
    responses = {
        metrics.MetricName.BUFFER_TRANSFER_LATENCY_US: [
            [0, 0, 100, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
        ],
    }
    expected = metrics.TransferLatencyDistribution(
        "1M", 8.0, 14.4, 15.2, 15.92
    )
    with self._fake_transfer_latency_tpu_metrics(responses) as fake_libtpu_addr:
      result = metrics.get_transfer_latency(
          "buffer_transfer_latency",
          fake_libtpu_addr,
          filters=None,
      )
      self.assertListEqual(result, [expected])

  def test_get_runtime_hbm_utilization(self):
    """Tests that runtime HBM utilization metrics are parsed correctly."""
    responses = {
        metrics.MetricName.HBM_BW_UTIL: [10.5, 20.0, 30.5, 40.0],
    }
    expected = [(0, 10.5), (1, 20.0), (2, 30.5), (3, 40.0)]
    with self._fake_gauge_tpu_metrics(responses) as fake_libtpu_addr:
      self.assertListEqual(
          metrics.get_runtime_hbm_utilization(fake_libtpu_addr), expected
      )

  def test_get_tensorcore_idle_duration(self):
    """Tests that TensorCore idle duration metrics are parsed correctly."""
    responses = {
        metrics.MetricName.TENSORCORE_IDLE_DURATION: [1.5, 2.0, 3.5, 4.0],
    }
    expected = [(0, 1.5), (1, 2.0), (2, 3.5), (3, 4.0)]
    with self._fake_gauge_tpu_metrics(responses) as fake_libtpu_addr:
      self.assertListEqual(
          metrics.get_tensorcore_idle_duration(fake_libtpu_addr), expected
      )

  def test_get_hlo_queue_size(self):
    """Tests that HLO queue size metrics are parsed correctly."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_QUEUE_SIZE: [0, 5, 10, 2],
    }
    expected = [
        metrics.HloQueueSize(0, 0),
        metrics.HloQueueSize(1, 5),
        metrics.HloQueueSize(2, 10),
        metrics.HloQueueSize(3, 2),
    ]
    with self._fake_gauge_tpu_metrics(responses) as fake_libtpu_addr:
      self.assertListEqual(
          metrics.get_hlo_queue_size(chip_type, fake_libtpu_addr), expected
      )

  def test_get_hlo_queue_size_with_int_attr(self):
    """Tests that HLO queue size metrics are parsed correctly."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_QUEUE_SIZE: [0, 5, 10, 2],
    }
    expected = [
        metrics.HloQueueSize(0, 0),
        metrics.HloQueueSize(1, 5),
        metrics.HloQueueSize(2, 10),
        metrics.HloQueueSize(3, 2),
    ]
    with self._fake_gauge_tpu_metrics(
        responses, device_ordinal_type="int"
    ) as fake_libtpu_addr:
      self.assertListEqual(
          metrics.get_hlo_queue_size(chip_type, fake_libtpu_addr), expected
      )

  def test_get_hlo_exec_timing(self):
    """Tests that HLO execution timing metrics are parsed correctly."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_EXECUTION_TIMING_DISTRIBUTION_MICROSECONDS: [
            0,
            0,
            100,
            0,
            400,
            0,
            300,
            0,
            200,
            0,
        ],
    }
    # mean = 10.0, p50 = 32.0, p90 = 192.0, p95 = 224.0, p999 = 255.36
    expected = [
        metrics.HloExecutionTiming(
            0,
            10.0,
            32.0,
            192.0,
            224.0,
            255.36,
        )
    ]
    with self._fake_transfer_latency_tpu_metrics(responses) as fake_libtpu_addr:
      self.assertListEqual(
          metrics.get_hlo_exec_timing(chip_type, fake_libtpu_addr), expected
      )

  def test_get_hlo_exec_timing_with_int_attr(self):
    """Tests that HLO execution timing metrics are parsed correctly."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_EXECUTION_TIMING_DISTRIBUTION_MICROSECONDS: [
            0,
            0,
            100,
            0,
            400,
            0,
            300,
            0,
            200,
            0,
        ],
    }
    # mean = 10.0, p50 = 32.0, p90 = 192.0, p95 = 224.0, p999 = 255.36
    expected = [
        metrics.HloExecutionTiming(
            0,
            10.0,
            32.0,
            192.0,
            224.0,
            255.36,
        )
    ]
    with self._fake_transfer_latency_tpu_metrics(
        responses, device_ordinal_type="int"
    ) as fake_libtpu_addr:
      self.assertListEqual(
          metrics.get_hlo_exec_timing(chip_type, fake_libtpu_addr), expected
      )

  def test_get_hlo_exec_timing_empty(self):
    """Tests that HLO execution timing metrics return default when empty."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_EXECUTION_TIMING_DISTRIBUTION_MICROSECONDS: [
            [0, 0]
        ],
    }
    expected = [
        metrics.HloExecutionTiming(
            0,
            10.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )
    ]
    with self._fake_transfer_latency_tpu_metrics(responses) as fake_libtpu_addr:
      self.assertListEqual(
          metrics.get_hlo_exec_timing(chip_type, fake_libtpu_addr), expected
      )

  def test_get_hlo_queue_size_with_invalid_string_attr(self):
    """Tests that HLO queue size metrics fail with invalid string."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_QUEUE_SIZE: [0],
    }
    with self._fake_gauge_tpu_metrics(
        responses, device_ordinal_type="string_non_digit"
    ) as fake_libtpu_addr:
      with self.assertRaisesRegex(
          ValueError,
          "device_ordinal string_attr is not a digit",
      ):
        metrics.get_hlo_queue_size(chip_type, fake_libtpu_addr)

  def test_get_hlo_queue_size_with_unsupported_attr_type(self):
    """Tests that HLO queue size metrics fail with unsupported attr type."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_QUEUE_SIZE: [0],
    }
    with self._fake_gauge_tpu_metrics(
        responses, device_ordinal_type="unsupported_type"
    ) as fake_libtpu_addr:
      with self.assertRaisesRegex(
          TypeError,
          "Unexpected type for device_ordinal attribute",
      ):
        metrics.get_hlo_queue_size(chip_type, fake_libtpu_addr)

  def test_get_hlo_exec_timing_with_invalid_string_attr(self):
    """Tests that HLO execution timing metrics fail with invalid string."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_EXECUTION_TIMING_DISTRIBUTION_MICROSECONDS: [
            0,
        ],
    }
    with self._fake_transfer_latency_tpu_metrics(
        responses, device_ordinal_type="string_non_digit"
    ) as fake_libtpu_addr:
      with self.assertRaisesRegex(
          ValueError,
          "device_ordinal string_attr is not a digit",
      ):
        metrics.get_hlo_exec_timing(chip_type, fake_libtpu_addr)

  def test_get_hlo_exec_timing_with_unsupported_attr_type(self):
    """Tests that HLO execution timing metrics fail with unsupported attr type."""
    chip_type = device.TpuChip.V4
    responses = {
        metrics.MetricName.HLO_EXECUTION_TIMING_DISTRIBUTION_MICROSECONDS: [
            0,
        ],
    }
    with self._fake_transfer_latency_tpu_metrics(
        responses, device_ordinal_type="unsupported_type"
    ) as fake_libtpu_addr:
      with self.assertRaisesRegex(
          TypeError,
          "Unexpected type for device_ordinal attribute",
      ):
        metrics.get_hlo_exec_timing(chip_type, fake_libtpu_addr)

  def test_get_tpuz_info(self):
    """Tests that the TPU status metrics are parsed correctly."""
    fake_response = tpu_metric_service_pb2.GetTpuRuntimeStatusResponse()
    fake_response.core_states[0].CopyFrom(
        tpu_telemetry_pb2.CurrentCoreStateSummary(
            core_id=tpu_telemetry_pb2.TpuCoreIdentifier(
                global_core_id=0,
                chip_id=0,
                core_on_chip=tpu_telemetry_pb2.TpuCoreOnChipProto(
                    type=tpu_telemetry_pb2.TpuCoreTypeProto.TPU_CORE_TYPE_TENSOR_CORE,
                    index=0,
                ),
            ),
            xdb_server_running=True,
            program_fingerprint=b"\xaa\xbb",
        )
    )
    fake_response.core_states[1].CopyFrom(
        tpu_telemetry_pb2.CurrentCoreStateSummary(
            core_id=tpu_telemetry_pb2.TpuCoreIdentifier(
                global_core_id=1,
                chip_id=0,
                core_on_chip=tpu_telemetry_pb2.TpuCoreOnChipProto(
                    type=tpu_telemetry_pb2.TpuCoreTypeProto.TPU_CORE_TYPE_TENSOR_CORE,
                    index=1,
                ),
            ),
            xdb_server_running=False,
            program_fingerprint=b"\xcc\xdd",
            error_message="Core 1 error",
        )
    )

    with self._fake_tpu_status_metrics(fake_response) as fake_libtpu_addr:
      core_states = metrics.get_tpuz_info(fake_libtpu_addr)

      expected = [
          metrics.CoreState(
              global_core_id=0,
              chip_id=0,
              core_on_chip_index=0,
              core_type="TPU_CORE_TYPE_TENSOR_CORE",
              xdb_server=True,
              sequencer_states=[],
              program_fingerprint="aabb",
              error_message=None,
          ),
          metrics.CoreState(
              global_core_id=1,
              chip_id=0,
              core_on_chip_index=1,
              core_type="TPU_CORE_TYPE_TENSOR_CORE",
              xdb_server=False,
              sequencer_states=[],
              program_fingerprint="ccdd",
              error_message="Core 1 error",
          ),
      ]
      self.assertListEqual(core_states, expected)

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_scrape_prometheus_success(self, mock_urlopen):
    mock_response = mock.create_autospec(
        http.client.HTTPResponse, instance=True
    )
    mock_response.read.return_value = textwrap.dedent("""\
        # HELP test_metric Test metric.
        # TYPE test_metric counter
        test_metric{label="val"} 1.0
        """).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    families = metrics.scrape_prometheus(9431)
    self.assertLen(families, 1)
    self.assertEqual(families[0].name, "test_metric")
    self.assertEqual(families[0].type, "counter")
    self.assertEqual(families[0].samples[0].value, 1.0)
    self.assertEqual(families[0].samples[0].labels, {"label": "val"})

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_scrape_prometheus_error(self, mock_urlopen):
    mock_urlopen.side_effect = Exception("Connection refused")
    with self.assertRaises(metrics.PrometheusConnectionError):
      metrics.scrape_prometheus(9431)

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_get_orbax_metrics(self, mock_urlopen):
    mock_response = mock.create_autospec(
        http.client.HTTPResponse, instance=True
    )
    mock_response.read.return_value = textwrap.dedent("""\
        # HELP jax_orbax_write_gbytes Total gigabytes written during a checkpoint save operation.
        # TYPE jax_orbax_write_gbytes counter
        jax_orbax_write_gbytes 10.0
        # HELP other_metric Other metric.
        # TYPE other_metric counter
        other_metric 2.0
        """).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    orbax_metrics = metrics.get_orbax_metrics()
    self.assertLen(orbax_metrics, 1)
    self.assertEqual(orbax_metrics[0].name, "jax_orbax_write_gbytes")
    mock_urlopen.assert_called_once_with(
        "http://localhost:9431/metrics", timeout=2
    )

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  @mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "invalid_port"})
  def test_get_orbax_metrics_invalid_port(self, mock_urlopen):
    mock_response = mock.create_autospec(
        http.client.HTTPResponse, instance=True
    )
    mock_response.read.return_value = b""
    mock_urlopen.return_value.__enter__.return_value = mock_response

    # Should fallback to default port (9431).
    metrics.get_orbax_metrics()
    mock_urlopen.assert_called_once_with(
        "http://localhost:9431/metrics", timeout=2
    )

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  @mock.patch.dict(os.environ, {"ORBAX_PROMETHEUS_PORT": "1234"})
  def test_get_orbax_metrics_valid_port(self, mock_urlopen):
    mock_response = mock.create_autospec(
        http.client.HTTPResponse, instance=True
    )
    mock_response.read.return_value = b""
    mock_urlopen.return_value.__enter__.return_value = mock_response

    metrics.get_orbax_metrics()
    mock_urlopen.assert_called_once_with(
        "http://localhost:1234/metrics", timeout=2
    )

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  def test_get_pygrain_metrics(self, mock_urlopen):
    mock_response = mock.create_autospec(
        http.client.HTTPResponse, instance=True
    )
    mock_response.read.return_value = textwrap.dedent("""\
        # HELP grain_python_dataset_next_duration_ns Time spent in next call for dataset.
        # TYPE grain_python_dataset_next_duration_ns counter
        grain_python_dataset_next_duration_ns 50.0
        # HELP other_metric Other metric.
        # TYPE other_metric counter
        other_metric 2.0
        """).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    pygrain_metrics = metrics.get_pygrain_metrics()
    self.assertLen(pygrain_metrics, 1)
    self.assertEqual(
        pygrain_metrics[0].name, "grain_python_dataset_next_duration_ns"
    )
    mock_urlopen.assert_called_once_with(
        "http://localhost:9431/metrics", timeout=2
    )

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  @mock.patch.dict(os.environ, {"PYGRAIN_PROMETHEUS_PORT": "invalid_port"})
  def test_get_pygrain_metrics_invalid_port(self, mock_urlopen):
    mock_response = mock.create_autospec(
        http.client.HTTPResponse, instance=True
    )
    mock_response.read.return_value = b""
    mock_urlopen.return_value.__enter__.return_value = mock_response

    # Should fallback to default port (9431).
    metrics.get_pygrain_metrics()
    mock_urlopen.assert_called_once_with(
        "http://localhost:9431/metrics", timeout=2
    )

  @mock.patch.object(urllib.request, "urlopen", autospec=True)
  @mock.patch.dict(os.environ, {"PYGRAIN_PROMETHEUS_PORT": "5678"})
  def test_get_pygrain_metrics_valid_port(self, mock_urlopen):
    mock_response = mock.create_autospec(
        http.client.HTTPResponse, instance=True
    )
    mock_response.read.return_value = b""
    mock_urlopen.return_value.__enter__.return_value = mock_response

    metrics.get_pygrain_metrics()
    mock_urlopen.assert_called_once_with(
        "http://localhost:5678/metrics", timeout=2
    )

  def test_get_percentile_direct_zero_count(self):
    val = metrics._get_percentile(
        percentile_count=0,
        total_count=0,
        buckets=[0, 0],
        scale=1.0,
        growth_factor=2.0,
    )
    self.assertEqual(val, 1.0)

  def test_get_percentile_direct_fallback(self):
    val = metrics._get_percentile(
        percentile_count=5,
        total_count=10,
        buckets=[10, 0, 0],
        scale=1.0,
        growth_factor=2.0,
    )
    self.assertEqual(val, 1.0)

  def test_getattr_valid_metrics(self):
    valid_metrics = metrics.VALID_METRICS
    self.assertIsInstance(valid_metrics, frozenset)
    self.assertIn("hbm_usage", valid_metrics)

  def test_getattr_metric_filter_schema(self):
    schemas = metrics.METRIC_FILTER_SCHEMA
    self.assertIsInstance(schemas, dict)
    self.assertIn("buffer_transfer_latency", schemas)

  def test_getattr_invalid_attribute(self):
    with self.assertRaisesRegex(
        AttributeError, r"module .* has no attribute 'NON_EXISTENT_METRIC_ATTR'"
    ):
      _ = metrics.NON_EXISTENT_METRIC_ATTR


if __name__ == "__main__":
  absltest.main()
