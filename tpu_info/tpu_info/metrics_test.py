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

from concurrent import futures
import contextlib
import functools
import itertools
from typing import Dict, List, Union

from absl.testing import absltest
from absl.testing import parameterized
from tpu_info import device
from tpu_info import metrics
import grpc

from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics
from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc


class FakeTpuMetrics(tpu_metrics_grpc.RuntimeMetricServiceServicer):

  def __init__(
      self, responses: Dict[metrics.MetricName, List[Union[int, float]]]
  ):
    self._responses = responses
    super().__init__()

  @functools.singledispatchmethod
  def _gauge(self, val):
    raise NotImplementedError(f"Cannot create Gauge from {val}")

  @_gauge.register
  def _(self, val: int):
    return tpu_metrics.Gauge(as_int=val)

  @_gauge.register
  def _(self, val: float):
    return tpu_metrics.Gauge(as_double=val)

  def GetRuntimeMetric(self, request: tpu_metrics.MetricRequest, context):
    metric_name = metrics.MetricName(request.metric_name)
    resp = self._responses[metric_name]

    return tpu_metrics.MetricResponse(
        metric=tpu_metrics.TPUMetric(
            name=metric_name.value,
            metrics=[
                tpu_metrics.Metric(
                    attribute=tpu_metrics.Attribute(
                        key="device-id", value=tpu_metrics.AttrValue(int_attr=i)
                    ),
                    gauge=self._gauge(val),
                )
                for i, val in enumerate(resp)
            ],
        )
    )


class TestMetrics(parameterized.TestCase):

  @staticmethod
  @contextlib.contextmanager
  def _fake_tpu_metrics(responses) -> str:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    tpu_metrics_grpc.add_RuntimeMetricServiceServicer_to_server(
        FakeTpuMetrics(responses), server
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
    with self._fake_tpu_metrics(responses) as fake_libtpu_addr:
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


if __name__ == "__main__":
  absltest.main()
