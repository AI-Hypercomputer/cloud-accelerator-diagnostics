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

"""Client library for libtpu runtime metrics."""

import enum
import itertools
import typing
from typing import List

from tpu_info import device
import grpc

from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics
from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc


class MetricName(enum.Enum):
  """Metric names defined in libtpu."""

  TOTAL_MEMORY = "tpu.runtime.hbm.memory.total.bytes"
  MEMORY_USAGE = "tpu.runtime.hbm.memory.usage.bytes"
  DUTY_CYCLE_PCT = "tpu.runtime.tensorcore.dutycycle.percent"
  BUFFER_TRANSFER_LATENCY_US = (
      "megascale.dcn_transfer_latencies.microsecond.cumulative.distribution"
  )
  HOST_TO_DEVICE_TRANSFER_LATENCY_US = "megascale.host_to_device_transfer_latencies.microsecond.cumulative.distribution"
  DEVICE_TO_HOST_TRANSFER_LATENCY_US = "megascale.device_to_host_transfer_latencies.microsecond.cumulative.distribution"
  COLLECTIVE_E2E_LATENCY_US = "megascale.collective_end_to_end_latencies.microsecond.cumulative.distribution"


class Usage(typing.NamedTuple):
  """Usage measurements for a TPU device."""

  device_id: int
  memory_usage: int
  total_memory: int
  duty_cycle_pct: float


class TransferLatencyDistribution(typing.NamedTuple):
  """Distribution measurements."""

  buffer_size: str
  p50: float
  p90: float
  p95: float
  p999: float


PERCENTILE_PARAM = {"percentile": {"p50", "p90", "p95", "p999"}}

VALID_METRICS_WITH_PARAMS = {
    "hbm_usage": None,
    "duty_cycle_percent": None,
    "tensorcore_utilization": None,
    "buffer_transfer_latency": PERCENTILE_PARAM,
    "host_to_device_transfer_latency": PERCENTILE_PARAM,
    "device_to_host_transfer_latency": PERCENTILE_PARAM,
    "collective_e2e_latency": PERCENTILE_PARAM,
}

LIBTPU_METRIC_MAP = {
    "buffer_transfer_latency": MetricName.BUFFER_TRANSFER_LATENCY_US.value,
    "host_to_device_transfer_latency": (
        MetricName.HOST_TO_DEVICE_TRANSFER_LATENCY_US.value
    ),
    "device_to_host_transfer_latency": (
        MetricName.DEVICE_TO_HOST_TRANSFER_LATENCY_US.value
    ),
    "collective_e2e_latency": MetricName.COLLECTIVE_E2E_LATENCY_US.value,
}


def get_chip_usage(
    chip_type: device.TpuChip, addr: str = "localhost:8431"
) -> List[Usage]:
  """Gets usage statistics for all attached TPU devices.

  Args:
    chip_type: TPU chip version. Determines how metrics are interpreted.
    addr: GRPC address of libtpu metrics server.

  Returns:
    List of usage statistics for each TPU device.
  """
  channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
  client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)

  def sorted_metric_response(
      metric_name: MetricName,
  ) -> List[tpu_metrics.Metric]:
    # Manually annotate type until GRPC supports annotations
    # See https://github.com/grpc/grpc/issues/29041
    resp: tpu_metrics.MetricResponse = client.GetRuntimeMetric(
        tpu_metrics.MetricRequest(metric_name=metric_name.value)
    )
    return sorted(resp.metric.metrics, key=lambda m: m.attribute.value.int_attr)

  totals = sorted_metric_response(MetricName.TOTAL_MEMORY)
  usages = sorted_metric_response(MetricName.MEMORY_USAGE)
  duty_cycle_pct = sorted_metric_response(MetricName.DUTY_CYCLE_PCT)

  # Duty cycle is always measured per-chip, while memory is measured per-core.
  # Repeat if necessary so these responses are the same length.
  duty_cycle_pct_per_core = list(
      itertools.chain.from_iterable(
          itertools.repeat(d, chip_type.value.devices_per_chip)
          for d in duty_cycle_pct
      )
  )

  assert (
      len(totals) == len(usages) == len(duty_cycle_pct_per_core)
  ), "Metrics not found for all chips"

  return [
      Usage(
          u.attribute.value.int_attr,
          u.gauge.as_int,
          t.gauge.as_int,
          d.gauge.as_double,
      )
      for u, t, d in zip(usages, totals, duty_cycle_pct_per_core)
  ]


def _get_percentile(
    percentile_count: int,
    total_count: int,
    buckets: List[int],
    scale: float,
    growth_factor: float,
) -> float:
  """Gets a percentile value from a distribution."""
  for i in range(len(buckets) - 1, 0, -1):
    total_count -= buckets[i]
    if total_count <= percentile_count:
      delta = percentile_count - total_count
      lower_bound = scale * (growth_factor ** (i - 1))
      return lower_bound * (1 + (delta / buckets[i]) * (growth_factor - 1))
  return 1


def get_transfer_latency(
    metric_arg: str,
    addr: str = "localhost:8431",
) -> List[TransferLatencyDistribution]:
  """Gets latency statistics for all attached TPU devices based on the metric name.

  Args:
    metric_arg: Metric name to fetch.
    addr: GRPC address of libtpu metrics server.

  Returns:
    List of latency statistics for each TPU device.
  """
  channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
  client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)

  metric_name = LIBTPU_METRIC_MAP[metric_arg]
  resp: tpu_metrics.MetricResponse = client.GetRuntimeMetric(
      tpu_metrics.MetricRequest(metric_name=metric_name)
  )

  latency_distributions = []

  for metric in resp.metric.metrics:
    attribute = metric.attribute
    distribution = metric.distribution
    bucket = list(distribution.bucket_counts)
    count = distribution.count
    scale = distribution.bucket_options.exponential_buckets.scale
    growth_factor = (
        distribution.bucket_options.exponential_buckets.growth_factor
    )

    p50_count = int(count * 0.5)
    p90_count = int(count * 0.9)
    p95_count = int(count * 0.95)
    p999_count = int(count * 0.999)

    p50 = _get_percentile(p50_count, count, bucket, scale, growth_factor)
    p90 = _get_percentile(p90_count, count, bucket, scale, growth_factor)
    p95 = _get_percentile(p95_count, count, bucket, scale, growth_factor)
    p999 = _get_percentile(p999_count, count, bucket, scale, growth_factor)

    latency_distributions.append(
        TransferLatencyDistribution(
            attribute.value.kvlist_attr.attributes[0].value.string_attr,
            p50,
            p90,
            p95,
            p999,
        )
    )

  return latency_distributions
