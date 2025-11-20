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
import dataclasses
import enum
import itertools
import typing
from typing import Any, Dict, List, Optional

from tpu_info import device
import grpc

from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics
from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc
from tpu_info.proto import tpu_telemetry_pb2


@dataclasses.dataclass
class QueuedProgram:
  """Structured data for a program queued for execution."""

  run_id: int
  launch_id: int
  program_fingerprint: str


@dataclasses.dataclass
class SequencerState:
  """Structured data for the state of a sequencer of a single TPU core."""

  sequencer_type: str
  sequencer_index: int
  pc: int
  tag: int
  tracemark: int
  program_id: int
  run_id: int
  hlo_location: Optional[str] = None
  hlo_detailed_info: Optional[str] = None


@dataclasses.dataclass
class CoreState:
  """Structured data for the state of a single TPU core."""

  global_core_id: int
  chip_id: int
  core_on_chip_index: int
  core_type: str
  xdb_server: bool
  sequencer_states: List[SequencerState]
  program_fingerprint: str
  error_message: Optional[str] = None
  queued_programs: List[QueuedProgram] = dataclasses.field(default_factory=list)


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
  GRPC_TCP_MIN_RTT_US = "megascale.grpc_tcp_min_rtt.microsecond.cumulative.distribution"
  GRPC_TCP_DELIVERY_RATE_MBPS = "megascale.grpc_tcp_delivery_rate.Mbps.cumulative.distribution"


class Usage(typing.NamedTuple):
  """Usage measurements for a TPU device."""

  device_id: int
  memory_usage: int
  total_memory: int
  duty_cycle_pct: float


class TransferLatencyDistribution(typing.NamedTuple):
  """Distribution measurements."""

  transfer_size: str
  p50: float
  p90: float
  p95: float
  p999: float


# A schema defining the allowed filter keys for each metric.
# The value is a set of valid filter keys.
METRIC_FILTER_SCHEMA = {
    "buffer_transfer_latency": {"percentile"},
    "host_to_device_transfer_latency": {"percentile"},
    "device_to_host_transfer_latency": {"percentile"},
    "collective_e2e_latency": {"percentile"},
}

# A set of all valid metric names for quick lookup.
VALID_METRICS = {
    "hbm_usage",
    "duty_cycle_percent",
    "tensorcore_utilization",
    "buffer_transfer_latency",
    "host_to_device_transfer_latency",
    "device_to_host_transfer_latency",
    "collective_e2e_latency",
    "grpc_tcp_min_rtt",
    "grpc_tcp_delivery_rate",
    "core_state",
    "sequencer_state",
    "sequencer_state_detailed",
    "queued_programs",
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
    "grpc_tcp_min_rtt": MetricName.GRPC_TCP_MIN_RTT_US.value,
    "grpc_tcp_delivery_rate": MetricName.GRPC_TCP_DELIVERY_RATE_MBPS.value,
}


def get_chip_usage_new(
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
  duty_cycle_pct_per_core = duty_cycle_pct

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
    filters: Optional[Dict[str, Any]] = None,
) -> List[TransferLatencyDistribution]:
  """Gets latency statistics for all attached TPU devices based on the metric name.

  Args:
    metric_arg: Metric name to fetch.
    addr: GRPC address of libtpu metrics server.
    filters: Optional filters to apply to the metric request. Currently unused.

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

    attributes = attribute.value.kvlist_attr.attributes
    transfer_size = attributes[0].value.string_attr if attributes else "N/A"
    latency_distributions.append(
        TransferLatencyDistribution(
            transfer_size,
            p50,
            p90,
            p95,
            p999,
        )
    )

  return latency_distributions


def get_tpuz_info(
    addr: str = "localhost:8431",
    include_hlo_info: bool = False,
) -> List[CoreState]:
  """Gets TPUz info from libtpu and parses it into CoreState objects.

  Args:
    addr: GRPC address of libtpu metrics server.
    include_hlo_info: Whether to include HLO info in the response. This can be
      computationally expensive so it is disabled by default.

  Returns:
    List of CoreState objects, one for each TPU core.
  """
  channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
  client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)
  status_request = tpu_metrics.GetTpuRuntimeStatusRequest(
      include_hlo_info=include_hlo_info
  )
  response = client.GetTpuRuntimeStatus(status_request)

  core_states = []
  # The map keys are global core IDs. Iterate in sorted order for consistency.
  for core_id in sorted(response.core_states.keys()):
    core_summary = response.core_states[core_id]

    # Parse sequencer information.
    sequencers = []
    for seq_info in core_summary.sequencer_info:
      sequencers.append(
          SequencerState(
              sequencer_type=tpu_telemetry_pb2.TpuSequencerTypeProto.Name(
                  seq_info.sequencer_type
              ),
              sequencer_index=seq_info.sequencer_index,
              pc=seq_info.pc,
              tag=seq_info.tag,
              tracemark=seq_info.tracemark,
              program_id=seq_info.program_id,
              run_id=seq_info.run_id,
              hlo_location=(
                  seq_info.hlo_location
                  if seq_info.HasField("hlo_location")
                  else None
              ),
              hlo_detailed_info=(
                  seq_info.hlo_detailed_info
                  if seq_info.HasField("hlo_detailed_info")
                  else None
              ),
          )
      )

    queued_programs = []
    for prog_info in core_summary.queued_program_info:
      queued_programs.append(
          QueuedProgram(
              run_id=prog_info.run_id,
              launch_id=prog_info.launch_id,
              program_fingerprint=prog_info.program_fingerprint.hex(),
          )
      )

    # Parse core information.
    core_state = CoreState(
        global_core_id=core_id,
        chip_id=core_summary.core_id.chip_id,
        core_on_chip_index=core_summary.core_id.core_on_chip.index,
        core_type=tpu_telemetry_pb2.TpuCoreTypeProto.Name(
            core_summary.core_id.core_on_chip.type
        ),
        xdb_server=core_summary.xdb_server_running,
        sequencer_states=sequencers,
        program_fingerprint=core_summary.program_fingerprint.hex(),
        error_message=(
            core_summary.error_message if core_summary.HasField("error_message")
            else None
        ),
        queued_programs=queued_programs,
    )
    core_states.append(core_state)

  return core_states
