<!--
 Copyright 2023 Google LLC
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
      https://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->
# `tpu-info` CLI

`tpu-info` is a simple CLI tool for detecting Cloud TPU devices and reading
runtime metrics from `libtpu`, including memory usage and duty cycle. It
supports both a static, one-time snapshot and a live streaming mode to monitor
metrics continuously.

**Note**: to access `libtpu` utilization metrics, you must have a workload
running with a supported ML framework, such as JAX or PyTorch/XLA. See the
[Usage](#usage) section for more information.

***

## What's New in Version 0.8.1

🚀 **New Features**

* Metrics `hlo_exec_timing` & `hlo_queue_size` introduced in `0.8.0` work for
  some older versions of `libtpu`
  - It is still recommended to update to most recent `libtpu` version for full
    support in all metrics available in `tpu-info`
* Addresses a `DeprecationWarning` in Python 3.11+ related to classes in `enum`.

***

## Installing

Install the latest release using `pip`:

```
pip install tpu-info
```

Alternatively, install `tpu-info` from source:

```bash
pip install git+https://github.com/google/cloud-accelerator-diagnostics/#subdirectory=tpu_info
```

***

## Usage

To view current TPU utilization data, `tpu-info` requires a running TPU workload
with supported ML framework[^1] such as JAX or PyTorch/XLA. For example:

```
# JAX
>>> import jax
>>> jax.device_count()
4
# Create a tensor on the TPU
>>> t = jax.numpy.ones((300, 300))

# PyTorch/XLA
>>> import torch
>>> import torch_xla
>>> t = torch.randn((300, 300), device=torch_xla.device())
```

Then, on the same machine, you can run the `tpu-info` command in your terminal.

### Static Mode

Run the following command for a one-time snapshot of the current metrics.

```bash
$ tpu-info
Libtpu version: 0.0.19.dev20250721+nightly
Accelerator type: v6e

TPU Chips
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Chip         ┃ Type         ┃ Devices ┃ PID    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╕━━━━━━━━━╕━━━━━━━━┩
│ /dev/vfio/0  │ TPU v6e chip │ 1       │ 1052   │
│ /dev/vfio/1  │ TPU v6e chip │ 1       │ 1052   │
│ /dev/vfio/2  │ TPU v6e chip │ 1       │ 1052   │
│ /dev/vfio/3  │ TPU v6e chip │ 1       │ 1052   │
└──────────────┴──────────────┴─────────┴────────┘
TPU Runtime Utilization
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip   ┃ HBM usage                ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╕━━━━━━━━━━━━┩
│ 8      │ 18.45 GiB / 31.25 GiB    │    100.00% │
│ 9      │ 10.40 GiB / 31.25 GiB    │    100.00% │
│ 12     │ 10.40 GiB / 31.25 GiB    │    100.00% │
│ 13     │ 10.40 GiB / 31.25 GiB    │    100.00% │
└────────┴──────────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │                  13.60%│
│ 1       │                  14.81%│
│ 2       │                  14.36%│
│ 3       │                  13.60%│
└─────────┴────────────────────────┘
TPU Buffer Transfer Latency
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Buffer Size  ┃ P50          ┃ P90          ┃ P95          ┃ P999         ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╕━━━━━━━━━━━━━━╕━━━━━━━━━━━━━━╕━━━━━━━━━━━━━━┩
│ 8MB+         │ 108978.82 us │ 164849.81 us │ 177366.42 us │ 212419.07 us │
│ 4MB+         │ 21739.38 us  │ 38126.84 us  │ 42110.12 us  │ 55474.21 us  │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
TPU gRPC TCP Minimum RTT
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ P50      ┃ P90      ┃ P95      ┃ P999     ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ 35.99 us │ 52.15 us │ 53.83 us │ 55.51 us │
└──────────┴──────────┴──────────┴──────────┘
TPU gRPC TCP Delivery Rate
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ P50           ┃ P90           ┃ P95           ┃ P999          ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 12305.96 Mbps │ 18367.10 Mbps │ 24872.11 Mbps │ 44841.55 Mbps │
└───────────────┴───────────────┴───────────────┴───────────────┘
```

### Streaming Mode

You can run `tpu-info` in a streaming mode to periodically refresh and display
the utilization statistics.

```bash
# Refresh stats every 2 seconds
tpu-info --streaming --rate 2
```
```
Refresh rate: 0.1s
Last update: 2025-07-24 11:00:59 UTC
Libtpu version: 0.0.19.dev20250721+nightly
Accelerator type: v6e

TPU Chips
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Chip         ┃ Type         ┃ Devices ┃ PID    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╢━━━━━━━━━╢━━━━━━━━┪
│ /dev/vfio/0  │ TPU v6e chip │ 1       │ 1022   │
│ /dev/vfio/1  │ TPU v6e chip │ 1       │ 1022   │
│ /dev/vfio/2  │ TPU v6e chip │ 1       │ 1022   │
│ /dev/vfio/3  │ TPU v6e chip │ 1       │ 1022   │
└──────────────┴──────────────┴─────────┴────────┘
TPU Runtime Utilization
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip   ┃ HBM usage                ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╕━━━━━━━━━━━━┩
│ 8      │ 17.26 GiB / 31.25 GiB    │    100.00% │
│ 9      │  9.26 GiB / 31.25 GiB    │    100.00% │
│ 12     │  9.26 GiB / 31.25 GiB    │    100.00% │
│ 13     │  9.26 GiB / 31.25 GiB    │    100.00% │
└────────┴──────────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │                  15.17%│
│ 1       │                  14.62%│
│ 2       │                  14.68%│
│ 3       │                  15.14%│
└─────────┴────────────────────────┘
TPU Buffer Transfer Latency
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Buffer Size  ┃ P50          ┃ P90          ┃ P95          ┃ P999         ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╕━━━━━━━━━━━━━━╕━━━━━━━━━━━━━━╕━━━━━━━━━━━━━━┩
│ 8MB+         │ 18264.03 us  │ 33263.06 us  │ 35990.98 us  │ 53997.32 us  │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
TPU gRPC TCP Minimum RTT
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ P50      ┃ P90      ┃ P95      ┃ P999     ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ 35.99 us │ 52.15 us │ 53.83 us │ 55.51 us │
└──────────┴──────────┴──────────┴──────────┘
TPU gRPC TCP Delivery Rate
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ P50           ┃ P90           ┃ P95           ┃ P999          ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 12305.96 Mbps │ 18367.10 Mbps │ 24872.11 Mbps │ 44841.55 Mbps │
└───────────────┴───────────────┴───────────────┴───────────────┘
```

### Version

To check the installed version of `tpu-info`, libtpu version and accelerator
type of the TPU chip, use the `--version` or `-v` flag.

##### Compatible Environment:
```bash
$ tpu-info --version
- tpu-info version: 0.8.0
- libtpu version: 0.0.18
- accelerator type: v6e
```

##### Incompatible Environment (Python 3.12+):
```bash
$ tpu-info --version
- tpu-info version: 0.8.0
- libtpu version: N/A (incompatible environment)
- accelerator type: N/A (incompatible environment)
```

### Process

You can use the `--process` or `-p` flag to display information about the
processes currently running on the TPU.

```bash
$ tpu-info --process
TPU Process Info
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Chip        ┃ PID    ┃ Process Name ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ /dev/vfio/0 │ 799657 │ python3      │
│ /dev/vfio/1 │ 799657 │ python3      │
│ /dev/vfio/2 │ 799657 │ python3      │
│ /dev/vfio/3 │ 799657 │ python3      │
│ /dev/vfio/4 │ 799657 │ python3      │
│ /dev/vfio/5 │ 799657 │ python3      │
│ /dev/vfio/6 │ 799657 │ python3      │
│ /dev/vfio/7 │ 799657 │ python3      │
└─────────────┴────────┴──────────────┘
```

### List Metrics

You can use the `--list_metrics` flag to display all supported metrics that can
be given along with the `--metric` flag.

```bash
$ tpu-info --list_metrics
╭─ Supported Metrics ─────────────────────────────────────────────────────────────────────────────╮
│         buffer_transfer_latency                                                                 │
│         collective_e2e_latency                                                                  │
│         core_state                                                                              │
│         device_to_host_transfer_latency                                                         │
│         duty_cycle_percent                                                                      │
│         grpc_tcp_delivery_rate                                                                  │
│         grpc_tcp_min_rtt                                                                        │
│         hbm_usage                                                                               │
│         hlo_exec_timing                                                                         │
│         hlo_queue_size                                                                          │
│         host_to_device_transfer_latency                                                         │
│         queued_programs                                                                         │
│         sequencer_state                                                                         │
│         sequencer_state_detailed                                                                │
│         tensorcore_utilization                                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Metric

You can use the `--metric` flag to display specific metrics. You can specify
multiple metrics separated by spaces with multiple `--metric` flags.

```bash
$ tpu-info --metric duty_cycle_percent --metric hbm_usage
TPU Duty Cycle
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ Duty Cycle (%) ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 0       │ 100.00%        │
│ 1       │ 100.00%        │
│ 2       │ 100.00%        │
│ 3       │ 100.00%        │
│ 4       │ 100.00%        │
│ 5       │ 100.00%        │
│ 6       │ 100.00%        │
│ 7       │ 100.00%        │
└─────────┴────────────────┘
TPU HBM Usage
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Chip   ┃ HBM Usage (GiB)       ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0      │ 29.50 GiB / 31.25 GiB │
│ 1      │ 21.50 GiB / 31.25 GiB │
│ 2      │ 21.50 GiB / 31.25 GiB │
│ 3      │ 21.50 GiB / 31.25 GiB │
│ 4      │ 21.50 GiB / 31.25 GiB │
│ 5      │ 21.50 GiB / 31.25 GiB │
│ 6      │ 21.50 GiB / 31.25 GiB │
│ 7      │ 21.50 GiB / 31.25 GiB │
└────────┴───────────────────────┘
```

[^1]: Releases from before 2024 may not be compatible.
