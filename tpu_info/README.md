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
runtime metrics from `libtpu`, including memory usage and duty cycle. It supports both a static, one-time snapshot and a live streaming mode to monitor metrics continuously.

**Note**: to access `libtpu` utilization metrics, you must have a workload running
with a supported ML framework, such as JAX or PyTorch/XLA. See the
[Usage](#usage) section for more information.

***

## What's New in Version 0.4.0

🚀 **New Features**

* **Streaming Mode**: Introduced a live streaming mode using the `--streaming` flag for continuous monitoring.
* **Version Flag**: Added `--version` (and `-v`) flags to easily check the tool's installed version.

🐛 **Bug Fixes & Compatibility**

* `tpu-info` is now more robust and avoids crashes by maintaining backward compatibility with different versions of the underlying `libtpu` library.

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
┃ Device ┃ HBM usage                ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╕━━━━━━━━━━━━┩
│ 8      │ 18.45 GiB / 31.25 GiB    │    100.00% │
│ 9      │ 10.40 GiB / 31.25 GiB    │    100.00% │
│ 12     │ 10.40 GiB / 31.25 GiB    │    100.00% │
│ 13     │ 10.40 GiB / 31.25 GiB    │    100.00% │
└────────┴──────────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Chip ID ┃ TensorCore Utilization ┃
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
```

### Streaming Mode

You can run `tpu-info` in a streaming mode to periodically refresh and display the utilization statistics.

```bash
# Refresh stats every 2 seconds
tpu-info --streaming --rate 2
```
```
                                  Refresh rate: 0.1s
                                  Last update: 2025-06-30 02:36:14

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
┃ Device ┃ HBM usage                ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╕━━━━━━━━━━━━┩
│ 8      │ 17.26 GiB / 31.25 GiB    │    100.00% │
│ 9      │  9.26 GiB / 31.25 GiB    │    100.00% │
│ 12     │  9.26 GiB / 31.25 GiB    │    100.00% │
│ 13     │  9.26 GiB / 31.25 GiB    │    100.00% │
└────────┴──────────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Chip ID ┃ TensorCore Utilization ┃
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
```

### Version

To check the installed version of `tpu-info`, use the `--version` or `-v` flag.

```bash
$ tpu-info --version
tpu-info version: 0.4.0
```

[^1]: Releases from before 2024 may not be compatible.
