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
runtime metrics from `libtpu`, including memory usage.

Note: to access `libtpu` utilization metrics, you must have a workload running
with a supported ML framework, such as JAX or PyTorch/XLA. See the
[Usage](#usage) section for more information.

## Installing

Install the latest release using `pip`:

```
pip install tpu-info
```

Alternatively, install `tpu-info` from source:

```bash
pip install git+https://github.com/google/cloud-accelerator-diagnostics/#subdirectory=tpu_info
```

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

Then, on the same machine, run the `tpu-info` command line tool:

```bash
$ tpu-info
TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Chip        ┃ Type        ┃ Devices ┃ PID    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ /dev/accel0 │ TPU v4 chip │ 1       │ 130007 │
│ /dev/accel1 │ TPU v4 chip │ 1       │ 130007 │
│ /dev/accel2 │ TPU v4 chip │ 1       │ 130007 │
│ /dev/accel3 │ TPU v4 chip │ 1       │ 130007 │
└─────────────┴─────────────┴─────────┴────────┘
Connected to libtpu at grpc://localhost:8431...
TPU Utilization
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Device ┃ Memory usage         ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0      │ 0.00 GiB / 31.75 GiB │      0.00% │
│ 1      │ 0.00 GiB / 31.75 GiB │      0.00% │
│ 2      │ 0.00 GiB / 31.75 GiB │      0.00% │
│ 3      │ 0.00 GiB / 31.75 GiB │      0.00% │
└────────┴──────────────────────┴────────────┘
```

[^1]: Releases from before 2024 may not be compatible.
