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
with a supported ML framework, such as JAX or PyTorch/XLA with `libtpu`'s
runtime metrics enabled. See the [Usage](#usage) section for more information.

## Installing

To build the tool, use `pip` :

```bash
# Install pip if you don't already have it
sudo apt install python3-pip python-is-python3
# Install directly on current machine
pip install .
# Build the wheel to install on another machine
pip wheel --no-deps .
```

## Usage

To view current TPU utilization data, `tpu-info` requires a running TPU workload
with supported ML framework[^1] such as JAX or PyTorch/XLA. To enable the
metrics server within `libtpu`, set the `TPU_RUNTIME_METRICS_PORTS` environment
variable accordingly:

```bash
# Single process (JAX and PyTorch/XLA SPMD)
TPU_RUNTIME_METRICS_PORTS=8431 python script.py
# PyTorch/XLA multiprocess
TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434 python script.py
```

Then, on the same machine, run the `tpu-info` command line tool:

```bash
$ tpu-info
TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┓
┃ Device      ┃ Type        ┃ Cores ┃ PID     ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━┩
│ /dev/accel0 │ TPU v4 chip │ 1     │ 3786361 │
│ /dev/accel1 │ TPU v4 chip │ 1     │ 3786361 │
│ /dev/accel2 │ TPU v4 chip │ 1     │ 3786361 │
│ /dev/accel3 │ TPU v4 chip │ 1     │ 3786361 │
└─────────────┴─────────────┴───────┴─────────┘
Connected to libtpu at `localhost:8431`...
TPU Chip Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Core ID ┃ Memory usage         ┃ Duty cycle ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0       │ 0.00 GiB / 30.75 GiB │      0.00% │
│ 1       │ 0.00 GiB / 30.75 GiB │      0.00% │
│ 2       │ 0.00 GiB / 30.75 GiB │      0.00% │
│ 3       │ 0.00 GiB / 30.75 GiB │      0.00% │
└─────────┴──────────────────────┴────────────┘
```

[^1]: Releases from before 2024 may not be compatible.
