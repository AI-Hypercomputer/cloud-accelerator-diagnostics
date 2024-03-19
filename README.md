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
# Cloud Accelerator Diagnostics

## Overview
Cloud Accelerator Diagnostics is a library to monitor, debug and profile the workloads running on Cloud accelerators like TPUs and GPUs. Additionally, this library provides a streamlined approach to automatically upload data to Tensorboard Experiments in Vertex AI. The package allows users to create a Tensorboard instance and Experiments in Vertex AI, and upload logs to them.

## Installation
To install the Cloud Accelerator Diagnostics package, run the following command:

 ```bash
 pip install cloud-accelerator-diagnostics
 ```

## Automating Uploads to Vertex AI Tensorboard
Before creating and uploading logs to Vertex AI Tensorboard, you must enable [Vertex AI API](https://cloud.google.com/vertex-ai/docs/start/cloud-environment#enable_vertexai_apis) in your Google Cloud console. Also, make sure to assign the [Vertex AI User IAM role](https://cloud.google.com/vertex-ai/docs/general/access-control#aiplatform.user) to the service account that will call the APIs in `cloud-accelerator-diagnostics` package. This is required to create and access the Vertex AI Tensorboard in the Google Cloud console.

### Create Vertex AI Tensorboard
To learn about Vertex AI Tensorboard, visit this [page](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-introduction).

Here is an example script to create a Vertex AI Tensorboard instance with the name `test-instance` in Google Cloud Project `test-project`.

Note: Vertex AI is available in only [these](https://cloud.google.com/vertex-ai/docs/general/locations#available-regions) regions.

```
from cloud_accelerator_diagnostics import tensorboard

instance_id = tensorboard.create_instance(project="test-project",
                                          location="us-central1",
                                          tensorboard_name="test-instance")
print("Vertex AI Tensorboard created: ", instance_id)
```

### Create Vertex AI Experiment
To learn about Vertex AI Experiments, visit this [page]
(https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments).

The following script will create a Vertex AI Experiment named `test-experiment` in your Google Cloud Project `test-project`. Here's how it handles attaching a Tensorboard instance:

**Scenario 1: Tensorboard Instance Exist**

If a Tensorboard instance named `test-instance` already exists in your project, the script will attach it to the new Experiment.

**Scenario 2: No Tensorboard Instance Present**

If `test-instance` does not exist, the script will create a new Tensorboard instance with that name and attach it to the Experiment.

```
from cloud_accelerator_diagnostics import tensorboard

instance_id, tensorboard_url = tensorboard.create_experiment(project="test-project",
                                                             location="us-central1",
                                                             experiment_name="test-experiment",
                                                             tensorboard_name="test-instance")

print("View your Vertex AI Tensorboard here: ", tensorboard_url)
```

If a Vertex AI Experiment with the specified name exists, a new one will not be created, and the existing Experiment's URL will be returned.

Note: You can attach multiple Vertex AI Experiments to a single Vertex AI Tensorboard.

### Upload Logs to Vertex AI Tensorboard
The following script will continuously monitor for new data in the directory (`logdir`), and uploads it to your Vertex AI Tensorboard Experiment. Note that after calling `start_upload_to_tensorboard()`, the thread will be kept alive even if an exception is thrown. To ensure the thread gets shut down, put any code after `start_upload_to_tensorboard()` and before `stop_upload_to_tensorboard()` in a `try` block, and call `stop_upload_to_tensorboard()` in `finally` block. This example shows how you can upload the [profile logs](https://jax.readthedocs.io/en/latest/profiling.html#programmatic-capture) collected for your JAX workload on Vertex AI Tensorboard.

```
from cloud_accelerator_diagnostics import uploader

uploader.start_upload_to_tensorboard(project="test-project",
                                     location="us-central1",
                                     experiment_name="test-experiment",
                                     tensorboard_name="test-instance",
                                     logdir="gs://test-directory/testing")
try:
  jax.profiler.start_trace("gs://test-directory/testing")
  <your code goes here>
  jax.profiler.stop_trace()
finally:
  uploader.stop_upload_to_tensorboard()
```
