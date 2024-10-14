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
# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the changes to be released.

-->

# [0.1.1] - 2024-10-14
* Version 0.1.1 of `cloud-accelerator-diagnostics` PyPI package
* Features:
  * Use Vertex AI's continuous uploader directly

# [0.1.0] - 2024-03-20
* Initial release of `cloud-accelerator-diagnostics` PyPI package
* Features:
  * Create a Vertex AI Tensorboard instance in Google Cloud Project
  * Create a Vertex AI Experiment in Google Cloud Project
  * Automatically upload logs to Vertex AI Tensorboard Experiment
