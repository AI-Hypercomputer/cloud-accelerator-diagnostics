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

import threading

from absl.testing import absltest
from cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader import uploader


class UploaderTest(absltest.TestCase):

  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.TensorboardExperiment"
  )
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.TensorboardExperiment.list"
  )
  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  def testThreadRunningForUploadToTensorboard(
      self,
      mock_tensorboard_list,
      mock_tensorboard,
      mock_experiment_list,
      mock_experiment,
  ):
    mock_tensorboard_instance = mock_tensorboard.return_value
    mock_tensorboard_instance.display_name = "test-instance"
    mock_tensorboard_instance.name = "123"
    mock_tensorboard_list.return_value = [mock_tensorboard_instance]

    mock_experiment_instance = mock_experiment.return_value
    mock_experiment_instance.display_name = "test-experiment"
    mock_experiment_list.return_value = [mock_experiment_instance]

    uploader.start_upload_to_tensorboard(
        "test-project",
        "us-central1",
        "test-experiment",
        "test-instance",
        "logdir",
    )
    self.assertEqual(threading.active_count(), 2)

    uploader.stop_upload_to_tensorboard()
    self.assertEqual(threading.active_count(), 1)

  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  def testThreadNotRunningForUploadWhenNoTensorboardExist(
      self, mock_tensorboard_list
  ):
    mock_tensorboard_list.return_value = []

    with self.assertLogs(level="ERROR") as log:
      uploader.start_upload_to_tensorboard(
          "test-project",
          "us-central1",
          "test-experiment",
          "test-instance",
          "logdir",
      )

    self.assertEqual(threading.active_count(), 1)
    self.assertRegex(
        log.output[0],
        "No Tensorboard instance with the name test-instance present in the"
        " project test-project.",
    )

    uploader.stop_upload_to_tensorboard()
    self.assertEqual(threading.active_count(), 1)

  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.TensorboardExperiment.list"
  )
  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  def testThreadNotRunningForUploadWhenNoExperimentExist(
      self, mock_tensorboard_list, mock_tensorboard, mock_experiment_list
  ):
    mock_tensorboard_instance = mock_tensorboard.return_value
    mock_tensorboard_instance.display_name = "test-instance"
    mock_tensorboard_instance.name = "123"
    mock_tensorboard_list.return_value = [mock_tensorboard_instance]
    mock_experiment_list.return_value = []

    with self.assertLogs(level="ERROR") as log:
      uploader.start_upload_to_tensorboard(
          "test-project",
          "us-central1",
          "test-experiment",
          "test-instance",
          "logdir",
      )

    self.assertEqual(threading.active_count(), 1)
    self.assertRegex(
        log.output[0],
        "No Tensorboard experiment with the name test-experiment present in"
        " the project test-project.",
    )

    uploader.stop_upload_to_tensorboard()
    self.assertEqual(threading.active_count(), 1)


if __name__ == "__main__":
  absltest.main()
