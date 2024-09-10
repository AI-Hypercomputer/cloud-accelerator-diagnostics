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
      "cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader.uploader.tensorboard"
  )
  @absltest.mock.patch(
      "cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader.uploader.aiplatform"
  )
  def testWhenUploadToTensorboardThenVertexUploaderIsCalled(
      self,
      mock_aiplatform,
      mock_tensorboard,
  ):
    # given
    mock_tensorboard.get_instance_identifiers.return_value = ["test_experiment"]
    mock_tensorboard.get_experiment.return_value = "test-experiment"

    # when
    uploader.start_upload_to_tensorboard(
        "test-project",
        "us-central1",
        "test-experiment",
        "test-instance",
        "logdir",
    )

    # then
    mock_aiplatform.init.assert_called_once_with(
        project="test-project", location="us-central1"
    )
    mock_aiplatform.start_upload_tb_log.assert_called_once_with(
        tensorboard_id="test_experiment",
        tensorboard_experiment_name="test-experiment",
        logdir="logdir",
    )

  @absltest.mock.patch(
      "cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader.uploader.tensorboard"
  )
  @absltest.mock.patch(
      "cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader.uploader.aiplatform"
  )
  def testWhenNoTensorboardExistsThenVertexUploaderNotCalled(
      self,
      mock_aiplatform,
      mock_tensorboard,
  ):
    # given
    mock_tensorboard.get_instance_identifiers.return_value = []

    # when
    with self.assertLogs(level="ERROR") as log:
      uploader.start_upload_to_tensorboard(
          "test-project",
          "us-central1",
          "test-experiment",
          "test-instance",
          "logdir",
      )

    # then
    self.assertEqual(threading.active_count(), 1)
    self.assertRegex(
        log.output[0],
        "No Tensorboard instance with the name test-instance present in the"
        " project test-project.",
    )
    mock_aiplatform.init.assert_called_once_with(
        project="test-project", location="us-central1"
    )
    mock_aiplatform.start_upload_tb_log.assert_not_called()

  @absltest.mock.patch(
      "cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader.uploader.tensorboard"
  )
  @absltest.mock.patch(
      "cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader.uploader.aiplatform"
  )
  def testWhenNoExperimentExistsThenVertexUploaderNotCalled(
      self,
      mock_aiplatform,
      mock_tensorboard,
  ):
    # given
    mock_tensorboard.get_instance_identifiers.return_value = ["test_experiment"]
    mock_tensorboard.get_experiment.return_value = None

    # when
    with self.assertLogs(level="ERROR") as log:
      uploader.start_upload_to_tensorboard(
          "test-project",
          "us-central1",
          "test-experiment",
          "test-instance",
          "logdir",
      )

    # then
    self.assertRegex(
        log.output[0],
        "No Tensorboard experiment with the name test-experiment present in"
        " the project test-project.",
    )
    mock_aiplatform.init.assert_called_once_with(
        project="test-project", location="us-central1"
    )
    mock_aiplatform.start_upload_tb_log.assert_not_called()

  @absltest.mock.patch(
      "cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader.uploader.aiplatform"
  )
  def testWhenStopUploadToTensorboardIsCalledThenVertexUploadIsStopped(
      self,
      mock_aiplatform,
  ):
    # when
    uploader.stop_upload_to_tensorboard()

    # then
    mock_aiplatform.end_upload_tb_log.assert_called_once()


if __name__ == "__main__":
  absltest.main()
