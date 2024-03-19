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

from absl.testing import absltest
from cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader import tensorboard


class TensorboardTest(absltest.TestCase):

  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard.create")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  def testCreateInstanceWhenNoInstanceExist(
      self, mock_tensorboard_list, mock_tensorboard_create
  ):
    mock_tensorboard_list.return_value = []
    mock_tensorboard_create.return_value.name = "123"

    instance_id = tensorboard.create_instance(
        "test-project", "us-central1", "test-instance"
    )

    mock_tensorboard_list.assert_called_once()
    mock_tensorboard_create.assert_called_once_with(
        project="test-project",
        location="us-central1",
        display_name="test-instance",
    )
    self.assertEqual(instance_id, "123")

  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard")
  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard.create")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  def testCreateInstanceWhenSameNameInstanceExist(
      self, mock_tensorboard_list, mock_tensorboard_create, mock_tensorboard
  ):
    mock_tensorboard_instance = mock_tensorboard.return_value
    mock_tensorboard_instance.display_name = "test-instance"
    mock_tensorboard_list.return_value = [mock_tensorboard_instance]

    instance_id = tensorboard.create_instance(
        "test-project", "us-central1", "test-instance"
    )

    mock_tensorboard_list.assert_called_once()
    mock_tensorboard_create.assert_not_called()
    self.assertEqual(instance_id, mock_tensorboard_instance.name)

  def testCreateInstanceForUnsupportedRegion(self):
    with self.assertLogs(level="ERROR") as log:
      instance_id = tensorboard.create_instance(
          "test-project", "us-central2", "test-instance"
      )

    self.assertRegex(
        log.output[0], "ValueError: Unsupported region for Vertex AI"
    )
    self.assertIsNone(instance_id)

  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard.create")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  def testCreateInstanceWhenExceptionIsThrown(
      self, mock_tensorboard_list, mock_tensorboard_create
  ):
    mock_tensorboard_list.return_value = []
    mock_tensorboard_create.return_value = Exception("Exception is thrown...")

    with self.assertLogs(level="ERROR"):
      instance_id = tensorboard.create_instance(
          "test-project", "us-central1", "test-instance"
      )

    mock_tensorboard_list.assert_called_once()
    mock_tensorboard_create.assert_called_once_with(
        project="test-project",
        location="us-central1",
        display_name="test-instance",
    )
    self.assertIsNone(instance_id)

  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.TensorboardExperiment.list"
  )
  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.TensorboardExperiment.create"
  )
  def testCreateExperimentWhenTensorboardInstanceExist(
      self,
      mock_experiment_create,
      mock_tensorboard_list,
      mock_tensorboard,
      mock_experiment_list,
  ):
    mock_tensorboard_instance = mock_tensorboard.return_value
    mock_tensorboard_instance.display_name = "test-instance"
    mock_tensorboard_instance.name = "123"
    mock_tensorboard_list.return_value = [mock_tensorboard_instance]
    mock_experiment_list.return_value = []
    expected_resource_name = "projects/770040921623/locations/us-central1/tensorboards/123/experiments/test-experiment"
    mock_experiment_create.return_value.resource_name = expected_resource_name
    expected_tensorboard_url = (
        "https://us-central1.tensorboard.googleusercontent.com/experiment/"
        + expected_resource_name.replace("/", "+")
    )

    instance_id, tensorboard_url = tensorboard.create_experiment(
        "test-project", "us-central1", "test-experiment", "test-instance"
    )

    mock_tensorboard_list.assert_called_once()
    mock_experiment_create.assert_called_once_with(
        tensorboard_experiment_id="test-experiment",
        tensorboard_name="123",
        display_name="test-experiment",
    )
    self.assertEqual(instance_id, "123")
    self.assertEqual(tensorboard_url, expected_tensorboard_url)

  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.TensorboardExperiment.list"
  )
  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard.create")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.TensorboardExperiment.create"
  )
  def testCreateExperimentWhenNoTensorboardInstanceExist(
      self,
      mock_experiment_create,
      mock_tensorboard_list,
      mock_tensorboard_create,
      mock_experiment_list,
  ):
    mock_tensorboard_list.return_value = []
    mock_tensorboard_create.return_value.name = "123"
    mock_experiment_list.return_value = []
    expected_resource_name = "projects/770040921623/locations/us-central1/tensorboards/123/experiments/test-experiment"
    mock_experiment_create.return_value.resource_name = expected_resource_name
    expected_tensorboard_url = (
        "https://us-central1.tensorboard.googleusercontent.com/experiment/"
        + expected_resource_name.replace("/", "+")
    )

    instance_id, tensorboard_url = tensorboard.create_experiment(
        "test-project", "us-central1", "test-experiment", "test-instance"
    )

    mock_tensorboard_list.assert_called()
    mock_tensorboard_create.assert_called_once_with(
        project="test-project",
        location="us-central1",
        display_name="test-instance",
    )
    mock_experiment_create.assert_called_once_with(
        tensorboard_experiment_id="test-experiment",
        tensorboard_name="123",
        display_name="test-experiment",
    )
    self.assertEqual(instance_id, "123")
    self.assertEqual(tensorboard_url, expected_tensorboard_url)

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
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.TensorboardExperiment.create"
  )
  def testCreateExperimentWhenTensorboardInstanceAndExperimentExist(
      self,
      mock_experiment_create,
      mock_tensorboard_list,
      mock_tensorboard,
      mock_experiment_list,
      mock_experiment,
  ):
    mock_tensorboard_instance = mock_tensorboard.return_value
    mock_tensorboard_instance.display_name = "test-instance"
    mock_tensorboard_instance.name = "123"
    mock_tensorboard_list.return_value = [mock_tensorboard_instance]
    expected_resource_name = "projects/770040921623/locations/us-central1/tensorboards/123/experiments/test-experiment"
    expected_tensorboard_url = (
        "https://us-central1.tensorboard.googleusercontent.com/experiment/"
        + expected_resource_name.replace("/", "+")
    )
    mock_experiment_instance = mock_experiment.return_value
    mock_experiment_instance.display_name = "test-experiment"
    mock_experiment_instance.resource_name = expected_resource_name
    mock_experiment_list.return_value = [mock_experiment_instance]

    instance_id, tensorboard_url = tensorboard.create_experiment(
        "test-project", "us-central1", "test-experiment", "test-instance"
    )

    mock_tensorboard_list.assert_called_once()
    mock_experiment_create.assert_not_called()
    self.assertEqual(instance_id, "123")
    self.assertEqual(tensorboard_url, expected_tensorboard_url)

  def testCreateExperimentForUnsupportedRegion(self):
    with self.assertLogs(level="ERROR") as log:
      instance_id, tensorboard_url = tensorboard.create_experiment(
          "test-project", "us-central2", "test-experiment", "test-instance"
      )

    self.assertRegex(
        log.output[0], "ValueError: Unsupported region for Vertex AI"
    )
    self.assertIsNone(instance_id)
    self.assertIsNone(tensorboard_url)

  @absltest.mock.patch("google.cloud.aiplatform.aiplatform.Tensorboard.create")
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.tensorboard.Tensorboard.list"
  )
  @absltest.mock.patch(
      "google.cloud.aiplatform.aiplatform.TensorboardExperiment.create"
  )
  def testCreateExperimentWhenCreateInstanceFails(
      self,
      mock_experiment_create,
      mock_tensorboard_list,
      mock_tensorboard_create,
  ):
    mock_tensorboard_list.return_value = []
    mock_tensorboard_create.return_value = Exception("Exception is thrown...")

    with self.assertLogs(level="ERROR") as log:
      instance_id, tensorboard_url = tensorboard.create_experiment(
          "test-project", "us-central1", "test-experiment", "test-instance"
      )

    mock_tensorboard_list.assert_called()
    mock_tensorboard_create.assert_called_once_with(
        project="test-project",
        location="us-central1",
        display_name="test-instance",
    )
    mock_experiment_create.assert_not_called()
    self.assertRegex(
        log.output[0], "Error while creating Tensorboard instance."
    )
    self.assertIsNone(instance_id)
    self.assertIsNone(tensorboard_url)


if __name__ == "__main__":
  absltest.main()
