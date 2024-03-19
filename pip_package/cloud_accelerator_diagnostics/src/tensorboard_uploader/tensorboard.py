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

"""Tensorboard module.

This module provides the functionality to create Tensorboard instance and
Experiment in Vertex AI.
"""

import logging

from google.cloud.aiplatform import aiplatform


logger = logging.getLogger(__name__)

# The API URI for accessing the Tensorboard UI
WEB_SERVER_URI = "tensorboard.googleusercontent.com"


def create_instance(project, location, tensorboard_name):
  """Creates a new Tensorboard instance in Vertex AI.

  Args:
      project (str): Google Cloud Project to create the Tensorboard instance to.
      location (str): Location to create the Tensorboard instance to. See
        https://cloud.google.com/vertex-ai/docs/general/locations#available-regions
          for the list of available VertexAI locations.
      tensorboard_name (str): The user-defined name of the Tensorboard. The name
        can be up to 128 characters long and can be consist of any UTF-8
        characters.

  Returns:
    str: The Tensorboard instance identifier.
  """
  try:
    aiplatform.init(project=project, location=location)
    tensorboard_identifiers = get_instance_identifiers(tensorboard_name)
    if not tensorboard_identifiers:
      # create a new Tensorboard instance if an instance doesn't exist
      logger.info(
          "Creating a Tensorboard instance with the name: %s", tensorboard_name
      )
      tensorboard = aiplatform.Tensorboard.create(
          display_name=tensorboard_name,
          project=project,
          location=location,
      )
      return tensorboard.name
    else:
      logger.info(
          "Tensorboard instance with the name: %s already exist in project: %s"
          " and location: %s. Not creating a new Tensorboard instance.",
          tensorboard_name,
          project,
          location,
      )
      # return the first Tensorboard instance even if multiple instances exist
      return tensorboard_identifiers[0]
  except (ValueError, Exception):
    logger.exception("Error while creating Tensorboard instance.")
    return None


def create_experiment(project, location, experiment_name, tensorboard_name):
  """Creates a new Tensorboard Experiment in VertexAI.

  Args:
    project (str): Google Cloud Project to create the Tensorboard experiment to.
    location (str): Location to create the Tensorboard experiment to. See
      https://cloud.google.com/vertex-ai/docs/general/locations#available-regions
        for the list of available VertexAI locations.
    experiment_name (str): The name of the Tensorboard experiment to create.
      This value should be 1-128 characters, and valid characters are
      /[a-z][0-9]-/.
    tensorboard_name (str): The name of the Tensorboard to create the
      Tensorboard Experiment in.

  Returns:
    str: The Tensorboard instance identifier.
    str: The URL to access the Tensorboard UI.
  """
  try:
    aiplatform.init(project=project, location=location)

    # Get the identifier for the Tensorboard instance. If no Tensorboard
    # instance is present, then create a new instance.
    tensorboard_identifiers = get_instance_identifiers(tensorboard_name)
    if not tensorboard_identifiers:
      logger.info(
          "No Tensorboard instance present in the project: %s. Creating"
          " a new Tensorboard instance with the name: %s",
          project,
          tensorboard_name,
      )
      tensorboard_id = create_instance(project, location, tensorboard_name)
      # create_instance() failed to create a Tensorboard instance
      if tensorboard_id is None:
        return None, None
    else:
      # get the first Tensorboard instance even if multiple instances exist
      tensorboard_id = tensorboard_identifiers[0]

    # check if an experiment already exist for the tensorboard_id
    experiment = get_experiment(tensorboard_id, experiment_name)
    if experiment is not None:
      logger.info(
          "Experiment with the name: %s already exist in the project: %s."
          " Not creating a new Experiment.",
          experiment_name,
          project,
      )
    else:
      logger.info(
          "Creating Experiment for Tensorboard instance id: %s", tensorboard_id
      )
      experiment = aiplatform.TensorboardExperiment.create(
          tensorboard_experiment_id=experiment_name,
          display_name=experiment_name,
          tensorboard_name=tensorboard_id,
      )
    experiment_resource_name = experiment.resource_name
    tensorboard_url = "https://{}.{}/experiment/{}".format(
        location,
        WEB_SERVER_URI,
        experiment_resource_name.replace("/", "+"),
    )
    return tensorboard_id, tensorboard_url
  except (ValueError, Exception):
    logger.exception("Error while creating Tensorboard Experiment.")
    return None, None


def get_instance_identifiers(tensorboard_name):
  """Retrieves a list of Tensorboard instance identifiers that match the given `tensorboard_name`.

  Args:
    tensorboard_name (str): The name of the Tensorboard instance to search for.

  Returns:
    list: A list of Tensorboard instance identifiers that match
    `tensorboard_name`.
  """
  tensorboard_instances = aiplatform.tensorboard.Tensorboard.list()
  tensorboard_identifiers = []
  for tensorboard in tensorboard_instances:
    if tensorboard.display_name == tensorboard_name:
      tensorboard_identifiers.append(tensorboard.name)
  return tensorboard_identifiers


def get_experiment(tensorboard_id, experiment_name):
  """Retrieves the experiment object if an experiment with the given `experiment_name` exists for the given `tensorboard_id`.

  Args:
    tensorboard_id (str): The id of Tensorboard instance.
    experiment_name (str): The name of Tensorboard experiment.

  Returns:
    TensorboardExperiment object if an experiment with the given name exist
    in the project, None otherwise.
  """
  experiment_list = aiplatform.tensorboard.TensorboardExperiment.list(
      tensorboard_id
  )
  for experiment in experiment_list:
    if experiment.display_name == experiment_name:
      return experiment
  return None
