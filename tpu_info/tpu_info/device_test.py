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

import contextlib
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tpu_info import device


class TestDevice(parameterized.TestCase):

  @parameterized.named_parameters(
      [
          (
              "v2-8",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x0027",
              "0x004e",
              device.TpuChip.V2,
              4,
          ),
          (
              "v3-8",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x0027",
              "0x004f",
              device.TpuChip.V3,
              4,
          ),
          (
              "v4-8",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x005e",
              "0x0000",
              device.TpuChip.V4,
              4,
          ),
          (
              "v5litepod-4",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x0063",
              "0x0000",
              device.TpuChip.V5E,
              4,
          ),
          (
              "v5litepod-8",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x0063",
              "0x0000",
              device.TpuChip.V5E,
              8,
          ),
          (
              "v5p-8",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x0062",
              "0x0000",
              device.TpuChip.V5P,
              4,
          ),
          ("not_google", "0x1234", "0x0000", "0x0000", None, 1),
          (
              "invalid_device_id",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x9999",
              "0x0000",
              None,
              1,
          ),
      ],
  )
  def test_get_local_chips(
      self, vendor_id, device_id, subsystem_id, expected_chip, count
  ):
    # Create fake sysfs paths for this device
    tmp_path = pathlib.Path(self.create_tempdir())
    device_paths = []
    for i in range(count):
      tempdir = tmp_path / f"device_{i}"
      tempdir.mkdir()
      (tempdir / "vendor").write_text(vendor_id)
      (tempdir / "device").write_text(device_id)
      (tempdir / "subsystem_device").write_text(subsystem_id)
      device_paths.append(str(tempdir))

    mock_paths = {"/sys/bus/pci/devices/*": device_paths}
    with mock.patch("glob.glob", mock_paths.get):
      res = device.get_local_chips()
      if expected_chip:
        self.assertEqual(res, (expected_chip, count))
      else:
        self.assertEqual(res, (None, 0))

  @parameterized.parameters(
      [
          (device.TpuChip.V2, 0, "/dev/accel0"),
          (device.TpuChip.V3, 0, "/dev/accel0"),
          (device.TpuChip.V4, 1, "/dev/accel1"),
          (device.TpuChip.V5P, 0, "/dev/vfio/0"),
          (device.TpuChip.V5E, 3, "/dev/vfio/3"),
      ],
  )
  def test_device_path(self, chip_type, index, expected):
    self.assertEqual(device.chip_path(chip_type, index), expected)

  @staticmethod
  @contextlib.contextmanager
  def _mock_proc_paths(mock_links):
    """Mocks /proc paths for open files."""
    mock_paths = {"/proc/*/fd/*": mock_links.keys()}
    with mock.patch(
        "os.readlink", lambda link: mock_links.get(link, "")
    ), mock.patch("glob.glob", mock_paths.get):
      yield

  @parameterized.named_parameters(
      [
          (
              "old_driver",
              {
                  "/proc/123/fd/4": "/dev/accel0",
                  "/proc/124/fd/5": "/dev/accel1",
                  "/proc/125/fd/6": "/dev/accel2",
                  "/proc/126/fd/7": "/dev/accel3",
              },
              {
                  "/dev/accel0": 123,
                  "/dev/accel1": 124,
                  "/dev/accel2": 125,
                  "/dev/accel3": 126,
              },
          ),
          (
              "new_driver",
              {
                  "/proc/123/fd/8": "/dev/vfio/0",
                  "/proc/124/fd/9": "/dev/vfio/1",
                  "/proc/125/fd/10": "/dev/vfio/2",
                  "/proc/126/fd/11": "/dev/vfio/3",
              },
              {
                  "/dev/vfio/0": 123,
                  "/dev/vfio/1": 124,
                  "/dev/vfio/2": 125,
                  "/dev/vfio/3": 126,
              },
          ),
          (
              "one_link",
              {"/proc/456/fd/7": "/dev/accel2"},
              {"/dev/accel2": 456},
          ),
          (
              "other_device",
              {"/proc/789/fd/9": "/dev/some_other_device"},
              {},
          ),
          (
              "no_links",
              {},
              {},
          ),
          (
              "invalid_device",
              {"/proc/111/fd/1": "/dev/vfio/"},
              {},
          ),
      ],
  )
  def test_get_chip_owners(self, mock_links, expected_owners):
    with self._mock_proc_paths(mock_links):
      result = device.get_chip_owners()
      self.assertEqual(result, expected_owners)

  @parameterized.named_parameters(
      [
          ("missing_fd_number", {"/proc/222/fd/": "/dev/accel3"}),
          ("invalid_fd_number", {"/proc/222/fd/abc": "/dev/accel3"}),
      ],
  )
  def test_get_chip_owners_error(self, mock_links):
    """Test expected errors with invalid /proc paths."""
    with self.assertRaises(RuntimeError), self._mock_proc_paths(mock_links):
      device.get_chip_owners()


if __name__ == "__main__":
  absltest.main()
