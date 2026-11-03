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
import enum
import os
import pathlib
import sys
import tempfile
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
          (
              "v6e-4",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x006f",
              "0x0000",
              device.TpuChip.V6E,
              4,
          ),
          (
              "v7x-8",
              device.GOOGLE_PCI_VENDOR_ID,
              "0x0076",
              "0x0000",
              device.TpuChip.V7X,
              8,
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
    # Create fake sysfs paths for this device. Use tempfile rather than
    # self.create_tempdir() so the test also runs under pytest, which imports
    # the module without parsing absl flags (--test_tmpdir).
    tmp_path = pathlib.Path(
        self.enter_context(tempfile.TemporaryDirectory())
    )
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

    def readlink_side_effect(link: str):
      value_or_exception = mock_links[link]
      if isinstance(value_or_exception, Exception):
        raise value_or_exception
      return value_or_exception

    mock_paths = {"/proc/*/fd/*": mock_links.keys()}

    with mock.patch(
        "os.readlink", side_effect=readlink_side_effect
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
  def test_get_chip_owners_runtime_error(self, mock_links):
    """Test expected errors with invalid /proc paths."""
    with self.assertRaises(RuntimeError), self._mock_proc_paths(mock_links):
      device.get_chip_owners()

  @parameterized.named_parameters(
      [
          (
              "missing_file",
              {"/proc/9999/fd/": FileNotFoundError("Mock File not found")},
          ),
      ],
  )
  def test_get_chip_owners_file_not_found_error(self, mock_path):
    """Tests that FileNotFoundError from os.readlink is handled correctly."""
    with self._mock_proc_paths(mock_path):
      with mock.patch("re.fullmatch") as mock_fullmatch:
        device.get_chip_owners()

    mock_fullmatch.assert_not_called()

  @parameterized.named_parameters([
      (
          "with_core_0",
          device.ChipInfo(
              "0000:01:00",
              cores={
                  0: device.CoreInfo(0, "0x0", "/dev/vfio/10", "00:00.0"),
                  1: device.CoreInfo(1, "0x0", "/dev/vfio/1", "00:00.1"),
              },
          ),
          ["/dev/vfio/", 10, ""],
      ),
      (
          "without_core_0",
          device.ChipInfo(
              "0000:01:00",
              cores={
                  1: device.CoreInfo(1, "0x0", "/dev/vfio/1", "00:00.1"),
              },
          ),
          ["/dev/vfio/", 1, ""],
      ),
      (
          "no_cores",
          device.ChipInfo("0000:01:00", cores={}),
          [""],
      ),
      (
          "vfio_path_digit_2",
          device.ChipInfo(
              "0000:01:00",
              cores={
                  0: device.CoreInfo(0, "0x0", "/dev/vfio/2", "00:00.0"),
              },
          ),
          ["/dev/vfio/", 2, ""],
      ),
      (
          "vfio_path_digit_11",
          device.ChipInfo(
              "0000:01:00",
              cores={
                  0: device.CoreInfo(0, "0x0", "/dev/vfio/11", "00:00.0"),
              },
          ),
          ["/dev/vfio/", 11, ""],
      ),
  ])
  def test_chip_sort_key(self, chip, expected):
    self.assertEqual(device._chip_sort_key(chip), expected)


  def test_nonmember_decorator(self):
    """Tests that the nonmember decorator is used correctly."""
    expected = (
        enum.nonmember if sys.version_info >= (3, 11) else device._identity
    )
    self.assertEqual(expected, device.nonmember)

    # Before 3.11 there is no enum.nonmember, so Info unavoidably becomes a
    # member; only assert it is kept out of the members where that is possible.
    if sys.version_info >= (3, 11):
      self.assertNotIn("Info", device.TpuChip.__members__)

  def test_identity_function(self):
    """Tests that the identity function is used correctly."""
    obj = object()
    self.assertIs(device._identity(obj), obj)


class GetChipsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_glob = self.enter_context(mock.patch("glob.glob", autospec=True))
    self.mock_islink = self.enter_context(
        mock.patch("os.path.islink", autospec=True)
    )
    self.mock_readlink = self.enter_context(
        mock.patch("os.readlink", autospec=True)
    )

    def _open_side_effect(path, mode):
      if os.path.basename(path) == "vendor":
        if "non-google" in path:
          content = "0xNONGOOGLE"
        else:
          content = device.GOOGLE_PCI_VENDOR_ID
      elif os.path.basename(path) == "device":
        if "0000:01:00" in path:
          content = "0x0027"
        elif "0000:02:00" in path:
          content = "0x005e"
        else:
          content = "0xUNKNOWN"
      else:
        raise FileNotFoundError(path)
      return mock.mock_open(read_data=content)()

    self.mock_open = self.enter_context(mock.patch("builtins.open"))
    self.mock_open.side_effect = _open_side_effect

  def test_get_chips_no_devices(self):
    self.mock_glob.return_value = []
    chips = device.get_chips()
    self.assertEqual(chips, [])

  def test_get_chips_sorting_vfio_paths_numerical_no_core0(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:01:00.1",
        "/sys/bus/pci/devices/0000:02:00.1",
    ]
    self.mock_islink.return_value = True
    self.mock_readlink.side_effect = [
        "../../../../kernel/iommu_groups/10",  # for 0000:01:00.1
        "../../../../kernel/iommu_groups/9",  # for 0000:02:00.1
    ]

    chips = device.get_chips()
    self.assertLen(chips, 2)
    self.assertEqual(chips[0].base_addr, "0000:02:00")
    self.assertEqual(chips[0].cores[1].vfio_path, "/dev/vfio/9")
    self.assertEqual(chips[1].base_addr, "0000:01:00")
    self.assertEqual(chips[1].cores[1].vfio_path, "/dev/vfio/10")

  def test_get_chips_sort_without_core0(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:02:00.0",
        "/sys/bus/pci/devices/0000:01:00.1",
    ]
    self.mock_islink.return_value = True
    self.mock_readlink.side_effect = [
        "../../../../kernel/iommu_groups/12",  # for 0000:02:00.0
        "../../../../kernel/iommu_groups/10",  # for 0000:01:00.1
    ]

    chips = device.get_chips()

    self.assertLen(chips, 2)
    # 0000:01:00.1 with iommu 10 should come first.
    self.assertEqual(chips[0].base_addr, "0000:01:00")
    self.assertNotIn(0, chips[0].cores)
    self.assertIn(1, chips[0].cores)
    self.assertEqual(chips[0].cores[1].vfio_path, "/dev/vfio/10")
    self.assertEqual(chips[1].base_addr, "0000:02:00")
    self.assertEqual(chips[1].cores[0].vfio_path, "/dev/vfio/12")

  def test_get_chips_ignores_device_with_vendor_file_ioerror(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:01:00.0",  # Good device
        "/sys/bus/pci/devices/0000:0BAD:00.0",  # Bad device
    ]

    def _open_side_effect_with_ioerror(path, mode):
      if "0BAD" in path and os.path.basename(path) == "vendor":
        raise IOError("Cannot read vendor file")
      elif os.path.basename(path) == "vendor":
        return mock.mock_open(read_data=device.GOOGLE_PCI_VENDOR_ID)()
      elif os.path.basename(path) == "device":
        return mock.mock_open(read_data="0x0027")()
      raise FileNotFoundError(path)

    self.mock_open.side_effect = _open_side_effect_with_ioerror
    self.mock_islink.return_value = True
    self.mock_readlink.return_value = "../../../../kernel/iommu_groups/10"

    chips = device.get_chips()
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].base_addr, "0000:01:00")

  def test_get_chips_ignores_device_with_readlink_ioerror(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:01:00.0",  # Good device
        "/sys/bus/pci/devices/0000:02:00.0",  # Bad device, readlink fails
    ]
    self.mock_islink.return_value = True

    def readlink_side_effect_with_error(path):
      if "0000:02:00.0" in path:
        raise IOError("readlink failed")
      return "../../../../kernel/iommu_groups/10"

    self.mock_readlink.side_effect = readlink_side_effect_with_error

    chips = device.get_chips()
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].base_addr, "0000:01:00")

  def test_get_chips_ignores_device_with_device_file_ioerror(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:01:00.0",  # Good device
        "/sys/bus/pci/devices/0000:0BAD:00.0",  # Bad device
    ]

    def _open_side_effect_with_ioerror(path, mode):
      if "0BAD" in path and os.path.basename(path) == "device":
        raise IOError("Cannot read device file")
      elif os.path.basename(path) == "vendor":
        return mock.mock_open(read_data=device.GOOGLE_PCI_VENDOR_ID)()
      elif "0000:01:00" in path and os.path.basename(path) == "device":
        return mock.mock_open(read_data="0x0027")()
      raise FileNotFoundError(path)

    self.mock_open.side_effect = _open_side_effect_with_ioerror
    self.mock_islink.return_value = True
    self.mock_readlink.return_value = "../../../../kernel/iommu_groups/10"

    chips = device.get_chips()
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].base_addr, "0000:01:00")

  def test_get_chips_one_chip_one_core(self):
    self.mock_glob.return_value = ["/sys/bus/pci/devices/0000:01:00.0"]
    self.mock_islink.return_value = True
    self.mock_readlink.return_value = "../../../../kernel/iommu_groups/10"

    chips = device.get_chips()

    self.assertLen(chips, 1)
    chip = chips[0]
    self.assertEqual(chip.base_addr, "0000:01:00")
    self.assertIn(0, chip.cores)
    core = chip.cores[0]
    self.assertEqual(core.core_index, 0)
    self.assertEqual(core.device_id, "0x0027")
    self.assertEqual(core.vfio_path, "/dev/vfio/10")
    self.assertEqual(core.full_addr, "0000:01:00.0")

  def test_get_chips_one_chip_two_cores(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:01:00.0",
        "/sys/bus/pci/devices/0000:01:00.1",
    ]
    self.mock_islink.return_value = True
    self.mock_readlink.side_effect = [
        "../../../../kernel/iommu_groups/10",
        "../../../../kernel/iommu_groups/11",
    ]

    chips = device.get_chips()

    self.assertLen(chips, 1)
    chip = chips[0]
    self.assertEqual(chip.base_addr, "0000:01:00")
    self.assertLen(chip.cores, 2)
    self.assertIn(0, chip.cores)
    self.assertIn(1, chip.cores)
    core0 = chip.cores[0]
    core1 = chip.cores[1]
    self.assertEqual(core0.core_index, 0)
    self.assertEqual(core0.device_id, "0x0027")
    self.assertEqual(core0.vfio_path, "/dev/vfio/10")
    self.assertEqual(core0.full_addr, "0000:01:00.0")
    self.assertEqual(core1.core_index, 1)
    self.assertEqual(core1.device_id, "0x0027")
    self.assertEqual(core1.vfio_path, "/dev/vfio/11")
    self.assertEqual(core1.full_addr, "0000:01:00.1")

  def test_get_chips_multiple_chips_sorted(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:02:00.0",
        "/sys/bus/pci/devices/0000:01:00.0",
    ]
    self.mock_islink.return_value = True
    self.mock_readlink.side_effect = [
        "../../../../kernel/iommu_groups/12",  # for 0000:02:00.0
        "../../../../kernel/iommu_groups/10",  # for 0000:01:00.0
    ]

    chips = device.get_chips()

    self.assertLen(chips, 2)
    # They should be sorted by vfio path ('.../10' < '.../12'),
    # so 0000:01:00.0 (iommu 10) comes first
    self.assertEqual(chips[0].base_addr, "0000:01:00")
    self.assertEqual(chips[0].cores[0].vfio_path, "/dev/vfio/10")
    self.assertEqual(chips[1].base_addr, "0000:02:00")
    self.assertEqual(chips[1].cores[0].vfio_path, "/dev/vfio/12")

  def test_get_chips_non_google_device_ignored(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/non-google/0000:03:00.0",
        "/sys/bus/pci/devices/0000:02:00.0",
    ]
    self.mock_islink.return_value = True
    self.mock_readlink.return_value = "../../../../kernel/iommu_groups/10"

    chips = device.get_chips()
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].base_addr, "0000:02:00")

  def test_get_chips_no_iommu_group_ignored(self):
    self.mock_glob.return_value = ["/sys/bus/pci/devices/0000:01:00.0"]
    self.mock_islink.return_value = False  # iommu_group is not a symlink

    chips = device.get_chips()
    self.assertEqual(chips, [])

  def test_get_chips_chip_with_no_cores_is_filtered_out(self):
    self.mock_glob.return_value = [
        "/sys/bus/pci/devices/0000:01:00.0",  # This device fails iommu check
        "/sys/bus/pci/devices/0000:02:00.0",  # This device is valid
    ]

    def islink_side_effect(path):
      if "0000:01:00.0" in path:
        return False  # device for chip 0000:01:00 has no iommu group
      return True

    self.mock_islink.side_effect = islink_side_effect
    self.mock_readlink.return_value = "../../../../kernel/iommu_groups/10"

    chips = device.get_chips()
    self.assertLen(chips, 1)
    self.assertEqual(chips[0].base_addr, "0000:02:00")
    self.assertIn(0, chips[0].cores)

  def test_get_chips_large_range_sorting(self):
    device_iommu_map = {
        f"/sys/bus/pci/devices/0000:{i:02x}:00.0": f"{i}" for i in range(12)
    }
    shuffled_devices = list(device_iommu_map.keys())
    # Shuffle devices to ensure sorting works
    shuffled_devices.reverse()
    self.mock_glob.return_value = shuffled_devices
    self.mock_islink.return_value = True

    def readlink_sort_test_side_effect(path):
      dev_path = os.path.dirname(path)
      if dev_path in device_iommu_map:
        return "../../../../kernel/iommu_groups/" + device_iommu_map[dev_path]
      raise FileNotFoundError(path)

    self.mock_readlink.side_effect = readlink_sort_test_side_effect

    def open_sort_test_side_effect(path, mode):
      if os.path.basename(path) == "vendor":
        return mock.mock_open(read_data=device.GOOGLE_PCI_VENDOR_ID)()
      elif os.path.basename(path) == "device":
        return mock.mock_open(read_data="0x005e")()
      raise FileNotFoundError(path)

    self.mock_open.side_effect = open_sort_test_side_effect

    chips = device.get_chips()
    self.assertLen(chips, 12)
    for i in range(12):
      self.assertEqual(chips[i].cores[0].vfio_path, f"/dev/vfio/{i}")


if __name__ == "__main__":
  absltest.main()
