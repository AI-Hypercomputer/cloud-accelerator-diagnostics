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

"""Utilities for detecting locally-attached TPU devices."""

import collections
import dataclasses
import enum
import glob
import os
import pathlib
import re
import typing
from typing import Dict, Literal, Optional, Tuple

GOOGLE_PCI_VENDOR_ID = "0x1ae0"


class TpuChip(enum.Enum):
  """TPU chip versions and basic specs."""

  class Info(typing.NamedTuple):
    """Specs for a specific TPU chip version."""

    name: str
    hbm_gib: int
    devices_per_chip: Literal[1, 2]

  V2 = Info("v2", hbm_gib=8, devices_per_chip=2)
  V3 = Info("v3", hbm_gib=16, devices_per_chip=2)
  V4 = Info("v4", hbm_gib=32, devices_per_chip=1)
  V5E = Info("v5e", hbm_gib=16, devices_per_chip=1)
  V5P = Info("v5p", hbm_gib=95, devices_per_chip=1)
  V6E = Info("v6e", hbm_gib=32, devices_per_chip=1)
  V7X = Info("7x", hbm_gib=192, devices_per_chip=2)

  @classmethod
  def from_pci_device_id(
      cls, device_id: str, subsystem_id: str
  ) -> Optional["TpuChip"]:
    """Returns TPU chip type for given PCI IDs, or None if not a TPU device."""
    # TPU v2 and v3 share a device ID
    if device_id == "0x0027":
      if subsystem_id == "0x004e":
        return cls.V2
      elif subsystem_id == "0x004f":
        return cls.V3

    device_id_to_device = {
        "0x005e": cls.V4,
        "0x0063": cls.V5E,
        "0x0062": cls.V5P,
        "0x006f": cls.V6E,
        "0x0076": cls.V7X,
    }

    return device_id_to_device.get(device_id)

  def __str__(self):
    """Human-readable name of TPU chip type."""
    # If it is v7x, string is TPU7x (not TPU v7x)
    if self.value.name == "7x":
      return "TPU7x chip"
    else:
      return f"TPU {self.value.name} chip"


@dataclasses.dataclass
class CoreInfo:
  """Information about a TPU core."""
  core_index: int
  device_id: str
  vfio_path: str
  full_addr: str


@dataclasses.dataclass
class ChipInfo:
  """Information about a TPU chip."""
  base_addr: str
  # Core index to CoreInfo mapping.
  cores: dict[int, CoreInfo] = dataclasses.field(default_factory=dict)


def get_actual_chips() -> list[ChipInfo]:
  """Returns the list of information about chips with core information."""
  pci_devices_path = "/sys/bus/pci/devices/*"
  # Organize chips by base address & associate cores/devices with chips.
  chips_by_base_addr: dict[str, ChipInfo] = {}
  for device_path in glob.glob(pci_devices_path):
    try:
      pci_addr = os.path.basename(device_path)
      # Check vendor ID
      vendor_path = os.path.join(device_path, "vendor")
      with open(vendor_path, "r") as f:
        vendor_id = f.read().strip()
      if vendor_id != GOOGLE_PCI_VENDOR_ID:
        continue
    except IOError:
      continue
    # Check for IOMMU group
    iommu_group_path = os.path.join(device_path, "iommu_group")
    if not os.path.islink(iommu_group_path):
      continue
    try:
      # Read the symlink to get the IOMMU group number
      iommu_link = os.readlink(iommu_group_path)
      iommu_group = os.path.basename(iommu_link)

      # Read device ID using the path (should be just device ID)
      with open(os.path.join(device_path, "device"), "r") as f:
        device_id = f.read().strip()

      # Group by base PCI address (Domain:Bus:Device)
      pci_base = pci_addr.split(".")[0]
      core_index = int(pci_addr.split(".")[-1])

      # Either grab the existing ChipInfo object or create a new one.
      chip_info = chips_by_base_addr.get(
          pci_base,
          ChipInfo(base_addr=pci_base, cores=dict())
      )
      # Overwrite in the unusual case of a core with the same index already
      # existing on the chip (shouldn't happen).
      core_info = CoreInfo(
          core_index=core_index,
          device_id=device_id,
          vfio_path=f"/dev/vfio/{iommu_group}",
          full_addr=pci_addr,
      )
      chip_info.cores[core_index] = core_info

      # Make sure we have the new/updated ChipInfo object with core information.
      chips_by_base_addr[pci_base] = chip_info
    except IOError:
      continue

  # Sort the chips by the chip's zero-indexed core's vfio path.
  def chip_sort_key(chip: ChipInfo) -> str:
    # lambda chip: chip.cores[0].vfio_path
    if 0 in chip.cores:
      return chip.cores[0].vfio_path
    elif not chip.cores:
      return ""
    else:
      return next(iter(chip.cores.values())).vfio_path

  chips = sorted(
      chips_by_base_addr.values(),
      key=chip_sort_key,
  )
  return chips


def get_local_chips() -> Tuple[Optional[TpuChip], int]:
  """Returns the type and number of TPU chips available."""
  count = collections.Counter()
  for pci_path in glob.glob("/sys/bus/pci/devices/*"):
    vendor_path = os.path.join(pci_path, "vendor")
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != GOOGLE_PCI_VENDOR_ID:
      continue

    device_id_path = os.path.join(pci_path, "device")
    device_id = pathlib.Path(device_id_path).read_text().strip()
    subsystem_path = os.path.join(pci_path, "subsystem_device")
    subsystem_id = pathlib.Path(subsystem_path).read_text().strip()

    chip_type = TpuChip.from_pci_device_id(device_id, subsystem_id)
    if chip_type:
      count[chip_type] += 1

  assert len(count) <= 1, f"Expected one chip type, got {count}"
  return count.most_common()[0] if count else (None, 0)


def chip_path(chip_type: TpuChip, index: int):
  """Returns the expected `/dev` path for a given TPU device type."""
  if chip_type in {TpuChip.V5E, TpuChip.V5P, TpuChip.V6E, TpuChip.V7X}:
    return f"/dev/vfio/{index}"
  else:
    return f"/dev/accel{index}"


def get_chip_owners() -> Dict[str, int]:
  """Returns a mapping of device paths to PIDs of processes using that device."""
  device_owners = {}

  for link in glob.glob("/proc/*/fd/*"):
    try:
      file = os.readlink(link)
    except FileNotFoundError:
      continue

    # Matches /dev/accel_ or /dev/vfio/_ ending with a postiive integer.
    if re.fullmatch(r"/dev/(?:accel|vfio/)\d+", file):
      match = re.fullmatch(r"/proc/(\d+)/fd/\d+", link)
      if not match:
        raise RuntimeError("Unknown link pattern", link)

      device_owners[file] = int(match.group(1))

  return device_owners
