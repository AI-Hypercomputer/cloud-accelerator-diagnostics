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
import io
from unittest import mock

from absl.testing import absltest
from tpu_info import cli
from tpu_info import device
from tpu_info import metrics
import grpc


class CliTest(absltest.TestCase):

  def test_print_chip_info(self):
    chip_owners = {
        "/dev/accel0": 123,
        "/dev/accel1": 456,
        "/dev/accel2": 789,
        "/dev/accel3": 901,
    }
    chip_usage = [
        metrics.Usage(0, 0, 34088157184, 1.0),
        metrics.Usage(1, 0, 34088157184, 2.05),
        metrics.Usage(2, 0, 34088157184, 3.055),
        metrics.Usage(3, 0, 34088157184, 4.001),
    ]
    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value=chip_owners
    ), mock.patch.object(
        metrics,
        "get_chip_usage",
        return_value=chip_usage,
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()

      output = f.getvalue()

      for dev, pid in chip_owners.items():
        self.assertIn(dev, output)
        self.assertIn(str(pid), output)

      for s in [
          "TPU v4 chip",
          # Check correct rounding for numerical values
          "0.00 GiB",
          "31.75 GiB",
          "1.00%",
          "2.05%",
          "3.06%",
          "4.00%",
      ]:
        self.assertIn(s, output)

  def test_print_chip_info_no_chips(self):
    with mock.patch.object(device, "get_local_chips", return_value=(None, 0)):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()

      output = f.getvalue()

      self.assertIn("No TPU chips found.", output)

  def test_print_chip_info_error(self):
    class FakeRpcError(grpc.RpcError):

      def code(self):
        return grpc.StatusCode.UNAVAILABLE

    with mock.patch.object(
        device, "get_local_chips", return_value=(device.TpuChip.V4, 4)
    ), mock.patch.object(
        device, "get_chip_owners", return_value={}
    ), mock.patch.object(
        metrics,
        "get_chip_usage",
        side_effect=FakeRpcError,
    ):
      with contextlib.redirect_stdout(io.StringIO()) as f:
        cli.print_chip_info()

      output = f.getvalue()
      self.assertIn("Libtpu metrics unavailable.", output)
      self.assertIn("TPU_RUNTIME_METRICS_PORTS", output)


if __name__ == "__main__":
  absltest.main()
