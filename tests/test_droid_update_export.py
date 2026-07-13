import pathlib
import subprocess
import sys
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from acceleration.export_droid_onnx import make_export_wrapper
from frontend.droid_net import UpdateModule
from frontend.modules.clipping import GradientClip


class DroidUpdateExportTests(unittest.TestCase):
    def test_export_script_supports_direct_execution(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "acceleration" / "export_droid_onnx.py"),
                "--help",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Export the convolutional DROID update core", result.stdout)

    def test_export_wrapper_matches_forward_core_without_mutating_runtime_module(self):
        torch.manual_seed(13)
        update = UpdateModule().eval()
        wrapper = make_export_wrapper(update, replace_gradient_clip=True).eval()
        inputs = (
            torch.randn(4, 128, 3, 5),
            torch.randn(4, 128, 3, 5),
            torch.randn(4, 196, 3, 5),
            torch.randn(4, 4, 3, 5),
        )

        with torch.no_grad():
            expected = update.forward_core(*inputs)
            actual = wrapper(*inputs)

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

        self.assertIsInstance(update.delta[3], GradientClip)
        self.assertIsInstance(update.weight[3], GradientClip)
        self.assertIsInstance(wrapper.update.delta[3], torch.nn.Identity)
        self.assertIsInstance(wrapper.update.weight[3], torch.nn.Identity)
        self.assertIsNot(wrapper.update, update)


if __name__ == "__main__":
    unittest.main()
