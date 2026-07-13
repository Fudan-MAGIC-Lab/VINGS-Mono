import unittest
from unittest import mock

import numpy as np
import torch

from scripts.frontend.motion_gate import JetsonMotionGate, _import_vpi


def make_image(offset_x=0):
    image = np.zeros((1, 3, 64, 96), dtype=np.uint8)
    image[:, :, 20:36, 24 + offset_x:40 + offset_x] = 255
    return torch.from_numpy(image)


class JetsonMotionGateTests(unittest.TestCase):
    def test_disabled_gate_always_runs(self):
        gate = JetsonMotionGate({"jetson_motion_gate": {"enabled": False}})

        first = gate.decide(make_image())
        second = gate.decide(make_image())

        self.assertFalse(first["skip"])
        self.assertFalse(second["skip"])
        self.assertEqual(second["reason"], "disabled")

    def test_enabled_gate_skips_low_motion_after_initial_frame(self):
        gate = JetsonMotionGate(
            {
                "jetson_motion_gate": {
                    "enabled": True,
                    "backend": "opencv",
                    "threshold": 1.0,
                    "force_interval": 0,
                    "resize": [64, 96],
                }
            }
        )

        first = gate.decide(make_image())
        second = gate.decide(make_image())

        self.assertFalse(first["skip"])
        self.assertEqual(first["reason"], "initial")
        self.assertTrue(second["skip"])
        self.assertEqual(second["reason"], "low_motion")
        self.assertLess(second["score"], 1.0)

    def test_enabled_gate_runs_on_high_motion(self):
        gate = JetsonMotionGate(
            {
                "jetson_motion_gate": {
                    "enabled": True,
                    "backend": "opencv",
                    "threshold": 1.0,
                    "force_interval": 0,
                    "resize": [64, 96],
                }
            }
        )

        gate.decide(make_image())
        moved = gate.decide(make_image(offset_x=16))

        self.assertFalse(moved["skip"])
        self.assertEqual(moved["reason"], "motion")
        self.assertGreaterEqual(moved["score"], 1.0)

    def test_force_interval_runs_even_with_low_motion(self):
        gate = JetsonMotionGate(
            {
                "jetson_motion_gate": {
                    "enabled": True,
                    "backend": "opencv",
                    "threshold": 100.0,
                    "force_interval": 2,
                    "resize": [64, 96],
                }
            }
        )

        gate.decide(make_image())
        low = gate.decide(make_image())
        forced = gate.decide(make_image())

        self.assertTrue(low["skip"])
        self.assertFalse(forced["skip"])
        self.assertEqual(forced["reason"], "force_interval")

    def test_vpi_backend_uses_ofa_when_supported(self):
        try:
            _import_vpi()
        except Exception as exc:
            self.skipTest(f"VPI unavailable: {exc}")

        gate = JetsonMotionGate(
            {
                "jetson_motion_gate": {
                    "enabled": True,
                    "backend": "vpi",
                    "threshold": 1.0,
                    "force_interval": 0,
                    "resize": [64, 96],
                    "vpi_num_levels": 1,
                }
            }
        )

        gate.decide(make_image())
        moved = gate.decide(make_image(offset_x=16))

        self.assertIsNone(gate.vpi_disabled_reason)
        self.assertEqual(moved["backend"], "vpi")
        self.assertFalse(moved["skip"])
        self.assertGreaterEqual(moved["score"], 1.0)

    def test_vpi_backend_reuses_previous_preprocessed_frame(self):
        gate = JetsonMotionGate(
            {
                "jetson_motion_gate": {
                    "enabled": True,
                    "backend": "vpi",
                    "threshold": 100.0,
                    "force_interval": 0,
                    "resize": [64, 96],
                }
            }
        )
        converted = []

        def fake_convert(gray):
            token = f"frame-{len(converted)}"
            converted.append(gray.copy())
            return token

        def fake_score(prev_frame, frame):
            self.assertEqual(prev_frame, "frame-0")
            self.assertEqual(frame, "frame-1")
            return 0.0

        gate._to_vpi_ofa_frame = fake_convert
        gate._compute_score_vpi_frames = fake_score

        gate.decide(make_image())
        gate.decide(make_image(offset_x=1))

        self.assertEqual(len(converted), 2)
        self.assertEqual(gate._prev_vpi_frame, "frame-1")

    def test_vpi_single_level_uses_direct_block_linear_image(self):
        calls = []

        class FakeImage:
            def gaussian_pyramid(self, levels, backend=None):
                calls.append(("gaussian_pyramid", levels, backend))
                return self

            def convert(self, fmt, backend=None):
                calls.append(("convert", fmt, backend))
                return "block-linear-image"

        class FakeVpi:
            class Format:
                Y8_ER = "Y8_ER"
                Y8_ER_BL = "Y8_ER_BL"

            class Backend:
                CUDA = "CUDA"
                VIC = "VIC"

            @staticmethod
            def asimage(gray, fmt):
                calls.append(("asimage", fmt, tuple(gray.shape)))
                return FakeImage()

        gate = JetsonMotionGate(
            {"jetson_motion_gate": {"enabled": True, "backend": "vpi", "vpi_num_levels": 1}}
        )

        with mock.patch("scripts.frontend.motion_gate._import_vpi", return_value=FakeVpi):
            frame = gate._to_vpi_ofa_frame(np.zeros((16, 16), dtype=np.uint8))

        self.assertEqual(frame, "block-linear-image")
        self.assertNotIn(("gaussian_pyramid", 1, "CUDA"), calls)
        self.assertIn(("convert", "Y8_ER_BL", "VIC"), calls)

    def test_vpi_cpp_backend_uses_cpp_gate_with_raw_image_preprocess(self):
        calls = []

        class FakeCppGate:
            def __init__(self, height, width, grid_size, quality, percentile):
                calls.append(("init", height, width, grid_size, quality, percentile))

            def score_image(self, image):
                calls.append(("score_image", tuple(image.shape)))
                return len([call for call in calls if call[0] == "score_image"]) > 1, 4.25

        fake_module = type("FakeModule", (), {"VpiOfaMotionGate": FakeCppGate})
        gate = JetsonMotionGate(
            {
                "jetson_motion_gate": {
                    "enabled": True,
                    "backend": "vpi_cpp",
                    "threshold": 1.0,
                    "force_interval": 0,
                    "resize": [64, 96],
                    "grid_size": 4,
                    "vpi_quality": "low",
                }
            }
        )

        with mock.patch("scripts.frontend.motion_gate._import_vpi_ofa_cpp", return_value=fake_module):
            first = gate.decide(make_image())
            second = gate.decide(make_image(offset_x=8))

        self.assertFalse(first["skip"])
        self.assertEqual(first["backend"], "vpi_cpp")
        self.assertFalse(second["skip"])
        self.assertEqual(second["backend"], "vpi_cpp")
        self.assertEqual(calls[0], ("init", 64, 96, 4, "low", 90.0))
        self.assertEqual(calls[1], ("score_image", (3, 64, 96)))
        self.assertEqual(calls[2], ("score_image", (3, 64, 96)))

    def test_decide_packet_prefers_nvmm_backend_when_fd_is_available(self):
        calls = []

        class FakeNvmmGate:
            def __init__(self, height, width, grid_size, quality, percentile):
                calls.append(("init", height, width, grid_size, quality, percentile))

            def score_nvbuffer(self, fd, width, height, fmt):
                calls.append(("score_nvbuffer", fd, width, height, fmt))
                return len([call for call in calls if call[0] == "score_nvbuffer"]) > 1, 5.0

        fake_module = type("FakeModule", (), {"VpiOfaNvBufferMotionGate": FakeNvmmGate})
        gate = JetsonMotionGate(
            {
                "jetson_motion_gate": {
                    "enabled": True,
                    "backend": "vpi_nvbuffer",
                    "threshold": 1.0,
                    "force_interval": 0,
                    "resize": [64, 96],
                }
            }
        )
        packet = {
            "rgb": make_image(),
            "motion_gate_dmabuf_fd": 12,
            "motion_gate_nvmm_width": 192,
            "motion_gate_nvmm_height": 128,
            "motion_gate_nvmm_format": "NV12",
        }

        with mock.patch("scripts.frontend.motion_gate._import_vpi_ofa_cpp", return_value=fake_module):
            first = gate.decide_packet(packet)
            second = gate.decide_packet(packet)

        self.assertFalse(first["skip"])
        self.assertEqual(first["backend"], "vpi_nvbuffer")
        self.assertFalse(second["skip"])
        self.assertEqual(second["backend"], "vpi_nvbuffer")
        self.assertEqual(calls[0], ("init", 64, 96, 4, "low", 90.0))
        self.assertEqual(calls[1], ("score_nvbuffer", 12, 192, 128, "NV12"))
        self.assertEqual(calls[2], ("score_nvbuffer", 12, 192, 128, "NV12"))


if __name__ == "__main__":
    unittest.main()
