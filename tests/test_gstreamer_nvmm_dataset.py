import unittest
from unittest import mock

import numpy as np
import torch

from scripts.datasets.gstreamer_nvmm import (
    GStreamerNVMMFrame,
    GStreamerNVMMImageStream,
    extract_dmabuf_fd,
)


class GStreamerNVMMDatasetTests(unittest.TestCase):
    def test_extract_dmabuf_fd_returns_first_dmabuf_memory_fd(self):
        class FakeMemory:
            pass

        class FakeBuffer:
            def n_memory(self):
                return 1

            def peek_memory(self, index):
                if index != 0:
                    raise AssertionError(index)
                return FakeMemory()

        fake_allocators = mock.Mock()
        fake_allocators.is_dmabuf_memory.return_value = True
        fake_allocators.dmabuf_memory_get_fd.return_value = 42

        fd = extract_dmabuf_fd(FakeBuffer(), gst_allocators=fake_allocators)

        self.assertEqual(fd, 42)

    def test_packet_contains_rgb_tensor_and_nvmm_metadata(self):
        cfg = {
            "dataset": {
                "pipeline": "unused",
                "length": 1,
                "timestamp_start": 10.0,
                "timestamp_step": 0.5,
            },
            "frontend": {"image_size": [2, 4]},
            "intrinsic": {"H": 2, "W": 4, "fu": 8.0, "fv": 10.0, "cu": 2.0, "cv": 1.0},
            "device": {"tracker": "cpu"},
        }
        stream = GStreamerNVMMImageStream(cfg, frame_source=mock.Mock())
        frame = GStreamerNVMMFrame(
            rgb=np.full((2, 4, 3), fill_value=7, dtype=np.uint8),
            dmabuf_fd=11,
            width=4,
            height=2,
            format="NV12",
        )
        stream.frame_source.pull_frame.return_value = frame

        packet = stream[0]

        self.assertEqual(packet["timestamp"], 10.0)
        self.assertEqual(tuple(packet["rgb"].shape), (1, 3, 2, 4))
        self.assertEqual(packet["rgb"].dtype, torch.uint8)
        self.assertEqual(packet["motion_gate_dmabuf_fd"], 11)
        self.assertEqual(packet["motion_gate_nvmm_width"], 4)
        self.assertEqual(packet["motion_gate_nvmm_height"], 2)
        self.assertEqual(packet["motion_gate_nvmm_format"], "NV12")
        self.assertTrue(torch.equal(packet["rgb"][0, :, 0, 0], torch.tensor([7, 7, 7], dtype=torch.uint8)))


if __name__ == "__main__":
    unittest.main()
