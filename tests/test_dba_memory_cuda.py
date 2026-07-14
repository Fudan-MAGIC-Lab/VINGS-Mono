import unittest

import torch

from scripts.profiling.dba_memory import (
    DBAMemorySampleBuilder,
    TorchCUDAMemoryBackend,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DBAMemoryCUDATests(unittest.TestCase):
    def test_real_cuda_sample_records_positive_peak(self):
        backend = TorchCUDAMemoryBackend("cuda:0")
        baseline = torch.empty(1024, device="cuda:0", dtype=torch.float32)
        builder = DBAMemorySampleBuilder(backend)
        builder.begin(
            signature={
                "active_edges": 8,
                "ba_edges": 8,
                "source_poses": 2,
                "pose_window": 2,
                "use_inactive": True,
                "upsample": True,
                "dtype": "float16",
                "feature_shape": [43, 77],
                "frontend_image_size": [344, 616],
                "mode": "vo",
                "backend": "torch",
            },
            sample_ordinal=1,
            storage={
                "corr_resident_storage_bytes": 0,
                "recurrent_resident_storage_bytes": 0,
                "state_resident_storage_bytes": 0,
                "excluded_tensor_count": 0,
            },
        )

        temporary = torch.empty(
            8 * 1024 * 1024,
            device="cuda:0",
            dtype=torch.uint8,
        )
        temporary.fill_(1)
        builder.mark("corr_sample_complete")
        payload = builder.finish()

        self.assertGreater(payload["entry"]["total_bytes"], 0)
        self.assertGreater(payload["entry"]["free_bytes"], 0)
        self.assertEqual(
            [stage["name"] for stage in payload["stages"]],
            ["corr_sample_complete"],
        )
        stage = payload["stages"][0]
        self.assertGreater(stage["peak_delta_bytes"], 0)
        self.assertIsInstance(stage["retained_delta_bytes"], int)
        self.assertIsInstance(stage["peak_delta_bytes"], int)

        del temporary
        del baseline


if __name__ == "__main__":
    unittest.main()
