import pathlib
import sys
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from frontend.droid_net import UpdateModule


def reference_core(module, net, inp, corr, flow):
    corr_features = module.corr_encoder(corr)
    flow_features = module.flow_encoder(flow)
    updated_net = module.gru(net, inp, corr_features, flow_features)
    delta = module.delta(updated_net)
    weight = module.weight(updated_net)
    return updated_net, delta, weight


class DroidUpdateCoreTests(unittest.TestCase):
    def test_forward_core_matches_original_computation_for_supported_edge_counts(self):
        torch.manual_seed(7)
        module = UpdateModule().eval()

        for edge_count in (1, 4, 16, 32, 48):
            with self.subTest(edge_count=edge_count), torch.no_grad():
                net = torch.randn(edge_count, 128, 3, 5)
                inp = torch.randn(edge_count, 128, 3, 5)
                corr = torch.randn(edge_count, 196, 3, 5)
                flow = torch.randn(edge_count, 4, 3, 5).clamp(-64.0, 64.0)

                expected = reference_core(module, net, inp, corr, flow)
                actual = module.forward_core(net, inp, corr, flow)

                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual_tensor).all())

                wrapped = module(
                    net.unsqueeze(0),
                    inp.unsqueeze(0),
                    corr.unsqueeze(0),
                    flow.unsqueeze(0),
                    upsample=False,
                )
                torch.testing.assert_close(wrapped[0].squeeze(0), expected[0], rtol=0, atol=0)
                torch.testing.assert_close(
                    wrapped[1].squeeze(0).permute(0, 3, 1, 2), expected[1], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    wrapped[2].squeeze(0).permute(0, 3, 1, 2), expected[2], rtol=0, atol=0
                )
                self.assertEqual(len(wrapped), 3)


if __name__ == "__main__":
    unittest.main()
