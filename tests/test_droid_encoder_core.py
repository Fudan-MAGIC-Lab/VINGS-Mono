import unittest

import torch

from scripts.frontend.modules.extractor import BasicEncoder


def legacy_forward(encoder, x):
    batch, count, channels, height, width = x.shape
    flat = x.view(batch * count, channels, height, width)
    flat = encoder.relu1(encoder.norm1(encoder.conv1(flat)))
    flat = encoder.layer1(flat)
    flat = encoder.layer2(flat)
    flat = encoder.layer3(flat)
    flat = encoder.conv2(flat)
    _, output_channels, output_height, output_width = flat.shape
    return flat.view(
        batch, count, output_channels, output_height, output_width
    )


class DroidEncoderCoreTests(unittest.TestCase):
    def test_forward_core_and_5d_wrapper_match_original_order(self):
        torch.manual_seed(7)
        for output_dim, norm_fn in ((128, "instance"), (256, "none")):
            encoder = BasicEncoder(output_dim=output_dim, norm_fn=norm_fn).eval()
            for batch, count in ((1, 1), (1, 4), (2, 3)):
                with self.subTest(
                    output_dim=output_dim,
                    norm_fn=norm_fn,
                    batch=batch,
                    count=count,
                ):
                    image = torch.randn(batch, count, 3, 32, 48)
                    with torch.no_grad():
                        expected = legacy_forward(encoder, image.clone())
                        core = encoder.forward_core(
                            image.view(batch * count, 3, 32, 48).clone()
                        )
                        actual = encoder(image.clone())

                    self.assertEqual(
                        tuple(core.shape),
                        (batch * count, output_dim, 4, 6),
                    )
                    self.assertEqual(
                        tuple(actual.shape),
                        (batch, count, output_dim, 4, 6),
                    )
                    self.assertTrue(torch.isfinite(core).all().item())
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(
                        core.view_as(expected), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
