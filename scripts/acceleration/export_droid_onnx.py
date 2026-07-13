import argparse
import copy
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from frontend.droid_net import DroidNet


class UpdateCoreExportWrapper(nn.Module):
    def __init__(self, update):
        super().__init__()
        self.update = update

    def forward(self, net, inp, corr, flow):
        return self.update.forward_core(net, inp, corr, flow)


def make_export_wrapper(update, replace_gradient_clip=False):
    export_update = copy.deepcopy(update) if replace_gradient_clip else update
    if replace_gradient_clip:
        export_update.delta[3] = nn.Identity()
        export_update.weight[3] = nn.Identity()
    return UpdateCoreExportWrapper(export_update)


def load_update_module(checkpoint, device="cuda"):
    network = DroidNet()
    state_dict = OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in torch.load(checkpoint, map_location="cpu").items()
    )
    for head in ("weight", "delta"):
        state_dict[f"update.{head}.2.weight"] = state_dict[f"update.{head}.2.weight"][:2]
        state_dict[f"update.{head}.2.bias"] = state_dict[f"update.{head}.2.bias"][:2]
    network.load_state_dict(state_dict)
    return network.update.to(device).eval()


def export_update_core(update, output, replace_gradient_clip=False):
    import onnx

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = make_export_wrapper(update, replace_gradient_clip=replace_gradient_clip).eval()
    device = next(wrapper.parameters()).device
    example_inputs = (
        torch.randn(16, 128, 43, 77, device=device),
        torch.randn(16, 128, 43, 77, device=device),
        torch.randn(16, 196, 43, 77, device=device),
        torch.randn(16, 4, 43, 77, device=device).clamp(-64.0, 64.0),
    )
    dynamic_axes = {
        name: {0: "edges"}
        for name in ("net", "inp", "corr", "flow", "updated_net", "delta", "weight")
    }
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(output),
            opset_version=17,
            input_names=["net", "inp", "corr", "flow"],
            output_names=["updated_net", "delta", "weight"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred, str(output))
    return wrapper


def validate_onnx(onnx_path, wrapper, edge_counts=(1, 4, 16, 32, 48), provider="CUDAExecutionProvider"):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=[provider])
    device = next(wrapper.parameters()).device
    results = []
    for edge_count in edge_counts:
        torch.manual_seed(1000 + edge_count)
        inputs = (
            torch.randn(edge_count, 128, 43, 77, device=device),
            torch.randn(edge_count, 128, 43, 77, device=device),
            torch.randn(edge_count, 196, 43, 77, device=device),
            torch.randn(edge_count, 4, 43, 77, device=device).clamp(-64.0, 64.0),
        )
        with torch.no_grad():
            expected = [tensor.detach().cpu().numpy() for tensor in wrapper(*inputs)]
        feed = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(("net", "inp", "corr", "flow"), inputs)
        }
        actual = session.run(["updated_net", "delta", "weight"], feed)
        output_results = []
        for name, actual_array, expected_array in zip(
            ("updated_net", "delta", "weight"), actual, expected
        ):
            if actual_array.shape != expected_array.shape:
                raise AssertionError(
                    f"{name} shape mismatch for E={edge_count}: "
                    f"{actual_array.shape} != {expected_array.shape}"
                )
            if not np.isfinite(actual_array).all():
                raise AssertionError(f"{name} contains non-finite values for E={edge_count}")
            actual_flat = actual_array.astype(np.float64, copy=False).reshape(-1)
            expected_flat = expected_array.astype(np.float64, copy=False).reshape(-1)
            cosine = float(
                np.dot(actual_flat, expected_flat)
                / max(np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat), 1e-12)
            )
            mean_relative_error = float(
                np.mean(np.abs(actual_flat - expected_flat) / np.maximum(np.abs(expected_flat), 1e-3))
            )
            if cosine < 0.999:
                raise AssertionError(f"{name} cosine {cosine:.6f} < 0.999 for E={edge_count}")
            if mean_relative_error > 0.01:
                raise AssertionError(
                    f"{name} mean relative error {mean_relative_error:.6f} > 0.01 for E={edge_count}"
                )
            output_results.append(
                {
                    "name": name,
                    "shape": list(actual_array.shape),
                    "cosine": cosine,
                    "mean_relative_error": mean_relative_error,
                }
            )
        results.append({"edge_count": edge_count, "outputs": output_results})
    return results


def main():
    parser = argparse.ArgumentParser(description="Export the convolutional DROID update core to ONNX.")
    parser.add_argument("--checkpoint", default="ckpts/droid.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replace-gradient-clip", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    args = parser.parse_args()

    update = load_update_module(args.checkpoint, device=args.device)
    wrapper = export_update_core(
        update,
        args.output,
        replace_gradient_clip=args.replace_gradient_clip,
    )
    print(f"exported={args.output} replace_gradient_clip={int(args.replace_gradient_clip)}")
    if args.validate:
        for result in validate_onnx(args.output, wrapper, provider=args.provider):
            print(result)


if __name__ == "__main__":
    main()
