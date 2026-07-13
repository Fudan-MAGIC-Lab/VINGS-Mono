import argparse
from pathlib import Path


DELTA_LAYER_NAMES = (
    "/delta/delta.0/Conv",
    "/delta/delta.2/Conv",
)
WEIGHT_LAYER_NAME = "/weight/weight.2/Conv"


def verify_delta_precision_log(text):
    summary = [line for line in text.splitlines() if line.startswith("Layer(")]
    fused = [
        line
        for line in summary
        if "/delta/delta.0/Conv" in line and "/weight/weight.0/Conv" in line
    ]
    if fused:
        raise ValueError("delta and weight layers remain fused")

    for name in DELTA_LAYER_NAMES:
        matches = [line for line in summary if name in line]
        if not matches or any("Half[" in line for line in matches):
            raise ValueError(f"{name} is not confirmed FP32")
        if not any("Float[" in line for line in matches):
            raise ValueError(f"{name} is not confirmed FP32")

    weight = [line for line in summary if WEIGHT_LAYER_NAME in line]
    if not weight or not any("Half[" in line for line in weight):
        raise ValueError("weight output layer is not confirmed FP16")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify DROID delta-head TensorRT precision.")
    parser.add_argument("log")
    args = parser.parse_args()
    verify_delta_precision_log(Path(args.log).read_text(errors="replace"))
    print("delta_fp32_precision=PASS")


if __name__ == "__main__":
    main()
