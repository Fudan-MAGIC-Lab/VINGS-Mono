# Jetson Deployment Notes

This repository is written for an x86 + CUDA 11.8 environment by default. For Jetson, start with a headless SmallCity smoke test and keep the first run as small as possible.

## Recommended first target

- Model: NVIDIA Orin NX Developer Kit
- Mode: headless, no GUI
- Dataset: `configs/hierarchical/smallcity.yaml`
- Entry point: `scripts/run.py`

## What changed for Jetson

- `scripts/run.py` now accepts CLI overrides for dataset root, output directory, weight path, tracker device, mapper device, and headless/metric toggles.
- `scripts/run_jetson_smallcity.sh` wraps the common Jetson launch settings.

## Suggested environment order

1. Create or reuse a Jetson-compatible Python environment.
2. Install a Jetson/aarch64 build of PyTorch instead of the x86 `cu118` wheels from `set_env.sh`.
3. Install the Python dependencies that are available as wheels first.
4. Build the CUDA extensions in `submodules/dbaf` and `submodules/diff-surfel-rasterization`.
5. Download the required checkpoints.
6. Point the dataset root and output directory to local Jetson storage.

## Mirror-first hooks for network issues

When package downloads are unstable, use the local hooks to try domestic mirrors first and then fallback to the official index.

```bash
cd /home/jetson/VINGS-Mono
source hooks/cn_mirror_hooks.sh
```

After sourcing, you can use:

- `pipi <pip install args>`
- `pipd <pip download args>`

Examples:

```bash
pipi torchvision==0.16.1
pipd torchvision==0.16.1 -d /tmp/wheels
```

You can also call executable hook scripts directly:

```bash
hooks/pip-install-hook.sh torchvision==0.16.1
hooks/pip-download-hook.sh torchvision==0.16.1 -d /tmp/wheels
```

## Smoke-test command

```bash
bash scripts/run_jetson_smallcity.sh
```

Override paths if needed:

```bash
VINGS_DATASET_ROOT=/path/to/smallcity \
VINGS_OUTPUT_DIR=/path/to/output \
VINGS_DROID_WEIGHT=/path/to/droid.pth \
bash scripts/run_jetson_smallcity.sh
```

## Notes

- The first run should stay headless by forcing `--no-vis`.
- Metric depth is disabled in the Jetson wrapper to reduce memory pressure.
- If the run fails, capture the first import or CUDA extension error before changing multiple components.