# KV260 Motion SAD HLS Design

## Goal

Build and verify a minimal Vitis HLS accelerator that reduces two 320x180
grayscale frames to small motion signals. This first milestone is entirely
offline: it does not require a booted KV260, a microSD card, or a Jetson.

The accelerator will prove that the intended per-pixel SAD datapath can be
expressed as a one-pixel-per-cycle hardware pipeline before network transport,
frame buffering, and SLAM scheduling are added.

## Considered Approaches

### 1. Paired-pixel AXI4-Stream input (selected)

Each input word carries the previous and current grayscale values for one
pixel. The accelerator consumes exactly one fixed-size frame and emits scalar
motion statistics.

This is the smallest design that exercises the final arithmetic and streaming
interface. It avoids external memory and makes the C testbench deterministic.
Its limitation is that software must initially provide both frames.

### 2. Current-frame stream with an internal previous-frame buffer

The accelerator would accept only the current frame and retain the previous
frame in BRAM or URAM. This better matches the eventual runtime interface but
adds reset, first-frame, frame-boundary, and storage behavior before the core
arithmetic is validated.

### 3. AXI memory-mapped frame buffers

The accelerator would read two frames through AXI master ports. This is useful
for Linux/XRT integration but introduces address management, burst behavior,
and DMA concerns that are outside the first HLS learning milestone.

## Interface

The top-level function has these ports:

- One AXI4-Stream input with 16 data bits per word.
- `data[7:0]` is the previous-frame grayscale pixel.
- `data[15:8]` is the current-frame grayscale pixel.
- `keep` and `strb` must both be `0b11`, marking both data bytes valid.
- `last` must be asserted only on pixel 57,599.
- An 8-bit AXI4-Lite `change_threshold` control.
- A 32-bit AXI4-Lite `sad_sum` result.
- A 32-bit AXI4-Lite `changed_pixels` result.
- A 1-bit AXI4-Lite `frame_error` result.

The frame size is fixed at 320x180 for the first version. A pixel is counted as
changed when its absolute difference is strictly greater than
`change_threshold`.

The maximum SAD value is `255 * 57,600 = 14,688,000`, so a 32-bit accumulator
is sufficient. Normalization to a mean motion score remains a software task.

## Datapath

For up to 57,600 input words, the core will:

1. Unpack the previous and current 8-bit values.
2. Compute their unsigned absolute difference.
3. Accumulate the difference into `sad_sum`.
4. Increment `changed_pixels` when the difference exceeds the threshold.
5. Check `keep`, `strb`, and AXI4-Stream `last` against the frame contract.

The loop target is initiation interval 1, allowing one paired pixel per clock
after pipeline fill. The initial synthesis clock target is 100 MHz for the
K26 part used by KV260.

## Frame Error Behavior

`frame_error` is asserted when `last` appears before the final pixel, is
missing from the final pixel, or either byte qualifier is not `0b11`.

An early `last` ends the current transaction immediately. This preserves the
next packet already queued in the stream instead of consuming it as part of
the malformed frame. A missing final `last` consumes at most 57,600 words and
then returns an error. In either case, the caller must discard the partial
statistics and start a fresh core transaction with a complete frame.

## Files

Implementation will be isolated under `fpga/hls/motion_sad/`:

- `include/motion_sad_accel.hpp`: constants, AXI word type, and top-level declaration.
- `src/motion_sad.cpp`: synthesizable accelerator implementation.
- `tb/test_motion_sad.cpp`: self-checking C simulation testbench.
- `run_hls.tcl`: repeatable C simulation and C synthesis flow for Vitis HLS
  2023.1, with local NTFS staging for SSHFS-hosted workspaces.
- `run_hls.ps1`: checked Windows entry point that creates and cleans a unique
  staging directory and rejects Vitis error text even if the process exits 0.
- `README.md`: exact local commands and interpretation of generated reports.

Generated Vitis HLS projects and reports will be ignored rather than committed.

## Verification

The self-checking C testbench will cover:

- Identical frames: zero SAD, zero changed pixels, no frame error.
- Uniform known difference: exact SAD and changed-pixel count.
- Threshold boundary: a difference equal to the threshold is not counted.
- Sparse changed pixels: exact scalar results.
- Maximum reverse difference: accumulator range and unsigned subtraction.
- Early `last`: frame error asserted without consuming the following packet.
- Missing final `last`: frame error asserted.
- Invalid `keep` or `strb`: frame error asserted.

After C simulation passes, C synthesis must complete for the installed K26
part. The synthesis report will be checked for:

- Loop initiation interval 1.
- No unintended floating-point operators.
- Resource usage that is small relative to KV260 capacity.
- A timing estimate compatible with the 100 MHz target.

No claim of on-board operation will be made until KV260 PL power and boot are
available and a bitstream is actually downloaded.

## Out of Scope

This milestone does not include:

- Ethernet, UDP, USB, or Jetson communication.
- Linux, XRT, PetaLinux, or bare-metal software.
- DMA or external DDR frame buffers.
- Internal previous-frame storage.
- Block heatmaps, histograms, optical flow, or keypoint tracking.
- Integration with VINGS-Mono scheduling.
- Bitstream download to KV260.

The next hardware iteration may add internal frame storage and block-level SAD
outputs after this scalar streaming core is verified.
