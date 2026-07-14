# Motion SAD HLS Core

This is the first offline FPGA milestone for the Jetson + KV260 motion
scheduling path. It consumes paired 320x180 grayscale frames and returns small
motion statistics. A booted KV260 is not required for C simulation or HLS
synthesis.

## Interface

Each 16-bit AXI4-Stream word represents one pixel:

- Bits 7:0: previous-frame grayscale value.
- Bits 15:8: current-frame grayscale value.
- TKEEP and TSTRB: both must be `0b11`.
- TLAST: asserted only on pixel 57,599.

The AXI4-Lite control interface provides an 8-bit `change_threshold`. Results
are total SAD, the number of pixels whose absolute difference is strictly
greater than the threshold, and a frame-format error flag.

An early TLAST ends the malformed packet immediately so the next packet is
not consumed. A missing final TLAST consumes at most 57,600 words. Invalid
TLAST, TKEEP, or TSTRB asserts `frame_error`; software must discard the
reported partial statistics.

## Run C simulation

From this directory in PowerShell:

```powershell
.\run_hls.ps1
```

Success includes `All motion_sad tests passed` and `CSim done with 0 errors`.

## Run K26 synthesis

```powershell
.\run_hls.ps1 -Synthesize
```

The PowerShell wrapper stages the source files in a unique directory under
`%LOCALAPPDATA%` before invoking the HLS compiler, then removes that directory
after Vitis exits. This avoids a Vitis HLS 2023.1 Clang file-identity failure
when the repository is accessed through SSHFS/WinFSP. It also treats Vitis
`ERROR:` output as failure because this version can return process exit code 0
after a Tcl error. Old copied reports are removed before each run, so a failed
or CSim-only run cannot leave a stale synthesis result. C simulation and
synthesis reports are copied back under the ignored `build/` directory.

Vitis HLS 2023.1 does not handle a quoted include path correctly in this flow,
so the local staging path must not contain spaces.

The main reports are:

```text
build/motion_sad_prj/solution1/csim/report/motion_sad_csim.log
build/motion_sad_prj/solution1/syn/report/motion_sad_csynth.rpt
```

## Current synthesis result

For `xck26-sfvc784-2LV-c` with a 10 ns target clock, Vitis HLS 2023.1 reports:

- Frame loop achieved initiation interval: 1.
- Valid full-frame latency: 57,602 cycles, or 0.576 ms at 100 MHz. An early
  TLAST can return in as few as 2 cycles.
- Estimated clock period: 2.799 ns (`Fmax` estimate 357.27 MHz).
- Resources: 551 LUT, 274 FF, 0 DSP, 0 BRAM, and 0 URAM.

These are HLS estimates, not post-place-and-route timing or on-board
measurements. Generated `build/` contents are local artifacts and are not
committed.
