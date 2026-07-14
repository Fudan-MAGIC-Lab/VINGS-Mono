# KV260 Motion SAD HLS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a Vitis HLS 2023.1 accelerator that reduces paired 320x180 grayscale frames to scalar SAD and changed-pixel motion signals.

**Architecture:** A fixed-length AXI4-Stream carries one previous/current 8-bit pixel pair per word. One pipelined loop accumulates absolute differences, counts differences above an AXI4-Lite threshold, and validates TLAST; a self-checking C++ testbench runs before K26 synthesis.

**Tech Stack:** Vitis HLS 2023.1, synthesizable C++14, `hls::stream`, `ap_axiu`, Tcl, K26 part `xck26-sfvc784-2LV-c`.

---

## File Map

- `fpga/hls/motion_sad/include/motion_sad_accel.hpp`: constants, stream type, and top-level declaration.
- `fpga/hls/motion_sad/src/motion_sad.cpp`: synthesizable datapath and HLS interfaces.
- `fpga/hls/motion_sad/tb/test_motion_sad.cpp`: self-checking behavioral and TLAST tests.
- `fpga/hls/motion_sad/run_hls.tcl`: repeatable C simulation and synthesis flow.
- `fpga/hls/motion_sad/.gitignore`: generated project exclusion.
- `fpga/hls/motion_sad/README.md`: commands and interface documentation.

### Task 1: Establish the failing C simulation

**Files:**
- Create: `fpga/hls/motion_sad/include/motion_sad_accel.hpp`
- Create: `fpga/hls/motion_sad/src/motion_sad.cpp`
- Create: `fpga/hls/motion_sad/tb/test_motion_sad.cpp`
- Create: `fpga/hls/motion_sad/run_hls.tcl`
- Create: `fpga/hls/motion_sad/.gitignore`

- [ ] **Step 1: Create the public interface**

```cpp
#ifndef MOTION_SAD_ACCEL_HPP
#define MOTION_SAD_ACCEL_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

constexpr int MOTION_FRAME_WIDTH = 320;
constexpr int MOTION_FRAME_HEIGHT = 180;
constexpr int MOTION_FRAME_PIXELS = MOTION_FRAME_WIDTH * MOTION_FRAME_HEIGHT;

using motion_axis_t = ap_axiu<16, 0, 0, 0>;

void motion_sad(
    hls::stream<motion_axis_t>& pixel_pairs,
    ap_uint<8> change_threshold,
    ap_uint<32>& sad_sum,
    ap_uint<32>& changed_pixels,
    ap_uint<1>& frame_error);

#endif
```

- [ ] **Step 2: Add a compilable red-phase stub**

```cpp
#include "motion_sad_accel.hpp"

void motion_sad(
    hls::stream<motion_axis_t>& pixel_pairs,
    ap_uint<8> change_threshold,
    ap_uint<32>& sad_sum,
    ap_uint<32>& changed_pixels,
    ap_uint<1>& frame_error) {
    (void)change_threshold;
    for (int i = 0; i < MOTION_FRAME_PIXELS; ++i) {
        (void)pixel_pairs.read();
    }
    sad_sum = 0;
    changed_pixels = 0;
    frame_error = 0;
}
```

- [ ] **Step 3: Add the self-checking testbench**

```cpp
#include "motion_sad_accel.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

struct Result {
    uint32_t sad_sum;
    uint32_t changed_pixels;
    bool frame_error;
};

int failures = 0;

template <typename PreviousPixel, typename CurrentPixel>
Result run_frame(PreviousPixel previous_pixel, CurrentPixel current_pixel,
                 uint8_t threshold,
                 int last_index = MOTION_FRAME_PIXELS - 1) {
    hls::stream<motion_axis_t> input;
    for (int i = 0; i < MOTION_FRAME_PIXELS; ++i) {
        motion_axis_t word;
        word.data.range(7, 0) = ap_uint<8>(previous_pixel(i));
        word.data.range(15, 8) = ap_uint<8>(current_pixel(i));
        word.keep = -1;
        word.strb = -1;
        word.last = i == last_index;
        input.write(word);
    }

    ap_uint<32> sad_sum = 0;
    ap_uint<32> changed_pixels = 0;
    ap_uint<1> frame_error = 0;
    motion_sad(input, threshold, sad_sum, changed_pixels, frame_error);
    return {static_cast<uint32_t>(sad_sum),
            static_cast<uint32_t>(changed_pixels),
            static_cast<bool>(frame_error)};
}

template <typename Actual, typename Expected>
void expect_equal(const std::string& test_name, const std::string& field,
                  Actual actual, Expected expected) {
    if (actual != static_cast<Actual>(expected)) {
        std::cerr << "[FAIL] " << test_name << " " << field
                  << ": expected " << expected << ", got " << actual << '\n';
        ++failures;
    }
}

void test_identical_frames() {
    const Result result = run_frame(
        [](int i) { return static_cast<uint8_t>(i & 0xff); },
        [](int i) { return static_cast<uint8_t>(i & 0xff); }, 5);
    expect_equal("identical", "sad_sum", result.sad_sum, 0u);
    expect_equal("identical", "changed_pixels", result.changed_pixels, 0u);
    expect_equal("identical", "frame_error", result.frame_error, false);
}

void test_uniform_difference() {
    const Result result = run_frame(
        [](int) { return static_cast<uint8_t>(10); },
        [](int) { return static_cast<uint8_t>(20); }, 5);
    expect_equal("uniform", "sad_sum", result.sad_sum, 576000u);
    expect_equal("uniform", "changed_pixels", result.changed_pixels, 57600u);
    expect_equal("uniform", "frame_error", result.frame_error, false);
}

void test_threshold_boundary() {
    const Result result = run_frame(
        [](int) { return static_cast<uint8_t>(40); },
        [](int) { return static_cast<uint8_t>(47); }, 7);
    expect_equal("threshold", "sad_sum", result.sad_sum, 403200u);
    expect_equal("threshold", "changed_pixels", result.changed_pixels, 0u);
    expect_equal("threshold", "frame_error", result.frame_error, false);
}

void test_sparse_changes() {
    const Result result = run_frame(
        [](int) { return static_cast<uint8_t>(0); },
        [](int i) { return static_cast<uint8_t>(i % 1000 == 0 ? 100 : 0); },
        50);
    expect_equal("sparse", "sad_sum", result.sad_sum, 5800u);
    expect_equal("sparse", "changed_pixels", result.changed_pixels, 58u);
    expect_equal("sparse", "frame_error", result.frame_error, false);
}

void test_early_last() {
    const Result result = run_frame(
        [](int) { return static_cast<uint8_t>(0); },
        [](int) { return static_cast<uint8_t>(0); }, 0, 100);
    expect_equal("early_last", "frame_error", result.frame_error, true);
}

void test_missing_last() {
    const Result result = run_frame(
        [](int) { return static_cast<uint8_t>(0); },
        [](int) { return static_cast<uint8_t>(0); }, 0, -1);
    expect_equal("missing_last", "frame_error", result.frame_error, true);
}

}  // namespace

int main() {
    test_identical_frames();
    test_uniform_difference();
    test_threshold_boundary();
    test_sparse_changes();
    test_early_last();
    test_missing_last();

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "All motion_sad tests passed\n";
    return 0;
}
```

- [ ] **Step 4: Add the HLS script and ignore generated output**

```tcl
set source_dir [file dirname [file normalize [info script]]]
set run_synthesis [expr {
    [info exists ::env(MOTION_SAD_SYNTH)] &&
    $::env(MOTION_SAD_SYNTH) eq "1"
}]

if {[info exists ::env(LOCALAPPDATA)]} {
    set local_root $::env(LOCALAPPDATA)
} elseif {[info exists ::env(TEMP)]} {
    set local_root $::env(TEMP)
} else {
    error "LOCALAPPDATA or TEMP is required for the local HLS staging area"
}

set stage_dir [file join $local_root VINGS-Mono hls motion_sad_2023_1]
file delete -force $stage_dir
foreach relative_path {
    include/motion_sad_accel.hpp
    src/motion_sad.cpp
    tb/test_motion_sad.cpp
} {
    set source_path [file join $source_dir $relative_path]
    set stage_path [file join $stage_dir $relative_path]
    file mkdir [file dirname $stage_path]
    file copy -force $source_path $stage_path
}

set build_dir [file join $stage_dir build]
file mkdir $build_dir
cd $build_dir
open_project -reset motion_sad_prj
set_top motion_sad
add_files [file join $stage_dir src motion_sad.cpp] \
    -cflags "-I[file join $stage_dir include] -std=c++14"
add_files -tb [file join $stage_dir tb test_motion_sad.cpp] \
    -cflags "-I[file join $stage_dir include] -std=c++14"
open_solution -reset solution1
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
csim_design -clean

set project_dir [file join $build_dir motion_sad_prj]
set report_dir [file join $source_dir build motion_sad_prj solution1]
file mkdir [file join $report_dir csim report]
file copy -force \
    [file join $project_dir solution1 csim report motion_sad_csim.log] \
    [file join $report_dir csim report motion_sad_csim.log]

if {$run_synthesis} {
    csynth_design
    file mkdir [file join $report_dir syn report]
    file copy -force \
        [file join $project_dir solution1 syn report motion_sad_csynth.rpt] \
        [file join $report_dir syn report motion_sad_csynth.rpt]
}
exit
```

```gitignore
build/
```

- [ ] **Step 5: Run C simulation and verify the red phase**

```powershell
& 'D:\xilinx\Vitis_HLS\2023.1\bin\vitis_hls.bat' -f run_hls.tcl
```

Expected: nonzero exit. Uniform, threshold, sparse, early-TLAST, and missing-TLAST assertions fail, proving the testbench detects the stub.

### Task 2: Implement the pipelined SAD datapath

**Files:**
- Modify: `fpga/hls/motion_sad/src/motion_sad.cpp`
- Test: `fpga/hls/motion_sad/tb/test_motion_sad.cpp`

- [ ] **Step 1: Replace the stub with the synthesizable implementation**

```cpp
#include "motion_sad_accel.hpp"

void motion_sad(
    hls::stream<motion_axis_t>& pixel_pairs,
    ap_uint<8> change_threshold,
    ap_uint<32>& sad_sum,
    ap_uint<32>& changed_pixels,
    ap_uint<1>& frame_error) {
#pragma HLS INTERFACE axis port=pixel_pairs
#pragma HLS INTERFACE s_axilite port=change_threshold bundle=control
#pragma HLS INTERFACE s_axilite port=sad_sum bundle=control
#pragma HLS INTERFACE s_axilite port=changed_pixels bundle=control
#pragma HLS INTERFACE s_axilite port=frame_error bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    ap_uint<32> sad_accumulator = 0;
    ap_uint<32> changed_accumulator = 0;
    ap_uint<1> error = 0;

    for (int i = 0; i < MOTION_FRAME_PIXELS; ++i) {
#pragma HLS PIPELINE II=1
        const motion_axis_t word = pixel_pairs.read();
        const ap_uint<8> previous = word.data.range(7, 0);
        const ap_uint<8> current = word.data.range(15, 8);
        const ap_uint<8> difference =
            current >= previous ? current - previous : previous - current;
        const bool expected_last = i == MOTION_FRAME_PIXELS - 1;

        sad_accumulator += difference;
        if (difference > change_threshold) {
            ++changed_accumulator;
        }
        if (static_cast<bool>(word.last) != expected_last) {
            error = 1;
        }
    }

    sad_sum = sad_accumulator;
    changed_pixels = changed_accumulator;
    frame_error = error;
}
```

- [ ] **Step 2: Run C simulation and verify the green phase**

```powershell
& 'D:\xilinx\Vitis_HLS\2023.1\bin\vitis_hls.bat' -f run_hls.tcl
```

Expected: exit 0, `All motion_sad tests passed`, and `CSim done with 0 errors`.

- [ ] **Step 3: Commit the tested accelerator**

```powershell
git add fpga/hls/motion_sad/include/motion_sad_accel.hpp fpga/hls/motion_sad/src/motion_sad.cpp fpga/hls/motion_sad/tb/test_motion_sad.cpp fpga/hls/motion_sad/run_hls.tcl fpga/hls/motion_sad/.gitignore
git commit -m "feat: add KV260 motion SAD HLS core"
```

### Task 3: Synthesize for K26 and document the workflow

**Files:**
- Create: `fpga/hls/motion_sad/README.md`
- Inspect: `fpga/hls/motion_sad/build/motion_sad_prj/solution1/syn/report/motion_sad_csynth.rpt`

- [ ] **Step 1: Run C simulation and C synthesis**

```powershell
$env:MOTION_SAD_SYNTH = '1'
& 'D:\xilinx\Vitis_HLS\2023.1\bin\vitis_hls.bat' -f run_hls.tcl
Remove-Item Env:MOTION_SAD_SYNTH
```

Expected: exit 0, passing C simulation, and a report for `xck26-sfvc784-2LV-c` with a 10 ns target clock.

- [ ] **Step 2: Verify pipeline, timing, and resources**

```powershell
$report = 'build\motion_sad_prj\solution1\syn\report\motion_sad_csynth.rpt'
Select-String -Path $report -Pattern 'Timing|Latency|Interval|PIPELINE|DSP|BRAM|LUT|FF'
```

Expected: the frame loop achieves II 1, estimated period is below 10 ns, no floating-point operators appear, and scalar-core resource use is small relative to K26 capacity. If synthesis fails or II exceeds 1, stop at the first scheduling/dependency diagnostic before changing code.

- [ ] **Step 3: Add `README.md`**

```markdown
# Motion SAD HLS Core

This offline FPGA milestone consumes paired 320x180 grayscale frames and
returns scalar motion statistics. It does not require a booted KV260.

## Input

Each AXI4-Stream word is 16 bits: bits 7:0 contain the previous pixel, bits
15:8 contain the current pixel, and TLAST is asserted only on pixel 57,599.
The core returns total SAD, the number of differences strictly greater than
`change_threshold`, and a frame-format error flag.

## C simulation

```powershell
& 'D:\xilinx\Vitis_HLS\2023.1\bin\vitis_hls.bat' -f run_hls.tcl
```

Success includes `All motion_sad tests passed` and `CSim done with 0 errors`.

## K26 synthesis

```powershell
$env:MOTION_SAD_SYNTH = '1'
& 'D:\xilinx\Vitis_HLS\2023.1\bin\vitis_hls.bat' -f run_hls.tcl
Remove-Item Env:MOTION_SAD_SYNTH
```

The report is at
`build/motion_sad_prj/solution1/syn/report/motion_sad_csynth.rpt`. Check the
frame loop for II 1, the timing estimate against 10 ns, and BRAM, DSP, LUT,
and FF utilization. The generated `build/` directory is not committed.
```

- [ ] **Step 4: Re-run complete verification**

```powershell
$env:MOTION_SAD_SYNTH = '1'
& 'D:\xilinx\Vitis_HLS\2023.1\bin\vitis_hls.bat' -f run_hls.tcl
Remove-Item Env:MOTION_SAD_SYNTH
git diff --check
```

Expected: C simulation and synthesis pass, the loop achieves II 1, and `git diff --check` emits no output.

- [ ] **Step 5: Commit the documentation**

```powershell
git add fpga/hls/motion_sad/README.md
git commit -m "docs: document motion SAD HLS workflow"
```
