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

if {[info exists ::env(MOTION_SAD_STAGE_DIR)]} {
    set stage_dir $::env(MOTION_SAD_STAGE_DIR)
} else {
    set run_id [format "%s_%s" [pid] [clock milliseconds]]
    set stage_dir [file join $local_root VINGS-Mono hls \
        "motion_sad_2023_1_$run_id"]
}
puts "INFO: Local HLS stage: $stage_dir"

proc copy_atomic {source_path destination_path} {
    set temporary_path "${destination_path}.tmp_[pid]"
    file mkdir [file dirname $destination_path]
    file copy -force $source_path $temporary_path
    file rename -force $temporary_path $destination_path
}

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
set report_dir [file join $source_dir build motion_sad_prj solution1]
set csim_report [file join $report_dir csim report motion_sad_csim.log]
set synthesis_report [file join $report_dir syn report motion_sad_csynth.rpt]
file delete -force $csim_report $synthesis_report
file mkdir $build_dir
cd $build_dir
open_project -reset motion_sad_prj
set_top motion_sad
set compile_flags "-I[file join $stage_dir include] -std=c++14"
add_files [file join $stage_dir src motion_sad.cpp] \
    -cflags $compile_flags
add_files -tb [file join $stage_dir tb test_motion_sad.cpp] \
    -cflags $compile_flags
open_solution -reset solution1
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
csim_design -clean

set project_dir [file join $build_dir motion_sad_prj]
copy_atomic \
    [file join $project_dir solution1 csim report motion_sad_csim.log] \
    $csim_report

if {$run_synthesis} {
    csynth_design
    copy_atomic \
        [file join $project_dir solution1 syn report motion_sad_csynth.rpt] \
        $synthesis_report
}
exit
