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
