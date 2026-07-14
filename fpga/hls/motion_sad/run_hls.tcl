set script_dir [file dirname [file normalize [info script]]]
set build_dir [file join $script_dir build]

file mkdir $build_dir
cd $build_dir
open_project -reset motion_sad_prj
set_top motion_sad
add_files [file join $script_dir src motion_sad.cpp] \
    -cflags "-I[file join $script_dir include] -std=c++14"
add_files -tb [file join $script_dir tb test_motion_sad.cpp] \
    -cflags "-I[file join $script_dir include] -std=c++14"
open_solution -reset solution1
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
csim_design -clean
if {$argc > 0 && [lindex $argv 0] eq "synth"} {
    csynth_design
}
exit
