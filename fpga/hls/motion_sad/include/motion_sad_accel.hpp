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
