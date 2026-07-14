#include "motion_sad.hpp"

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
