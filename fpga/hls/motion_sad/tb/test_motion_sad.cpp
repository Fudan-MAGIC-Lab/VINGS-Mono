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
