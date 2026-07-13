#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <nvbufsurface.h>

#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/OpticalFlowDense.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/Rescale.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static void check_status(VPIStatus status, int line) {
    if (status == VPI_SUCCESS) {
        return;
    }
    char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
    vpiGetLastStatusMessage(buffer, sizeof(buffer));
    std::ostringstream ss;
    ss << "line " << line << ": " << vpiStatusGetName(status) << ": " << buffer;
    throw std::runtime_error(ss.str());
}

#define CHECK_VPI(STMT) check_status((STMT), __LINE__)

static VPIOpticalFlowQuality parse_quality(const std::string &quality) {
    if (quality == "medium") {
        return VPI_OPTICAL_FLOW_QUALITY_MEDIUM;
    }
    if (quality == "high") {
        return VPI_OPTICAL_FLOW_QUALITY_HIGH;
    }
    return VPI_OPTICAL_FLOW_QUALITY_LOW;
}

class VpiOfaMotionGate {
  public:
    VpiOfaMotionGate(int height, int width, int grid_size, const std::string &quality, double percentile)
        : height_(height), width_(width), grid_size_(grid_size), quality_(parse_quality(quality)), percentile_(percentile) {
        if (height_ <= 0 || width_ <= 0) {
            throw std::runtime_error("height and width must be positive");
        }
        if (!(grid_size_ == 1 || grid_size_ == 2 || grid_size_ == 4 || grid_size_ == 8)) {
            throw std::runtime_error("OFA grid_size must be one of 1, 2, 4, 8");
        }
        create_resources();
    }

    ~VpiOfaMotionGate() { destroy_resources(); }

    py::tuple score(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> gray) {
        py::buffer_info info = gray.request();
        if (info.ndim != 2) {
            throw std::runtime_error("VpiOfaMotionGate.score expects a 2D uint8 grayscale array");
        }
        cv::Mat cv_gray(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC1, info.ptr);
        prepare_resized_gray(cv_gray);
        return submit_current_gray();
    }

    py::tuple score_image(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image) {
        py::buffer_info info = image.request();
        prepare_image(info);
        return submit_current_gray();
    }

    void reset() { has_prev_ = false; }

  private:
    int height_;
    int width_;
    int grid_size_;
    VPIOpticalFlowQuality quality_;
    double percentile_;
    bool has_prev_ = false;

    VPIStream stream_ = nullptr;
    VPIImage input_pl_ = nullptr;
    VPIImage prev_bl_ = nullptr;
    VPIImage cur_bl_ = nullptr;
    VPIImage motion_bl_ = nullptr;
    VPIPayload payload_ = nullptr;
    cv::Mat wrapper_storage_;
    cv::Mat source_gray_;
    cv::Mat resized_gray_;

    void create_resources() {
        wrapper_storage_ = cv::Mat::zeros(height_, width_, CV_8UC1);
        resized_gray_.create(height_, width_, CV_8UC1);
        CHECK_VPI(vpiStreamCreate(VPI_BACKEND_OFA | VPI_BACKEND_VIC, &stream_));
        CHECK_VPI(vpiImageCreateWrapperOpenCVMat(wrapper_storage_, VPI_IMAGE_FORMAT_Y8_ER, 0, &input_pl_));
        CHECK_VPI(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &prev_bl_));
        CHECK_VPI(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &cur_bl_));
        int mv_width = (width_ + grid_size_ - 1) / grid_size_;
        int mv_height = (height_ + grid_size_ - 1) / grid_size_;
        CHECK_VPI(vpiImageCreate(mv_width, mv_height, VPI_IMAGE_FORMAT_2S16_BL, 0, &motion_bl_));
        int32_t grid = grid_size_;
        CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, width_, height_, VPI_IMAGE_FORMAT_Y8_ER_BL, &grid, 1,
                                            quality_, &payload_));
    }

    void destroy_resources() {
        if (payload_ != nullptr) {
            vpiPayloadDestroy(payload_);
            payload_ = nullptr;
        }
        if (motion_bl_ != nullptr) {
            vpiImageDestroy(motion_bl_);
            motion_bl_ = nullptr;
        }
        if (cur_bl_ != nullptr) {
            vpiImageDestroy(cur_bl_);
            cur_bl_ = nullptr;
        }
        if (prev_bl_ != nullptr) {
            vpiImageDestroy(prev_bl_);
            prev_bl_ = nullptr;
        }
        if (input_pl_ != nullptr) {
            vpiImageDestroy(input_pl_);
            input_pl_ = nullptr;
        }
        if (stream_ != nullptr) {
            vpiStreamDestroy(stream_);
            stream_ = nullptr;
        }
        has_prev_ = false;
    }

    void prepare_image(const py::buffer_info &info) {
        if (info.ndim == 2) {
            cv::Mat gray(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC1, info.ptr);
            prepare_resized_gray(gray);
            return;
        }
        if (info.ndim != 3) {
            throw std::runtime_error("VpiOfaMotionGate.score_image expects HW, HWC, or CHW uint8 input");
        }

        const auto d0 = static_cast<int>(info.shape[0]);
        const auto d1 = static_cast<int>(info.shape[1]);
        const auto d2 = static_cast<int>(info.shape[2]);
        const uint8_t *ptr = static_cast<const uint8_t *>(info.ptr);

        if ((d0 == 1 || d0 == 3) && !(d2 == 1 || d2 == 3)) {
            prepare_chw(ptr, d0, d1, d2);
            return;
        }
        if (d2 == 1 || d2 == 3) {
            prepare_hwc(ptr, d0, d1, d2);
            return;
        }
        if (d0 == 1 || d0 == 3) {
            prepare_chw(ptr, d0, d1, d2);
            return;
        }
        throw std::runtime_error("VpiOfaMotionGate.score_image cannot infer image layout");
    }

    void prepare_hwc(const uint8_t *ptr, int height, int width, int channels) {
        if (channels == 1) {
            cv::Mat gray(height, width, CV_8UC1, const_cast<uint8_t *>(ptr));
            prepare_resized_gray(gray);
            return;
        }
        cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t *>(ptr));
        cv::cvtColor(rgb, source_gray_, cv::COLOR_RGB2GRAY);
        prepare_resized_gray(source_gray_);
    }

    void prepare_chw(const uint8_t *ptr, int channels, int height, int width) {
        if (channels == 1) {
            cv::Mat gray(height, width, CV_8UC1, const_cast<uint8_t *>(ptr));
            prepare_resized_gray(gray);
            return;
        }
        source_gray_.create(height, width, CV_8UC1);
        const size_t plane_size = static_cast<size_t>(height) * static_cast<size_t>(width);
        const uint8_t *r = ptr;
        const uint8_t *g = ptr + plane_size;
        const uint8_t *b = ptr + (plane_size * 2);
        uint8_t *dst = source_gray_.ptr<uint8_t>(0);
        for (size_t i = 0; i < plane_size; ++i) {
            dst[i] = static_cast<uint8_t>((77u * r[i] + 150u * g[i] + 29u * b[i] + 128u) >> 8);
        }
        prepare_resized_gray(source_gray_);
    }

    void prepare_resized_gray(const cv::Mat &gray) {
        if (gray.empty()) {
            throw std::runtime_error("VpiOfaMotionGate received an empty image");
        }
        if (gray.type() != CV_8UC1) {
            throw std::runtime_error("VpiOfaMotionGate expects uint8 grayscale data after preprocessing");
        }
        resized_gray_.create(height_, width_, CV_8UC1);
        if (gray.rows == height_ && gray.cols == width_) {
            gray.copyTo(resized_gray_);
            return;
        }
        cv::resize(gray, resized_gray_, cv::Size(width_, height_), 0.0, 0.0, cv::INTER_AREA);
    }

    py::tuple submit_current_gray() {
        CHECK_VPI(vpiImageSetWrappedOpenCVMat(input_pl_, resized_gray_));

        VPIImage target_bl = has_prev_ ? cur_bl_ : prev_bl_;
        CHECK_VPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, input_pl_, target_bl, nullptr));

        if (!has_prev_) {
            CHECK_VPI(vpiStreamSync(stream_));
            has_prev_ = true;
            return py::make_tuple(false, 0.0);
        }

        CHECK_VPI(vpiSubmitOpticalFlowDense(stream_, VPI_BACKEND_OFA, payload_, prev_bl_, cur_bl_, motion_bl_));
        CHECK_VPI(vpiStreamSync(stream_));

        double result = compute_percentile();
        std::swap(prev_bl_, cur_bl_);
        return py::make_tuple(true, result);
    }

    double compute_percentile() {
        VPIImageData data;
        CHECK_VPI(vpiImageLockData(motion_bl_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
        cv::Mat mv_image;
        try {
            CHECK_VPI(vpiImageDataExportOpenCVMat(data, &mv_image));
            std::vector<float> magnitudes;
            magnitudes.reserve(static_cast<size_t>(mv_image.rows * mv_image.cols));
            for (int y = 0; y < mv_image.rows; ++y) {
                const cv::Vec<int16_t, 2> *row = mv_image.ptr<cv::Vec<int16_t, 2>>(y);
                for (int x = 0; x < mv_image.cols; ++x) {
                    float fx = static_cast<float>(row[x][0]) / 32.0f;
                    float fy = static_cast<float>(row[x][1]) / 32.0f;
                    magnitudes.push_back(std::sqrt(fx * fx + fy * fy));
                }
            }
            CHECK_VPI(vpiImageUnlock(motion_bl_));
            if (magnitudes.empty()) {
                return 0.0;
            }
            double clamped = std::max(0.0, std::min(100.0, percentile_));
            size_t index = static_cast<size_t>(std::round((clamped / 100.0) * (magnitudes.size() - 1)));
            std::nth_element(magnitudes.begin(), magnitudes.begin() + index, magnitudes.end());
            return static_cast<double>(magnitudes[index]);
        } catch (...) {
            vpiImageUnlock(motion_bl_);
            throw;
        }
    }
};

class VpiOfaNvBufferMotionGate {
  public:
    VpiOfaNvBufferMotionGate(int height, int width, int grid_size, const std::string &quality, double percentile)
        : height_(height), width_(width), grid_size_(grid_size), quality_(parse_quality(quality)), percentile_(percentile) {
        if (height_ <= 0 || width_ <= 0) {
            throw std::runtime_error("height and width must be positive");
        }
        if (!(grid_size_ == 1 || grid_size_ == 2 || grid_size_ == 4 || grid_size_ == 8)) {
            throw std::runtime_error("OFA grid_size must be one of 1, 2, 4, 8");
        }
        create_gate_resources();
    }

    ~VpiOfaNvBufferMotionGate() { destroy_resources(); }

    py::tuple score_nvbuffer(int fd, int width, int height, const std::string &format) {
        if (fd < 0) {
            throw std::runtime_error("NvBuffer fd must be non-negative");
        }
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("NvBuffer width and height must be positive");
        }
        if (!(format == "NV12" || format == "NV12_ER" || format == "NV12_BL" || format == "NV12_ER_BL")) {
            throw std::runtime_error("VpiOfaNvBufferMotionGate currently expects NV12/NV12_ER NvBuffer input");
        }
        ensure_source_resources(width, height);
        wrap_nvbuffer(fd);

        CHECK_VPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, nvbuffer_, source_y8_, nullptr));
        CHECK_VPI(vpiSubmitRescale(stream_, VPI_BACKEND_VIC, source_y8_, resized_y8_, VPI_INTERP_LINEAR,
                                   VPI_BORDER_CLAMP, 0));

        VPIImage target_bl = has_prev_ ? cur_bl_ : prev_bl_;
        CHECK_VPI(vpiSubmitConvertImageFormat(stream_, VPI_BACKEND_VIC, resized_y8_, target_bl, nullptr));

        if (!has_prev_) {
            CHECK_VPI(vpiStreamSync(stream_));
            has_prev_ = true;
            return py::make_tuple(false, 0.0);
        }

        CHECK_VPI(vpiSubmitOpticalFlowDense(stream_, VPI_BACKEND_OFA, payload_, prev_bl_, cur_bl_, motion_bl_));
        CHECK_VPI(vpiStreamSync(stream_));

        double result = compute_percentile();
        std::swap(prev_bl_, cur_bl_);
        return py::make_tuple(true, result);
    }

    void reset() { has_prev_ = false; }

  private:
    int height_;
    int width_;
    int grid_size_;
    VPIOpticalFlowQuality quality_;
    double percentile_;
    bool has_prev_ = false;
    int source_width_ = 0;
    int source_height_ = 0;

    VPIStream stream_ = nullptr;
    VPIImage nvbuffer_ = nullptr;
    VPIImage source_y8_ = nullptr;
    VPIImage resized_y8_ = nullptr;
    VPIImage prev_bl_ = nullptr;
    VPIImage cur_bl_ = nullptr;
    VPIImage motion_bl_ = nullptr;
    VPIPayload payload_ = nullptr;

    void create_gate_resources() {
        CHECK_VPI(vpiStreamCreate(VPI_BACKEND_OFA | VPI_BACKEND_VIC, &stream_));
        CHECK_VPI(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_Y8_ER, 0, &resized_y8_));
        CHECK_VPI(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &prev_bl_));
        CHECK_VPI(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &cur_bl_));
        int mv_width = (width_ + grid_size_ - 1) / grid_size_;
        int mv_height = (height_ + grid_size_ - 1) / grid_size_;
        CHECK_VPI(vpiImageCreate(mv_width, mv_height, VPI_IMAGE_FORMAT_2S16_BL, 0, &motion_bl_));
        int32_t grid = grid_size_;
        CHECK_VPI(vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, width_, height_, VPI_IMAGE_FORMAT_Y8_ER_BL, &grid, 1,
                                            quality_, &payload_));
    }

    void ensure_source_resources(int width, int height) {
        if (source_y8_ != nullptr && source_width_ == width && source_height_ == height) {
            return;
        }
        if (source_y8_ != nullptr) {
            vpiImageDestroy(source_y8_);
            source_y8_ = nullptr;
        }
        if (nvbuffer_ != nullptr) {
            vpiImageDestroy(nvbuffer_);
            nvbuffer_ = nullptr;
        }
        source_width_ = width;
        source_height_ = height;
        CHECK_VPI(vpiImageCreate(source_width_, source_height_, VPI_IMAGE_FORMAT_Y8_ER, 0, &source_y8_));
        has_prev_ = false;
    }

    void wrap_nvbuffer(int fd) {
        VPIImageData data = {};
        data.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
        data.buffer.fd = fd;
        if (nvbuffer_ == nullptr) {
            CHECK_VPI(vpiImageCreateWrapper(&data, nullptr, VPI_BACKEND_VIC, &nvbuffer_));
            return;
        }
        CHECK_VPI(vpiImageSetWrapper(nvbuffer_, &data));
    }

    void destroy_resources() {
        if (payload_ != nullptr) {
            vpiPayloadDestroy(payload_);
            payload_ = nullptr;
        }
        if (motion_bl_ != nullptr) {
            vpiImageDestroy(motion_bl_);
            motion_bl_ = nullptr;
        }
        if (cur_bl_ != nullptr) {
            vpiImageDestroy(cur_bl_);
            cur_bl_ = nullptr;
        }
        if (prev_bl_ != nullptr) {
            vpiImageDestroy(prev_bl_);
            prev_bl_ = nullptr;
        }
        if (resized_y8_ != nullptr) {
            vpiImageDestroy(resized_y8_);
            resized_y8_ = nullptr;
        }
        if (source_y8_ != nullptr) {
            vpiImageDestroy(source_y8_);
            source_y8_ = nullptr;
        }
        if (nvbuffer_ != nullptr) {
            vpiImageDestroy(nvbuffer_);
            nvbuffer_ = nullptr;
        }
        if (stream_ != nullptr) {
            vpiStreamDestroy(stream_);
            stream_ = nullptr;
        }
        has_prev_ = false;
    }

    double compute_percentile() {
        VPIImageData data;
        CHECK_VPI(vpiImageLockData(motion_bl_, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data));
        cv::Mat mv_image;
        try {
            CHECK_VPI(vpiImageDataExportOpenCVMat(data, &mv_image));
            std::vector<float> magnitudes;
            magnitudes.reserve(static_cast<size_t>(mv_image.rows * mv_image.cols));
            for (int y = 0; y < mv_image.rows; ++y) {
                const cv::Vec<int16_t, 2> *row = mv_image.ptr<cv::Vec<int16_t, 2>>(y);
                for (int x = 0; x < mv_image.cols; ++x) {
                    float fx = static_cast<float>(row[x][0]) / 32.0f;
                    float fy = static_cast<float>(row[x][1]) / 32.0f;
                    magnitudes.push_back(std::sqrt(fx * fx + fy * fy));
                }
            }
            CHECK_VPI(vpiImageUnlock(motion_bl_));
            if (magnitudes.empty()) {
                return 0.0;
            }
            double clamped = std::max(0.0, std::min(100.0, percentile_));
            size_t index = static_cast<size_t>(std::round((clamped / 100.0) * (magnitudes.size() - 1)));
            std::nth_element(magnitudes.begin(), magnitudes.begin() + index, magnitudes.end());
            return static_cast<double>(magnitudes[index]);
        } catch (...) {
            vpiImageUnlock(motion_bl_);
            throw;
        }
    }
};

class GstNvmmDualAppSinkSource {
  public:
    GstNvmmDualAppSinkSource(const std::string &pipeline, double timeout_s) {
        gst_init(nullptr, nullptr);
        timeout_ns_ = static_cast<GstClockTime>(timeout_s * static_cast<double>(GST_SECOND));
        GError *error = nullptr;
        pipeline_ = gst_parse_launch(pipeline.c_str(), &error);
        if (pipeline_ == nullptr) {
            std::string message = error != nullptr ? error->message : "unknown parse error";
            if (error != nullptr) {
                g_error_free(error);
            }
            throw std::runtime_error("Failed to parse GStreamer pipeline: " + message);
        }
        rgb_sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink_rgb");
        nvmm_sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink_nvmm");
        if (rgb_sink_ == nullptr || nvmm_sink_ == nullptr) {
            throw std::runtime_error("Pipeline must contain appsink_rgb and appsink_nvmm");
        }
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    }

    ~GstNvmmDualAppSinkSource() { close(); }

    void close() {
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }
        if (rgb_sink_ != nullptr) {
            gst_object_unref(rgb_sink_);
            rgb_sink_ = nullptr;
        }
        if (nvmm_sink_ != nullptr) {
            gst_object_unref(nvmm_sink_);
            nvmm_sink_ = nullptr;
        }
        if (pipeline_ != nullptr) {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    py::dict pull_frame() {
        GstSample *nvmm_sample = pull_sample(nvmm_sink_);
        GstSample *rgb_sample = pull_sample(rgb_sink_);
        try {
            py::dict result = rgb_sample_to_dict(rgb_sample);
            append_nvmm_metadata(nvmm_sample, result);
            gst_sample_unref(nvmm_sample);
            gst_sample_unref(rgb_sample);
            return result;
        } catch (...) {
            gst_sample_unref(nvmm_sample);
            gst_sample_unref(rgb_sample);
            throw;
        }
    }

  private:
    GstElement *pipeline_ = nullptr;
    GstElement *rgb_sink_ = nullptr;
    GstElement *nvmm_sink_ = nullptr;
    GstClockTime timeout_ns_ = 2 * GST_SECOND;

    GstSample *pull_sample(GstElement *sink) {
        GstSample *sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), timeout_ns_);
        if (sample != nullptr) {
            return sample;
        }
        GstBus *bus = gst_element_get_bus(pipeline_);
        GstMessage *msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        gst_object_unref(bus);
        if (msg != nullptr) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError *err = nullptr;
                gchar *debug = nullptr;
                gst_message_parse_error(msg, &err, &debug);
                std::string message = err != nullptr ? err->message : "unknown error";
                if (debug != nullptr) {
                    message += "; debug=";
                    message += debug;
                }
                if (err != nullptr) {
                    g_error_free(err);
                }
                if (debug != nullptr) {
                    g_free(debug);
                }
                gst_message_unref(msg);
                throw std::runtime_error("GStreamer error: " + message);
            }
            gst_message_unref(msg);
            throw std::runtime_error("GStreamer pipeline reached EOS");
        }
        throw std::runtime_error("Timed out waiting for GStreamer appsink sample");
    }

    py::dict rgb_sample_to_dict(GstSample *sample) {
        GstCaps *caps = gst_sample_get_caps(sample);
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        int width = 0;
        int height = 0;
        const char *format = gst_structure_get_string(structure, "format");
        gst_structure_get_int(structure, "width", &width);
        gst_structure_get_int(structure, "height", &height);
        if (format == nullptr || std::string(format) != "RGB") {
            throw std::runtime_error("appsink_rgb must output video/x-raw,format=RGB");
        }
        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            throw std::runtime_error("Unable to map appsink_rgb buffer");
        }
        try {
            const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
            if (map.size < expected) {
                throw std::runtime_error("appsink_rgb buffer is smaller than expected RGB payload");
            }
            py::array_t<uint8_t> rgb({height, width, 3});
            py::buffer_info out = rgb.request();
            std::memcpy(out.ptr, map.data, expected);
            gst_buffer_unmap(buffer, &map);
            py::dict result;
            result["rgb"] = rgb;
            result["width"] = width;
            result["height"] = height;
            result["format"] = "RGB";
            return result;
        } catch (...) {
            gst_buffer_unmap(buffer, &map);
            throw;
        }
    }

    void append_nvmm_metadata(GstSample *sample, py::dict &result) {
        GstCaps *caps = gst_sample_get_caps(sample);
        GstStructure *structure = gst_caps_get_structure(caps, 0);
        int width = 0;
        int height = 0;
        const char *format = gst_structure_get_string(structure, "format");
        gst_structure_get_int(structure, "width", &width);
        gst_structure_get_int(structure, "height", &height);
        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            throw std::runtime_error("Unable to map appsink_nvmm buffer as NvBufSurface");
        }
        try {
            if (map.size < sizeof(NvBufSurface)) {
                throw std::runtime_error("appsink_nvmm buffer is too small to contain NvBufSurface metadata");
            }
            auto *surface = reinterpret_cast<NvBufSurface *>(map.data);
            if (surface->numFilled < 1 || surface->surfaceList == nullptr) {
                throw std::runtime_error("appsink_nvmm NvBufSurface has no filled surfaces");
            }
            int fd = static_cast<int>(surface->surfaceList[0].bufferDesc);
            gst_buffer_unmap(buffer, &map);
            result["dmabuf_fd"] = fd;
            result["nvmm_width"] = width;
            result["nvmm_height"] = height;
            result["nvmm_format"] = format != nullptr ? format : "NV12";
        } catch (...) {
            gst_buffer_unmap(buffer, &map);
            throw;
        }
    }
};

PYBIND11_MODULE(vpi_ofa_gate_cpp, m) {
    py::class_<VpiOfaMotionGate>(m, "VpiOfaMotionGate")
        .def(py::init<int, int, int, const std::string &, double>())
        .def("score", &VpiOfaMotionGate::score)
        .def("score_image", &VpiOfaMotionGate::score_image)
        .def("reset", &VpiOfaMotionGate::reset);
    py::class_<VpiOfaNvBufferMotionGate>(m, "VpiOfaNvBufferMotionGate")
        .def(py::init<int, int, int, const std::string &, double>())
        .def("score_nvbuffer", &VpiOfaNvBufferMotionGate::score_nvbuffer)
        .def("reset", &VpiOfaNvBufferMotionGate::reset);
    py::class_<GstNvmmDualAppSinkSource>(m, "GstNvmmDualAppSinkSource")
        .def(py::init<const std::string &, double>())
        .def("pull_frame", &GstNvmmDualAppSinkSource::pull_frame)
        .def("close", &GstNvmmDualAppSinkSource::close);
}
