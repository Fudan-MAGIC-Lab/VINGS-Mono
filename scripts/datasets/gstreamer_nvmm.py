import time
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


def _add_system_python_paths():
    for candidate in ("/usr/lib/python3/dist-packages",):
        if Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)


def _import_gst():
    _add_system_python_paths()
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def _import_gst_allocators():
    _add_system_python_paths()
    import gi

    gi.require_version("GstAllocators", "1.0")
    from gi.repository import GstAllocators

    return GstAllocators


def _import_vpi_ofa_cpp():
    import importlib

    candidate_paths = [
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[1] / "frontend",
    ]
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return importlib.import_module("vpi_ofa_gate_cpp")


def extract_dmabuf_fd(buffer, gst_allocators=None):
    if gst_allocators is None:
        try:
            gst_allocators = _import_gst_allocators()
        except Exception:
            return None
    for index in range(buffer.n_memory()):
        memory = buffer.peek_memory(index)
        if gst_allocators.is_dmabuf_memory(memory):
            return int(gst_allocators.dmabuf_memory_get_fd(memory))
    return None


@dataclass
class GStreamerNVMMFrame:
    rgb: np.ndarray
    dmabuf_fd: int = None
    width: int = None
    height: int = None
    format: str = None


class GstAppSinkFrameSource:
    def __init__(self, pipeline, timeout_s=2.0):
        self.Gst = _import_gst()
        self.pipeline = self.Gst.parse_launch(pipeline)
        self.rgb_sink = self.pipeline.get_by_name("appsink_rgb")
        self.nvmm_sink = self.pipeline.get_by_name("appsink_nvmm")
        self.appsink = self.pipeline.get_by_name("appsink") or self.pipeline.get_by_name("sink")
        if self.rgb_sink is None and self.nvmm_sink is None and self.appsink is None:
            raise ValueError(
                "GStreamer pipeline must contain appsink_rgb/appsink_nvmm, or a single appsink named 'appsink'/'sink'"
            )
        self.timeout_ns = int(float(timeout_s) * self.Gst.SECOND)
        self.pipeline.set_state(self.Gst.State.PLAYING)

    def close(self):
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def pull_frame(self):
        if self.rgb_sink is not None and self.nvmm_sink is not None:
            nvmm_sample = self._pull_sample(self.nvmm_sink)
            rgb_sample = self._pull_sample(self.rgb_sink)
            try:
                return self._dual_samples_to_frame(nvmm_sample, rgb_sample)
            finally:
                nvmm_sample = None
                rgb_sample = None
        sample = self._pull_sample(self.appsink or self.rgb_sink or self.nvmm_sink)
        try:
            return self._sample_to_frame(sample)
        finally:
            sample = None

    def _pull_sample(self, sink):
        sample = sink.emit("try-pull-sample", self.timeout_ns)
        if sample is None:
            self._raise_bus_error_or_timeout()
        return sample

    def _dual_samples_to_frame(self, nvmm_sample, rgb_sample):
        rgb_frame = self._sample_to_frame(rgb_sample)
        caps = nvmm_sample.get_caps()
        structure = caps.get_structure(0)
        buffer = nvmm_sample.get_buffer()
        rgb_frame.dmabuf_fd = extract_dmabuf_fd(buffer)
        rgb_frame.width = int(structure.get_value("width"))
        rgb_frame.height = int(structure.get_value("height"))
        rgb_frame.format = str(structure.get_value("format"))
        return rgb_frame

    def _sample_to_frame(self, sample):
        try:
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            fmt = str(structure.get_value("format"))
            buffer = sample.get_buffer()
            dmabuf_fd = extract_dmabuf_fd(buffer)
            success, map_info = buffer.map(self.Gst.MapFlags.READ)
            if not success:
                raise RuntimeError("Unable to map GStreamer buffer for tracker RGB conversion")
            try:
                rgb = self._mapped_to_rgb(map_info.data, width, height, fmt)
            finally:
                buffer.unmap(map_info)
            return GStreamerNVMMFrame(rgb=rgb, dmabuf_fd=dmabuf_fd, width=width, height=height, format=fmt)
        finally:
            pass

    def _raise_bus_error_or_timeout(self):
        bus = self.pipeline.get_bus()
        msg = bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if msg is not None and msg.type == self.Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            raise RuntimeError(f"GStreamer error: {err}; debug={debug}")
        if msg is not None and msg.type == self.Gst.MessageType.EOS:
            raise StopIteration("GStreamer pipeline reached EOS")
        raise TimeoutError("Timed out waiting for a GStreamer appsink sample")

    def _mapped_to_rgb(self, data, width, height, fmt):
        raw = np.frombuffer(data, dtype=np.uint8)
        if fmt == "RGB":
            return raw[: height * width * 3].reshape(height, width, 3).copy()
        if fmt == "BGR":
            bgr = raw[: height * width * 3].reshape(height, width, 3)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if fmt in ("RGBA", "RGBx"):
            rgba = raw[: height * width * 4].reshape(height, width, 4)
            return rgba[:, :, :3].copy()
        if fmt in ("BGRA", "BGRx"):
            bgra = raw[: height * width * 4].reshape(height, width, 4)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        if fmt == "NV12":
            yuv = raw[: height * width * 3 // 2].reshape(height * 3 // 2, width)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
        raise ValueError(f"Unsupported GStreamer appsink format for tracker RGB conversion: {fmt}")


class CppGstAppSinkFrameSource:
    def __init__(self, pipeline, timeout_s=2.0):
        module = _import_vpi_ofa_cpp()
        self.source = module.GstNvmmDualAppSinkSource(pipeline, float(timeout_s))

    def close(self):
        self.source.close()

    def pull_frame(self):
        frame = self.source.pull_frame()
        return GStreamerNVMMFrame(
            rgb=np.asarray(frame["rgb"], dtype=np.uint8),
            dmabuf_fd=int(frame["dmabuf_fd"]),
            width=int(frame["nvmm_width"]),
            height=int(frame["nvmm_height"]),
            format=str(frame["nvmm_format"]),
        )


def make_frame_source(dataset_cfg):
    pipeline = dataset_cfg["pipeline"]
    timeout_s = float(dataset_cfg.get("timeout_s", 2.0))
    use_cpp_source = bool(dataset_cfg.get("use_cpp_source", True))
    if use_cpp_source and "appsink_nvmm" in pipeline and "appsink_rgb" in pipeline:
        try:
            return CppGstAppSinkFrameSource(pipeline, timeout_s=timeout_s)
        except Exception as exc:
            if not bool(dataset_cfg.get("allow_python_fallback", True)):
                raise
            print(f"[gstreamer_nvmm] C++ source unavailable, falling back to Python appsink: {exc}")
    return GstAppSinkFrameSource(pipeline, timeout_s=timeout_s)


class GStreamerNVMMImageStream:
    def __init__(self, cfg, frame_source=None):
        self.cfg = cfg
        dataset_cfg = cfg.get("dataset", {})
        self.h_resize = int(cfg["frontend"]["image_size"][0])
        self.w_resize = int(cfg["frontend"]["image_size"][1])
        self.length = int(dataset_cfg.get("length", dataset_cfg.get("max_frames", 10**9)))
        self.timestamp_start = float(dataset_cfg.get("timestamp_start", 0.0))
        self.timestamp_step = float(dataset_cfg.get("timestamp_step", 1.0))
        self.device = cfg.get("device", {}).get("tracker", "cuda:0")
        self.c2i = np.eye(4)
        self.intrinsic = None
        self.frame_source = frame_source or make_frame_source(dataset_cfg)

    def __len__(self):
        return self.length

    def preload_camtimestamp(self):
        timestamps = [self.timestamp_start + i * self.timestamp_step for i in range(self.length)]
        return np.asarray(timestamps, dtype=np.float64).reshape(-1, 1)

    def preload_imu(self):
        imu = np.zeros((self.length, 7), dtype=np.float64)
        imu[:, 0] = self.preload_camtimestamp()[:, 0]
        return imu

    def __getitem__(self, idx):
        frame = self.frame_source.pull_frame()
        rgb = cv2.resize(frame.rgb, (self.w_resize, self.h_resize), interpolation=cv2.INTER_AREA)
        tensor = torch.tensor(rgb, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).to(self.device)
        intrinsic = self._intrinsic_tensor(frame)
        packet = {
            "timestamp": self.timestamp_start + idx * self.timestamp_step,
            "rgb": tensor,
            "intrinsic": intrinsic,
        }
        if frame.dmabuf_fd is not None:
            packet["motion_gate_dmabuf_fd"] = int(frame.dmabuf_fd)
            packet["motion_gate_nvmm_width"] = int(frame.width)
            packet["motion_gate_nvmm_height"] = int(frame.height)
            packet["motion_gate_nvmm_format"] = frame.format
        return packet

    def _intrinsic_tensor(self, frame):
        intrinsic_cfg = self.cfg["intrinsic"]
        src_h = int(frame.height or intrinsic_cfg["H"])
        src_w = int(frame.width or intrinsic_cfg["W"])
        u_scale = self.h_resize / float(src_h)
        v_scale = self.w_resize / float(src_w)
        values = [
            intrinsic_cfg["fv"] * v_scale,
            intrinsic_cfg["fu"] * u_scale,
            intrinsic_cfg["cv"] * v_scale,
            intrinsic_cfg["cu"] * u_scale,
        ]
        return torch.tensor(values, dtype=torch.float32, device=self.device)


def get_dataset(config):
    return GStreamerNVMMImageStream(config)
