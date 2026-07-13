import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - import availability is environment dependent
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None




def _import_vpi_ofa_cpp():
    import importlib

    candidate_paths = [
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parent,
    ]
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return importlib.import_module("vpi_ofa_gate_cpp")

def _import_vpi():
    try:
        import vpi
        return vpi
    except ModuleNotFoundError:
        candidate_paths = [
            "/opt/nvidia/vpi2/lib/aarch64-linux-gnu/python",
            "/usr/lib/python3/dist-packages",
        ]
        for candidate in candidate_paths:
            if Path(candidate).exists() and candidate not in sys.path:
                sys.path.insert(0, candidate)
                try:
                    import vpi
                    return vpi
                except ModuleNotFoundError:
                    continue
        raise


class JetsonMotionGate:
    def __init__(self, cfg):
        gate_cfg = cfg.get("jetson_motion_gate", {})
        self.enabled = bool(gate_cfg.get("enabled", False))
        self.backend = str(gate_cfg.get("backend", "auto")).lower()
        self.threshold = float(gate_cfg.get("threshold", 0.75))
        self.force_interval = max(0, int(gate_cfg.get("force_interval", 8)))
        resize = gate_cfg.get("resize", [96, 160])
        if isinstance(resize, str):
            h_str, w_str = [part.strip() for part in resize.split(",")]
            resize = [int(h_str), int(w_str)]
        self.resize = (int(resize[0]), int(resize[1]))
        self.grid_size = int(gate_cfg.get("grid_size", 4))
        self.vpi_num_levels = max(1, int(gate_cfg.get("vpi_num_levels", 1)))
        self.vpi_quality = str(gate_cfg.get("vpi_quality", "low")).lower()
        self.score_percentile = float(gate_cfg.get("score_percentile", 90.0))
        self.prev_gray = None
        self._prev_vpi_frame = None
        self._cpp_gate = None
        self._nvbuffer_gate = None
        self.frame_count = 0
        self.run_count = 0
        self.skip_count = 0
        self.vpi_disabled_reason = None

    def decide_packet(self, data_packet):
        if not isinstance(data_packet, dict):
            return self.decide(data_packet)
        if self.enabled and self.backend in ("auto", "vpi_nvbuffer") and "motion_gate_dmabuf_fd" in data_packet:
            self.frame_count += 1
            nvbuffer_result = self._compute_score_vpi_nvbuffer(data_packet)
            if nvbuffer_result is not None:
                has_score, score = nvbuffer_result
                if not has_score:
                    self.run_count += 1
                    return self._decision(False, score, "initial", "vpi_nvbuffer")
                return self._decision_from_score(score, "vpi_nvbuffer")
            return self._decide_preprocessed(data_packet["rgb"], increment_frame=False)
        return self.decide(data_packet["rgb"])

    def decide(self, image):
        if self.enabled and self.backend in ("auto", "vpi_cpp"):
            self.frame_count += 1
            cpp_result = self._compute_score_vpi_cpp_image(image)
            if cpp_result is not None:
                has_score, score = cpp_result
                if not has_score:
                    self.run_count += 1
                    return self._decision(False, score, "initial", "vpi_cpp")
                return self._decision_from_score(score, "vpi_cpp")
            return self._decide_preprocessed(image, increment_frame=False)

        return self._decide_preprocessed(image, increment_frame=True)

    def _decide_preprocessed(self, image, increment_frame):
        gray = self._prepare_gray(image)
        if increment_frame:
            self.frame_count += 1

        if self.prev_gray is None:
            self.prev_gray = gray
            if self.enabled and self.backend in ("auto", "vpi"):
                self._prev_vpi_frame = self._try_make_vpi_frame(gray)
            self.run_count += 1
            return self._decision(False, 0.0, "initial", "none")

        if not self.enabled:
            self.prev_gray = gray
            self._prev_vpi_frame = None
            self.run_count += 1
            return self._decision(False, 0.0, "disabled", "none")

        score, backend = self._compute_score(self.prev_gray, gray)
        self.prev_gray = gray
        return self._decision_from_score(score, backend)

    def _decision_from_score(self, score, backend):
        if self.force_interval > 0 and (self.frame_count - 1) % self.force_interval == 0:
            self.run_count += 1
            return self._decision(False, score, "force_interval", backend)

        if score >= self.threshold:
            self.run_count += 1
            return self._decision(False, score, "motion", backend)

        self.skip_count += 1
        return self._decision(True, score, "low_motion", backend)

    def summary(self):
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "threshold": self.threshold,
            "force_interval": self.force_interval,
            "frames": self.frame_count,
            "runs": self.run_count,
            "skips": self.skip_count,
            "vpi_disabled_reason": self.vpi_disabled_reason,
        }

    def maybe_log_summary(self):
        if not self.enabled:
            return
        summary = self.summary()
        print(
            "[jetson_motion_gate] "
            f"enabled=1 backend={summary['backend']} threshold={summary['threshold']:.3f} "
            f"force_interval={summary['force_interval']} frames={summary['frames']} "
            f"runs={summary['runs']} skips={summary['skips']} "
            f"vpi_disabled_reason={summary['vpi_disabled_reason']}"
        )

    def _decision(self, skip, score, reason, backend):
        return {
            "skip": bool(skip),
            "score": float(score),
            "reason": reason,
            "backend": backend,
        }

    def _prepare_gray(self, image):
        array = self._to_numpy(image)
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3 and array.shape[2] == 3:
            if cv2 is None:
                gray = array.mean(axis=2).astype(np.uint8)
            else:
                gray = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif array.ndim == 3 and array.shape[2] == 1:
            gray = array[:, :, 0].astype(np.uint8)
        else:
            gray = array.astype(np.uint8)
        if self.resize is not None:
            h, w = self.resize
            if cv2 is None:
                gray = gray[:: max(1, gray.shape[0] // h), :: max(1, gray.shape[1] // w)]
                gray = gray[:h, :w]
            else:
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(gray)

    def _to_numpy(self, image):
        if torch is not None and isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()
        return np.asarray(image)

    def _compute_score(self, prev_gray, gray):
        if self.backend in ("vpi_cpp", "vpi_nvbuffer"):
            return self._compute_score_opencv(prev_gray, gray), "opencv_fallback"
        if self.backend in ("auto", "vpi"):
            score = self._compute_score_vpi(gray)
            if score is not None:
                return score, "vpi"
            if self.backend == "vpi":
                return self._compute_score_opencv(prev_gray, gray), "opencv_fallback"
        return self._compute_score_opencv(prev_gray, gray), "opencv"

    def _compute_score_vpi_cpp_image(self, image):
        try:
            cpp_image = self._prepare_cpp_image(image)
            gate = self._ensure_cpp_gate()
            if not hasattr(gate, "score_image"):
                return None
            has_score, score = gate.score_image(cpp_image)
            return bool(has_score), float(score)
        except Exception as exc:
            self.vpi_disabled_reason = f"vpi_cpp {type(exc).__name__}: {exc}"
            self._cpp_gate = None
            return None

    def _compute_score_vpi_nvbuffer(self, data_packet):
        try:
            gate = self._ensure_nvbuffer_gate()
            fd = int(data_packet["motion_gate_dmabuf_fd"])
            width = int(data_packet["motion_gate_nvmm_width"])
            height = int(data_packet["motion_gate_nvmm_height"])
            fmt = str(data_packet.get("motion_gate_nvmm_format", "NV12"))
            has_score, score = gate.score_nvbuffer(fd, width, height, fmt)
            return bool(has_score), float(score)
        except Exception as exc:
            self.vpi_disabled_reason = f"vpi_nvbuffer {type(exc).__name__}: {exc}"
            self._nvbuffer_gate = None
            return None

    def _compute_score_vpi_cpp(self, gray):
        try:
            gate = self._ensure_cpp_gate()
            has_score, score = gate.score(np.ascontiguousarray(gray.astype(np.uint8, copy=False)))
            if not has_score:
                return 0.0
            return float(score)
        except Exception as exc:
            self.vpi_disabled_reason = f"vpi_cpp {type(exc).__name__}: {exc}"
            self._cpp_gate = None
            return None

    def _ensure_cpp_gate(self):
        if self._cpp_gate is None:
            module = _import_vpi_ofa_cpp()
            self._cpp_gate = module.VpiOfaMotionGate(
                int(self.resize[0]),
                int(self.resize[1]),
                int(self.grid_size),
                self.vpi_quality,
                float(self.score_percentile),
            )
        return self._cpp_gate

    def _ensure_nvbuffer_gate(self):
        if self._nvbuffer_gate is None:
            module = _import_vpi_ofa_cpp()
            self._nvbuffer_gate = module.VpiOfaNvBufferMotionGate(
                int(self.resize[0]),
                int(self.resize[1]),
                int(self.grid_size),
                self.vpi_quality,
                float(self.score_percentile),
            )
        return self._nvbuffer_gate

    def _prepare_cpp_image(self, image):
        array = self._to_numpy(image)
        if array.ndim == 4:
            array = array[0]
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))

    def _compute_score_vpi(self, gray):
        if self.vpi_disabled_reason is not None or self._prev_vpi_frame is None:
            return None
        try:
            frame = self._to_vpi_ofa_frame(gray)
            score = self._compute_score_vpi_frames(self._prev_vpi_frame, frame)
            self._prev_vpi_frame = frame
            return score
        except Exception as exc:
            self.vpi_disabled_reason = f"{type(exc).__name__}: {exc}"
            self._prev_vpi_frame = None
            return None

    def _try_make_vpi_frame(self, gray):
        if self.vpi_disabled_reason is not None:
            return None
        try:
            return self._to_vpi_ofa_frame(gray)
        except Exception as exc:
            self.vpi_disabled_reason = f"{type(exc).__name__}: {exc}"
            return None

    def _compute_score_vpi_frames(self, ref, frame):
        vpi = _import_vpi()
        quality = self._vpi_quality(vpi)
        with vpi.Backend.OFA:
            flow = vpi.optflow_dense(
                ref,
                frame,
                gridsize=self.grid_size,
                quality=quality,
            )
        return self._flow_score(np.asarray(flow.cpu()))

    def _vpi_quality(self, vpi):
        if self.vpi_quality == "medium":
            return vpi.OptFlowQuality.MEDIUM
        if self.vpi_quality == "high":
            return vpi.OptFlowQuality.HIGH
        return vpi.OptFlowQuality.LOW

    def _to_vpi_ofa_frame(self, gray):
        vpi = _import_vpi()
        gray = np.ascontiguousarray(gray.astype(np.uint8))
        image = vpi.asimage(gray, vpi.Format.Y8_ER)
        if self.vpi_num_levels <= 1:
            return image.convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC)
        return image.gaussian_pyramid(self.vpi_num_levels, backend=vpi.Backend.CUDA).convert(
            vpi.Format.Y8_ER_BL,
            backend=vpi.Backend.VIC,
        )

    def _compute_score_opencv(self, prev_gray, gray):
        if cv2 is None:
            return float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        return self._flow_score(flow)

    def _flow_score(self, flow):
        flow = np.asarray(flow)
        if flow.ndim < 3 or flow.shape[-1] < 2:
            return 0.0
        if np.issubdtype(flow.dtype, np.integer):
            flow = flow.astype(np.float32) / float(1 << 5)
        magnitude = np.linalg.norm(flow[..., :2], axis=-1)
        if magnitude.size == 0:
            return 0.0
        return float(np.percentile(magnitude, self.score_percentile))
