import torch
import cv2
import numpy as np
import sys
import gc
from contextlib import nullcontext
from pathlib import Path


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    candidates = []
    for parent in current.parents:
        candidates.append(parent)
        if parent.name == '.worktrees' and parent.parent not in candidates:
            candidates.append(parent.parent)

    for root in candidates:
        if (root / 'submodules' / 'metric_modules' / '__init__.py').is_file():
            return root

    raise FileNotFoundError("Unable to locate repository root containing submodules/metric_modules")


REPO_ROOT = _find_repo_root()
SUBMODULES_ROOT = REPO_ROOT / 'submodules'
if str(SUBMODULES_ROOT) not in sys.path:
    sys.path.append(str(SUBMODULES_ROOT))
from metric_modules import Metric
try:
    from .metric3d_backends import Metric3DPyTorchBackend, Metric3DTensorRTBackend
except ImportError:
    from scripts.metric.metric3d_backends import (
        Metric3DPyTorchBackend,
        Metric3DTensorRTBackend,
    )
# from metric.metric3d import Metric3D_Model

class Metric_Model:
    def __init__(self, cfg, u_scale=None, v_scale=None):
        self.cfg = cfg
        device = self.cfg['device']['tracker']
        self.device = device
        self.depth_scale = min(1.0, max(0.1, float(self.cfg.get('metric_depth_scale', 1.0))))
        ckpt_path = REPO_ROOT / 'ckpts' / 'metric_depth_vit_small_800k.pth'
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Metric checkpoint not found: {ckpt_path}. "
                "Download metric_depth_vit_small_800k.pth into ckpts/ before enabling metric."
            )
        self.checkpoint = ckpt_path
        if u_scale is None:
            # u_scale, v_scale = self.cfg['frontend']['image_size'][0]/self.cfg['intrinsic']['H'], self.cfg['frontend']['image_size'][1]/self.cfg['intrinsic']['W']
            u_scale, v_scale = 1.0, 1.0
            # self.intr  = np.array([cfg['intrinsic']['fv'], cfg['intrinsic']['fu'], cfg['intrinsic']['cv'], cfg['intrinsic']['cu']])
            self.intr  = np.array([cfg['intrinsic']['fv']*v_scale, cfg['intrinsic']['fu']*u_scale, cfg['intrinsic']['cv']*v_scale, cfg['intrinsic']['cu']*u_scale])
        else:
            self.intr  = np.array([cfg['intrinsic']['fv']*v_scale, cfg['intrinsic']['fu']*u_scale, cfg['intrinsic']['cv']*v_scale, cfg['intrinsic']['cu']*u_scale])
        self.d_max = 300.0
        inference_cfg = self.cfg.get('inference', {})
        self.requested_backend = inference_cfg.get('metric3d_backend', 'torch')
        if self.requested_backend not in {'torch', 'tensorrt'}:
            raise ValueError(
                f"Unsupported Metric3D backend: {self.requested_backend}"
            )
        self.strict = bool(inference_cfg.get('tensorrt_strict', False))
        self.engine_path = inference_cfg.get('metric3d_engine')
        self.actual_backend = None
        self.fallback_reason = None
        self._fallback_logged = False
        self.backend = None

        if self.requested_backend == 'torch':
            self.backend = self._make_pytorch_backend()
            self.actual_backend = 'torch'
            self._log_backend_status()
        else:
            try:
                if not self.engine_path:
                    raise ValueError(
                        "Metric3D TensorRT backend requires inference.metric3d_engine"
                    )
                self.backend = self._make_tensorrt_backend()
                self.actual_backend = 'tensorrt'
                self._log_backend_status()
            except Exception as exc:
                self._fallback_to_pytorch(exc)

    def _make_pytorch_backend(self):
        return Metric3DPyTorchBackend(
            checkpoint=self.checkpoint,
            depth_scale=self.depth_scale,
            metric_class=Metric,
        )

    def _make_tensorrt_backend(self):
        return Metric3DTensorRTBackend(
            engine_path=self.engine_path,
            metadata_path=None,
            checkpoint=self.checkpoint,
            depth_scale=self.depth_scale,
            device=self.device,
            metric_class=Metric,
        )

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_backend_status(self):
        print(
            "[metric_depth] "
            f"requested_backend={self.requested_backend} "
            f"actual_backend={self.actual_backend} "
            f"strict={str(self.strict).lower()}"
        )

    def _log_fallback_once(self):
        if self._fallback_logged:
            return
        self._fallback_logged = True
        print(
            "[metric_depth] "
            f"requested_backend={self.requested_backend} "
            "actual_backend=torch "
            "fallback_from=tensorrt "
            f"strict={str(self.strict).lower()} "
            f"reason={self.fallback_reason}"
        )

    def _fallback_to_pytorch(self, exc):
        if self.strict:
            raise RuntimeError(
                "Metric3D TensorRT backend failed in strict mode"
            ) from exc
        if self.backend is not None:
            self.backend.close()
        self.backend = None
        self._cleanup_cuda()
        self.fallback_reason = f"{type(exc).__name__}: {exc}"
        self.backend = self._make_pytorch_backend()
        self.actual_backend = 'torch'
        self._log_fallback_once()

    def _infer_with_fallback(self, img_numpy, prepared):
        try:
            return self.backend.infer(prepared)
        except Exception as exc:
            if self.actual_backend != 'tensorrt':
                raise
            self._fallback_to_pytorch(exc)
            prepared = self.backend.preprocess(img_numpy, self.intr)
            return self.backend.infer(prepared)

    def backend_status(self):
        return {
            'requested_backend': self.requested_backend,
            'actual_backend': self.actual_backend,
            'strict': self.strict,
            'engine_path': str(self.engine_path) if self.engine_path else None,
            'fallback_reason': self.fallback_reason,
        }

    def predict(self, img, profiler=None, frame_idx=None):
        '''
        img: (3, H, W), torch.Tensor or ndarray image
        '''
        ''' ZoeDepth.
        H, W = img.shape[:2]
        pred_depth = self.predictor.infer_pil(img[..., :3], output_type="tensor")  # as torch tensor
        pred_depth_npy = cv2.resize(pred_depth.cpu().numpy(), (W, H))  # (H, W)
        pred_depth_npy = pred_depth_npy[np.newaxis, :, :, np.newaxis]
        '''
        '''
        Metric3D, img is ndarray (H, W, 3)
        '''
        def profile(stage):
            if profiler is None:
                return nullcontext()
            return profiler.time(stage, frame_idx)

        with profile("metric_depth_preprocess"):
            if isinstance(img, torch.Tensor):
                img_numpy = img.cpu().permute(1, 2, 0).numpy()
                output_device = img.device
            else:
                img_numpy = img
                output_device = self.device
            original_h, original_w = img_numpy.shape[:2]
            prepared = self.backend.preprocess(img_numpy, self.intr)

        with profile("metric_depth_model"):
            pred_depth = self._infer_with_fallback(img_numpy, prepared)

        with profile("metric_depth_postprocess"):
            depth = self.backend.postprocess(pred_depth, d_max=self.d_max, d_min=0)
            depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            depth = torch.tensor(depth, device=output_device, dtype=torch.float32)
        if profiler is not None and hasattr(profiler, 'set_metadata'):
            profiler.set_metadata('metric_depth', self.backend_status())
        return depth
    
