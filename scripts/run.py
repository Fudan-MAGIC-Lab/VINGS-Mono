import argparse
import importlib
import os
import shutil
import sys

import numpy as np
import torch
from lietorch import SE3

from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from runtime_budget import RuntimeBudgetController
from profiling.runtime_profiler import RuntimeProfiler
from metric_depth_schedule import MetricDepthScheduler


parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
parser.add_argument("--prefix", default='')
parser.add_argument("--dataset-root", default=None, help="Override dataset.root in the config")
parser.add_argument("--output-dir", default=None, help="Override output.save_dir in the config")
parser.add_argument("--frontend-weight", default=None, help="Override frontend.weight in the config")
parser.add_argument("--device-tracker", default=None, help="Override device.tracker in the config")
parser.add_argument("--device-mapper", default=None, help="Override device.mapper in the config")
parser.add_argument("--frontend-buffer", type=int, default=None, help="Override frontend.buffer to control memory usage")
parser.add_argument("--frontend-save-buffer", type=int, default=None, help="Override frontend CPU save buffer size")
parser.add_argument("--frontend-image-size", default=None, help="Override frontend.image_size as H,W (e.g. 256,448)")
parser.add_argument("--frontend-iters1", type=int, default=None, help="Override frontend non-keyframe graph update iterations")
parser.add_argument("--frontend-iters2", type=int, default=None, help="Override frontend keyframe graph update iterations")
parser.add_argument("--frontend-adaptive-iters", action="store_true", help="Enable motion-aware frontend graph update iterations")
parser.add_argument("--frontend-iters1-high", type=int, default=None, help="High-motion frontend non-keyframe graph update iterations")
parser.add_argument("--frontend-iters2-high", type=int, default=None, help="High-motion frontend keyframe graph update iterations")
parser.add_argument("--frontend-iters-high-motion-ratio", type=float, default=None, help="Use high frontend iterations when motion exceeds this multiple of the motion threshold")
parser.add_argument("--frontend-iters-force-interval", type=int, default=None, help="Force high frontend iterations every N update calls when adaptive iterations are enabled")
parser.add_argument("--lightglue-weight-dir", default=None, help="Override looper.lightglue_weight_dir")
parser.add_argument("--training-iters", type=int, default=None, help="Override training_args.iters to reduce mapper workload")
parser.add_argument("--loop-onnx-provider", choices=["cpu", "cuda"], default=None, help="Override loop ONNX Runtime provider")
parser.add_argument("--adaptive-runtime", action="store_true", help="Enable adaptive runtime budgeting for vis, loop, and mapper")
parser.add_argument("--enable-mapping-budget", action="store_true", help="Enable adaptive Gaussian addition budgeting in mapper")
parser.add_argument("--enable-jetson-pruning", action="store_true", help="Enable Jetson-aware Gaussian pruning scheduling")
parser.add_argument("--enable-pixel-budget", action="store_true", help="Enable dynamic pixel downsampling during mapper training")
parser.add_argument("--profile-runtime", action="store_true", help="Write module-level runtime profiling reports")
parser.add_argument("--profile-dba-memory", action="store_true", help="Sample synchronized CUDA memory at DBA stage boundaries; requires --profile-runtime")
parser.add_argument("--profile-dba-memory-samples-per-signature", type=int, default=2, help="Maximum synchronized memory samples for one exact DBA signature")
parser.add_argument("--enable-metric-depth-schedule", action="store_true", help="Run metric depth with interval or keyframe-aware scheduling instead of every frame")
parser.add_argument("--metric-depth-warmup", type=int, default=None, help="Number of initial frames that always run metric depth when scheduling is enabled")
parser.add_argument("--metric-depth-interval", type=int, default=None, help="Run metric depth every N frames after warmup when scheduling is enabled")
parser.add_argument("--metric-depth-mode", choices=["keyframe", "interval"], default=None, help="Metric depth scheduling mode")
parser.add_argument("--metric-depth-keyframe-min-interval", type=int, default=None, help="Minimum frame gap between metric depth predictions realized by keyframes")
parser.add_argument("--metric-depth-keyframe-force-interval", type=int, default=None, help="Force a metric depth prediction after this many keyframe frames without depth")
parser.add_argument("--metric-depth-high-motion-ratio", type=float, default=None, help="Predict metric depth early when keyframe motion exceeds this multiple of the frontend threshold")
parser.add_argument("--metric-depth-scale", type=float, default=None, help="Scale Metric3D internal forward crop size; depth output stays at the original image size")
parser.add_argument("--metric-depth-backend", choices=["torch", "tensorrt"], default=None, help="Metric3D inference backend")
parser.add_argument("--metric-depth-engine", default=None, help="TensorRT plan path for Metric3D depth inference")
parser.add_argument("--droid-cnet-backend", choices=["torch", "tensorrt"], default=None, help="DROID context encoder inference backend")
parser.add_argument("--droid-cnet-engine", default=None, help="TensorRT plan path for the DROID context encoder")
parser.add_argument("--droid-update-backend", choices=["torch", "tensorrt"], default=None, help="DROID update-core inference backend")
parser.add_argument("--droid-update-engine", default=None, help="TensorRT plan path for the DROID update core")
parser.add_argument("--tensorrt-strict", action="store_true", help="Fail instead of falling back when a requested TensorRT backend fails")
parser.add_argument("--no-vis", action="store_true", help="Force headless execution by disabling visualization")
parser.add_argument("--disable-loop", action="store_true", help="Disable loop closure to bypass optional loop dependencies")
parser.add_argument("--disable-metric", action="store_true", help="Disable metric depth inference")
parser.add_argument("--skip-save-ply", action="store_true", help="Skip final ply export to reduce optional dependency requirements")
parser.add_argument("--export-eval", action="store_true", help="Save lightweight keyframe pose/rgbdnua evaluation outputs without enabling online map visualization")
parser.add_argument("--export-eval-interval", type=int, default=None, help="Save lightweight evaluation outputs every N frame ids when --export-eval is enabled")
parser.add_argument("--enable-jetson-motion-gate", action="store_true", help="Use a low-cost Jetson motion gate before DROID motion filtering/DBA")
parser.add_argument("--motion-gate-backend", choices=["auto", "vpi_nvbuffer", "vpi_cpp", "vpi", "opencv"], default=None, help="Motion gate backend; auto tries VPI first and falls back to OpenCV")
parser.add_argument("--motion-gate-threshold", type=float, default=None, help="Optical-flow motion threshold for running full tracking")
parser.add_argument("--motion-gate-force-interval", type=int, default=None, help="Run full tracking at least every N input frames when motion gate is enabled")
parser.add_argument("--motion-gate-resize", default=None, help="Motion gate grayscale resize as H,W, e.g. 96,160")
parser.add_argument("--motion-gate-grid-size", type=int, default=None, help="VPI/OpenCV motion gate output grid size")
parser.add_argument("--motion-gate-vpi-levels", type=int, default=None, help="Number of VPI OFA pyramid levels; 1 is fastest and most robust")
parser.add_argument("--motion-gate-vpi-quality", choices=["low", "medium", "high"], default=None, help="VPI OFA optical-flow quality")
args = parser.parse_args()
if args.profile_dba_memory and not args.profile_runtime:
    parser.error("--profile-dba-memory requires --profile-runtime")
if args.profile_dba_memory_samples_per_signature < 1:
    parser.error("--profile-dba-memory-samples-per-signature must be at least 1")
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package, retrieve_to_tracker, datapacket_to_nerfslam
import time
from tqdm import tqdm
if config['mode'] == 'vo_nerfslam': from frontend_vo.vio_slam import VioSLAM


def apply_overrides(cfg):
    cfg.setdefault('dataset', {})
    cfg.setdefault('output', {})
    cfg.setdefault('frontend', {})
    cfg.setdefault('device', {})
    cfg.setdefault('looper', {})
    cfg.setdefault('runtime_budget', {})
    cfg.setdefault('mapping_budget', {})
    cfg.setdefault('pruning_budget', {})
    cfg.setdefault('pixel_budget', {})
    cfg.setdefault('profiling', {})
    cfg['profiling'].setdefault('runtime', {})
    cfg['profiling'].setdefault('dba_memory', {})
    cfg.setdefault('metric_depth_schedule', {})
    cfg.setdefault('inference', {})
    cfg.setdefault('eval_export', {})
    cfg.setdefault('jetson_motion_gate', {})

    if args.dataset_root is not None:
        cfg['dataset']['root'] = args.dataset_root
    if args.output_dir is not None:
        cfg['output']['save_dir'] = args.output_dir
    if args.frontend_weight is not None:
        cfg['frontend']['weight'] = args.frontend_weight
    if args.device_tracker is not None:
        cfg['device']['tracker'] = args.device_tracker
    if args.device_mapper is not None:
        cfg['device']['mapper'] = args.device_mapper
    if args.frontend_buffer is not None:
        cfg['frontend']['buffer'] = int(args.frontend_buffer)
    if args.frontend_save_buffer is not None:
        cfg['frontend']['save_buffer_size'] = int(args.frontend_save_buffer)
    if args.frontend_image_size is not None:
        try:
            h_str, w_str = [x.strip() for x in args.frontend_image_size.split(',')]
            cfg['frontend']['image_size'] = [int(h_str), int(w_str)]
        except Exception as exc:
            raise ValueError("--frontend-image-size must be in H,W format, e.g. 256,448") from exc
    if args.frontend_iters1 is not None:
        cfg['frontend']['iters1'] = int(args.frontend_iters1)
    if args.frontend_iters2 is not None:
        cfg['frontend']['iters2'] = int(args.frontend_iters2)
    if args.frontend_adaptive_iters:
        cfg['frontend']['adaptive_iters_enabled'] = True
    if args.frontend_iters1_high is not None:
        cfg['frontend']['iters1_high'] = int(args.frontend_iters1_high)
    if args.frontend_iters2_high is not None:
        cfg['frontend']['iters2_high'] = int(args.frontend_iters2_high)
    if args.frontend_iters_high_motion_ratio is not None:
        cfg['frontend']['iters_high_motion_ratio'] = float(args.frontend_iters_high_motion_ratio)
    if args.frontend_iters_force_interval is not None:
        cfg['frontend']['iters_force_interval'] = int(args.frontend_iters_force_interval)
    if args.lightglue_weight_dir is not None:
        cfg['looper']['lightglue_weight_dir'] = args.lightglue_weight_dir
    if args.training_iters is not None:
        cfg.setdefault('training_args', {})
        cfg['training_args']['iters'] = int(args.training_iters)
    if args.loop_onnx_provider is not None:
        cfg['looper']['onnx_provider'] = args.loop_onnx_provider
    if args.adaptive_runtime:
        cfg['runtime_budget']['enabled'] = True
    if args.enable_mapping_budget:
        cfg['mapping_budget']['enabled'] = True
    if args.enable_jetson_pruning:
        cfg['pruning_budget']['enabled'] = True
    if args.enable_pixel_budget:
        cfg['pixel_budget']['enabled'] = True
    if args.profile_runtime:
        cfg['profiling']['runtime']['enabled'] = True
    cfg['profiling']['dba_memory']['enabled'] = bool(args.profile_dba_memory)
    cfg['profiling']['dba_memory']['samples_per_signature'] = int(
        args.profile_dba_memory_samples_per_signature
    )
    if args.enable_metric_depth_schedule:
        cfg['metric_depth_schedule']['enabled'] = True
    if args.metric_depth_warmup is not None:
        cfg['metric_depth_schedule']['warmup_frames'] = int(args.metric_depth_warmup)
    if args.metric_depth_interval is not None:
        cfg['metric_depth_schedule']['interval'] = int(args.metric_depth_interval)
    if args.metric_depth_mode is not None:
        cfg['metric_depth_schedule']['mode'] = args.metric_depth_mode
    if args.metric_depth_keyframe_min_interval is not None:
        cfg['metric_depth_schedule']['keyframe_min_interval'] = int(args.metric_depth_keyframe_min_interval)
    if args.metric_depth_keyframe_force_interval is not None:
        cfg['metric_depth_schedule']['keyframe_force_interval'] = int(args.metric_depth_keyframe_force_interval)
    if args.metric_depth_high_motion_ratio is not None:
        cfg['metric_depth_schedule']['high_motion_ratio'] = float(args.metric_depth_high_motion_ratio)
    if args.metric_depth_scale is not None:
        cfg['metric_depth_scale'] = float(args.metric_depth_scale)
    if args.metric_depth_backend is not None:
        cfg['inference']['metric3d_backend'] = args.metric_depth_backend
    if args.metric_depth_engine is not None:
        cfg['inference']['metric3d_engine'] = args.metric_depth_engine
    if args.droid_cnet_backend is not None:
        cfg['inference']['droid_cnet_backend'] = args.droid_cnet_backend
    if args.droid_cnet_engine is not None:
        cfg['inference']['droid_cnet_engine'] = args.droid_cnet_engine
    if args.droid_update_backend is not None:
        cfg['inference']['droid_update_backend'] = args.droid_update_backend
    if args.droid_update_engine is not None:
        cfg['inference']['droid_update_engine'] = args.droid_update_engine
    if args.tensorrt_strict:
        cfg['inference']['tensorrt_strict'] = True
    if args.no_vis:
        cfg['use_vis'] = False
    if args.disable_loop:
        cfg['use_loop'] = False
    if args.disable_metric:
        cfg['use_metric'] = False
    if args.skip_save_ply:
        cfg['skip_save_ply'] = True
    if args.export_eval:
        cfg['eval_export']['enabled'] = True
    if args.export_eval_interval is not None:
        cfg['eval_export']['interval'] = int(args.export_eval_interval)
    if args.enable_jetson_motion_gate:
        cfg['jetson_motion_gate']['enabled'] = True
    if args.motion_gate_backend is not None:
        cfg['jetson_motion_gate']['backend'] = args.motion_gate_backend
    if args.motion_gate_threshold is not None:
        cfg['jetson_motion_gate']['threshold'] = float(args.motion_gate_threshold)
    if args.motion_gate_force_interval is not None:
        cfg['jetson_motion_gate']['force_interval'] = int(args.motion_gate_force_interval)
    if args.motion_gate_resize is not None:
        h_str, w_str = [part.strip() for part in args.motion_gate_resize.split(',')]
        cfg['jetson_motion_gate']['resize'] = [int(h_str), int(w_str)]
    if args.motion_gate_grid_size is not None:
        cfg['jetson_motion_gate']['grid_size'] = int(args.motion_gate_grid_size)
    if args.motion_gate_vpi_levels is not None:
        cfg['jetson_motion_gate']['vpi_num_levels'] = int(args.motion_gate_vpi_levels)
    if args.motion_gate_vpi_quality is not None:
        cfg['jetson_motion_gate']['vpi_quality'] = args.motion_gate_vpi_quality
    return cfg


def load_vis_utils():
    from gaussian.vis_utils import save_ply, vis_map, vis_bev
    return save_ply, vis_map, vis_bev


def sync_cuda_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        self.skip_save_ply = bool(cfg.get('skip_save_ply', False))
        
        if self.cfg['mode'] == 'vio' or self.cfg['mode'] == 'vo':
            self.tracker = DBAFusion(cfg)
        elif self.cfg['mode'] == 'vo_nerfslam':     
            self.tracker = VioSLAM(cfg)
        else: assert False, "Error \"mode\" in config file."
        
        if 'phone' not in cfg['dataset']['module']: self.tracker.dataset_length = len(self.dataset)
        
        self.mapper = GaussianModel(cfg)

        self.save_ply_fn = None
        self.vis_map_fn = None
        self.vis_bev_fn = None

        if 'use_loop' in cfg.keys() and cfg['use_loop']:
            from loop.loop_model import LoopModel
            self.use_loop = True
            self.looper = LoopModel(cfg)
        else:
            self.use_loop = False
            self.looper = None
        
        if 'use_metric' in cfg.keys() and cfg['use_metric']:
            try:
                from metric.metric_model import Metric_Model
            except Exception as exc:
                raise RuntimeError(
                    "Metric depth is enabled but metric dependencies are unavailable. "
                    "Disable metric with --disable-metric or set use_metric: False."
                ) from exc
            self.metric_predictor = Metric_Model(cfg)

        if cfg.get('use_vis', False) or (not self.skip_save_ply):
            try:
                self.save_ply_fn, self.vis_map_fn, self.vis_bev_fn = load_vis_utils()
            except Exception as exc:
                if cfg.get('use_vis', False):
                    raise RuntimeError(
                        "Visualization is enabled but visualization dependencies are unavailable. "
                        "Use --no-vis for headless deployment."
                    ) from exc
                if not self.skip_save_ply:
                    raise RuntimeError(
                        "PLY export requires optional visualization dependencies. "
                        "Use --skip-save-ply for smoke tests."
                    ) from exc
        
        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            from storage.storage_manage import StorageManager
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
            if cfg['dataset']['module'] != 'phone':
                self.storage_manager.dataset_length = self.dataset.rgbinfo_dict['timestamp'][-1] - self.dataset.rgbinfo_dict['timestamp'][0] 
        else:
            self.use_storage_manager = False

        self.runtime_budget = RuntimeBudgetController(cfg, dataset_length=len(self.dataset))
        self._last_runtime_budget_stage = None
        self.profiler = RuntimeProfiler.from_config(cfg, sync_callback=sync_cuda_if_available)
        if hasattr(self, "metric_predictor"):
            self.profiler.set_metadata(
                "metric_depth", self.metric_predictor.backend_status()
            )
        if hasattr(self.tracker, "set_runtime_profiler"):
            self.tracker.set_runtime_profiler(self.profiler)
        self.metric_depth_scheduler = MetricDepthScheduler(cfg)

    def _get_gaussian_count(self):
        xyz = getattr(self.mapper, "_xyz", None)
        if xyz is None:
            return 0
        return int(xyz.shape[0])

    def _sync_mapper_iters(self, target_iters):
        self.cfg.setdefault("training_args", {})
        self.cfg["training_args"]["iters"] = int(target_iters)
        self.mapper.cfg.setdefault("training_args", {})
        self.mapper.cfg["training_args"]["iters"] = int(target_iters)

    def _log_runtime_budget(self, runtime_budget, frame_idx, keyframe_id, gaussian_count):
        if not runtime_budget["enabled"]:
            return
        if runtime_budget["stage"] == self._last_runtime_budget_stage:
            return
        self._last_runtime_budget_stage = runtime_budget["stage"]
        print(
            "[runtime_budget] "
            f"stage={runtime_budget['stage']} "
            f"frame={frame_idx + 1} "
            f"keyframe={keyframe_id} "
            f"gaussians={gaussian_count} "
            f"mapper_iters={runtime_budget['mapper_iters']} "
            f"vis_every={runtime_budget['vis_interval']} "
            f"loop_every={runtime_budget['loop_interval']}"
        )

    def run(self):
        # Load imu data.
        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
        
        mapper_run_times = 0
        
        # Run Tracking.
        for idx in tqdm(range(len(self.dataset))):
            with self.profiler.time("frame_total", idx):
                with self.profiler.time("dataset_load", idx):
                    data_packet = self.dataset[idx]

                if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                    with self.profiler.time("mobile_preload_imu_stamp", idx):
                        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()

                if 'use_metric' in self.cfg.keys() and self.cfg['use_metric']:
                    has_depth = 'depth' in data_packet.keys() and data_packet['depth'] is not None
                    metric_decision = self.metric_depth_scheduler.decide(idx, has_depth)
                    self.metric_depth_scheduler.record(metric_decision)
                    if metric_decision["should_predict"]:
                        with self.profiler.time("metric_depth", idx):
                            data_packet['depth'] = self.metric_predictor.predict(
                                data_packet['rgb'][0], profiler=self.profiler, frame_idx=idx
                            )
                    elif metric_decision["reason"] == "keyframe_lazy":
                        def predict_metric_depth():
                            with self.profiler.time("metric_depth", idx):
                                return self.metric_predictor.predict(
                                    data_packet['rgb'][0], profiler=self.profiler, frame_idx=idx
                                )
                        data_packet['depth'] = self.metric_depth_scheduler.make_lazy_depth_provider(predict_metric_depth, frame_idx=idx)
                    elif metric_decision["reason"] == "scheduled_skip":
                        data_packet.pop('depth', None)

                with self.profiler.time("preload_imu_stamp", idx):
                    self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                    self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()

                # torch.set_grad_enabled(False)
                if hasattr(self.tracker, "set_runtime_profiler_frame_idx"):
                    self.tracker.set_runtime_profiler_frame_idx(idx)
                with self.profiler.time("tracking", idx):
                    self.tracker.track(data_packet if not self.cfg['mode']=='vo_nerfslam' else datapacket_to_nerfslam(data_packet, idx))
                # torch.set_grad_enabled(True)

                with self.profiler.time("cache_cleanup", idx):
                    torch.cuda.empty_cache()
                # Judge whether new keyframe is added and package keyframe dict.
                with self.profiler.time("package_keyframe", idx):
                    viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])

                if viz_out is not None and (self.cfg['mode'] in ['vo', 'vo_nerfslam'] or self.tracker.video.imu_enabled):
                    keyframe_id = int(viz_out["global_kf_id"][-1])
                    gaussian_count = self._get_gaussian_count()
                    with self.profiler.time("runtime_budget", idx):
                        runtime_budget = self.runtime_budget.evaluate(
                            frame_idx=idx,
                            keyframe_id=keyframe_id,
                            gaussian_count=gaussian_count,
                        )
                        self._log_runtime_budget(runtime_budget, idx, keyframe_id, gaussian_count)
                        self._sync_mapper_iters(runtime_budget["mapper_iters"])

                    # Save and check.
                    with self.profiler.time("mapping", idx):
                        new_viz_out = self.mapper.run(viz_out, True)

                    if self.use_loop:
                        if runtime_budget["should_run_loop"]:
                            with self.profiler.time("loop", idx):
                                self.looper.run(self.mapper, self.tracker, viz_out, idx)

                    if self.use_storage_manager and (idx+1) % 10 == 0:
                        with self.profiler.time("storage", idx):
                            self.storage_manager.run(self.tracker, self.mapper, viz_out)
                            torch.cuda.empty_cache()

                    if self.cfg['use_vis'] and runtime_budget["should_run_vis"]:
                        with self.profiler.time("visualization", idx):
                            if not self.cfg['use_storage_manager'] or self.storage_manager._xyz.shape[0]==0:
                                self.vis_map_fn(self.tracker, self.mapper)
                                self.vis_bev_fn(self.tracker, self.mapper)
                            else:
                                self.storage_manager.vis_map_storage(self.tracker, self.mapper)
                                self.storage_manager.vis_bev_storage(self.tracker, self.mapper)

                if (idx == len(self.dataset) - 1) and self.mapper._xyz.shape[0] > 0 and (not self.skip_save_ply):
                # if ((idx+1) % 100 == 0 or (idx == len(self.dataset) - 1)) and self.mapper._xyz.shape[0] > 0:
                    with self.profiler.time("save_ply", idx):
                        self.save_ply_fn(self.mapper, idx, save_mode='2dgs')
                    # save_ply(self.mapper, idx, save_mode='pth')

        motion_gate = getattr(self.tracker, 'motion_gate', None)
        if motion_gate is not None:
            motion_gate.maybe_log_summary()
        self.metric_depth_scheduler.maybe_log_summary()
        self.profiler.write_reports()
            

if __name__ == '__main__':
    config = apply_overrides(config)
    
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config)+'-{}-'.format(config_path.split('/')[-1].strip('.yaml'))+args.prefix)
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    
    runner.run()
    
    
