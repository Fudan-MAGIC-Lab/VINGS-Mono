import torch
import lietorch
import numpy as np
from frontend.droid_net import DroidNet
from frontend.depth_video import DepthVideo
from frontend.motion_filter import MotionFilter
from frontend.motion_gate import JetsonMotionGate
from frontend.droid_cnet_backends import install_droid_cnet_backend
from frontend.droid_update_backends import install_droid_update_backend
from frontend.dbaf_frontend import DBAFusionFrontend
from collections import OrderedDict
from contextlib import nullcontext
from torch.multiprocessing import Process
from lietorch import SE3
import frontend.geom.projective_ops as pops
import droid_backends
import pickle

try:
    import gtsam
except ModuleNotFoundError:
    gtsam = None

class DBAFusion:
    def __init__(self, cfg):
        super(DBAFusion, self).__init__()
        self.cfg = cfg
        self.load_weights(cfg['frontend']['weight']) # load DroidNet weights
        inference_config = cfg.get('inference', {})
        self.cnet_backend = install_droid_cnet_backend(
            self.net, inference_config
        )
        if self.cnet_backend is None:
            self.cnet_backend_status = {
                'requested_backend': 'torch',
                'actual_backend': 'torch',
                'strict': bool(inference_config.get('tensorrt_strict', False)),
                'engine_path': None,
                'metadata_path': None,
                'engine_sha256': None,
                'fallback_reason': None,
            }
        else:
            self.cnet_backend_status = self.cnet_backend.backend_status()
        self.update_backend = install_droid_update_backend(
            self.net.update, inference_config
        )
        if self.update_backend is None:
            self.update_backend_status = {
                'requested_backend': 'torch',
                'actual_backend': 'torch',
                'strict': bool(inference_config.get('tensorrt_strict', False)),
                'engine_path': None,
                'metadata_path': None,
                'engine_sha256': None,
                'precision': None,
                'fallback_reason': None,
            }
        else:
            self.update_backend_status = self.update_backend.backend_status()

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(cfg, cfg['frontend']['image_size'], cfg['frontend']['buffer'])
        self.video.Ti1c = cfg['frontend']['c2i']
        if cfg['mode'] == 'vio':
            if gtsam is None:
                raise ModuleNotFoundError(
                    "gtsam is required for VIO mode. Use mode=vo or install gtsam with Python>=3.9."
                )
            self.video.Tbc = gtsam.Pose3(self.video.Ti1c)
            self.video.state.set_imu_params([ 0.0003924 * 25,0.000205689024915 * 25, 0.004905 * 10, 0.000001454441043 * 500])
            self.video.init_pose_sigma = np.array([1.0, 1.0, 0.0001, 1.0, 1.0, 1.0])
            self.video.init_bias_sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        else:
            self.video.Tbc = None

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=cfg['frontend']['filter_thresh'])
        motion_gate = JetsonMotionGate(cfg)
        self.motion_gate = motion_gate if motion_gate.enabled else None
        
        # frontend process
        self.frontend = DBAFusionFrontend(self.net, self.video, self.cfg)
        
        self.frontend.translation_threshold  = 0.2
        self.frontend.graph.mask_threshold   = -1.0
        self.upsample = True
        self.profiler = None
        self.profiler_frame_idx = None
        
        self.dataset_length = None
        
    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v)
            for (k, v) in torch.load(weights, map_location="cpu").items()
        ])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def set_runtime_profiler(self, profiler):
        self.profiler = profiler
        cnet_backend = getattr(self, 'cnet_backend', None)
        if cnet_backend is None:
            cnet_status = getattr(self, 'cnet_backend_status', None)
            if cnet_status is not None and hasattr(profiler, 'set_metadata'):
                profiler.set_metadata('droid_cnet', cnet_status)
        else:
            cnet_backend.set_profiler(profiler)
        update_backend = getattr(self, 'update_backend', None)
        if update_backend is None:
            update_status = getattr(self, 'update_backend_status', None)
            if update_status is not None and hasattr(profiler, 'set_metadata'):
                profiler.set_metadata('droid_update', update_status)
        else:
            update_backend.set_profiler(profiler)
        self.filterx.profiler = profiler
        self.frontend.profiler = profiler
        self.frontend.graph.profiler = profiler
        self.net.update.profiler = profiler

    def set_runtime_profiler_frame_idx(self, frame_idx):
        self.profiler_frame_idx = frame_idx
        self.filterx.profiler_frame_idx = frame_idx
        self.frontend.profiler_frame_idx = frame_idx
        self.frontend.graph.profiler_frame_idx = frame_idx
        self.net.update.profiler_frame_idx = frame_idx

    def _profile(self, stage):
        profiler = getattr(self, "profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.time(stage, getattr(self, "profiler_frame_idx", None))

    def track(self, data_packet):
        """ main thread - update map """
        tstamp, image, intrinsic = data_packet['timestamp'], data_packet['rgb'], data_packet['intrinsic']
        with torch.no_grad():
            motion_gate = getattr(self, "motion_gate", None)
            if motion_gate is not None:
                with self._profile("frontend_motion_gate"):
                    if hasattr(motion_gate, "decide_packet"):
                        gate_decision = motion_gate.decide_packet(data_packet)
                    else:
                        gate_decision = motion_gate.decide(image)
                video = getattr(self, "video", None)
                if video is not None:
                    video.last_motion_gate_score = gate_decision.get("score")
                    video.last_motion_gate_reason = gate_decision.get("reason")
                    video.last_motion_gate_backend = gate_decision.get("backend")
                if gate_decision.get("skip", False):
                    return

            # check there is enough motion
            depth = None if 'depth' not in list(data_packet.keys()) else data_packet['depth']
            with self._profile("frontend_motion_filter"):
                self.filterx.track(tstamp, image, depth, intrinsic)
            # local bundle adjustment
            with self._profile("frontend_dba_update"):
                self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        del self.frontend
        if self.cnet_backend is not None:
            self.cnet_backend.close()
        if self.update_backend is not None:
            self.update_backend.close()

    # Tailored for debug Looper.
    def save_pt_ckpt(self, save_path):
        # Only save video's attrributes.
        save_dict = {'frontend': {'video': {'poses_save': None,}}}
        if hasattr(self, 'local_to_global_bias'):
            save_dict['local_to_global_bias'] = self.local_to_global_bias
        save_dict['frontend']['video']['tstamp_save']    = self.frontend.video.tstamp_save
        save_dict['frontend']['video']['poses_save']     = self.frontend.video.poses_save
        save_dict['frontend']['video']['images_up_save'] = self.frontend.video.images_up_save
        save_dict['frontend']['video']['disps_up_save']  = self.frontend.video.disps_up_save
        save_dict['frontend']['video']['disps_save']     = self.frontend.video.disps_save
        save_dict['frontend']['video']['poses']          = self.frontend.video.poses
        save_dict['frontend']['video']['disps_up']       = self.frontend.video.disps_up
        save_dict['frontend']['video']['disps']          = self.frontend.video.disps
        save_dict['frontend']['video']['depths_cov_up_save'] = self.frontend.video.depths_cov_up_save
        
        if hasattr(self.frontend.video, 'count_save'):
            save_dict['frontend']['video']['count_save'] = self.frontend.video.count_save
            save_dict['frontend']['video']['count_save_bias'] = self.frontend.video.count_save_bias
        
        torch.save(save_dict, save_path)
    
    def load_pt_ckpt(self, load_path):
        load_dict = torch.load(load_path)
        if 'local_to_global_bias' in load_dict.keys():
            self.local_to_global_bias          = load_dict['local_to_global_bias']
        self.frontend.video.tstamp_save    = load_dict['frontend']['video']['tstamp_save']
        self.frontend.video.poses_save     = load_dict['frontend']['video']['poses_save']
        self.frontend.video.images_up_save = load_dict['frontend']['video']['images_up_save']
        self.frontend.video.disps_up_save  = load_dict['frontend']['video']['disps_up_save']
        self.frontend.video.disps_save     = load_dict['frontend']['video']['disps_save']
        self.frontend.video.poses          = load_dict['frontend']['video']['poses']
        self.frontend.video.disps_up       = load_dict['frontend']['video']['disps_up']
        self.frontend.video.disps          = load_dict['frontend']['video']['disps']
        
        # TTD 2024/12/21
        # 这个很重要哈, 没这些完全不知道怎么继续跑;
        # PART 1 Motionfilter.track
        # self.frontend.video.counter.value  = None
        # self.filterx.net, self.filterx.inp, self.filterx.fmap       = None, None, None
        # TODO: Maybe self.filterx.video.append(..., intrinsics / 8.0, gmap, net[0], inp[0]) ?
        
        # PART 2 DBAFusionFrontend.__call__
        # self.net = None
        
        
        
        if 'count_save' in load_dict['frontend']['video'].keys():
            self.frontend.video.count_save      = load_dict['frontend']['video']['count_save']
            self.frontend.video.count_save_bias = load_dict['frontend']['video']['count_save_bias']
        
    

