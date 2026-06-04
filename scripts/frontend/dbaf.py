import torch
import lietorch
import numpy as np
from frontend.droid_net import DroidNet
from frontend.depth_video import DepthVideo
from frontend.motion_filter import MotionFilter
from frontend.dbaf_frontend import DBAFusionFrontend
from collections import OrderedDict
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
        
        # frontend process
        self.frontend = DBAFusionFrontend(self.net, self.video, self.cfg)
        
        self.frontend.translation_threshold  = 0.2
        self.frontend.graph.mask_threshold   = -1.0
        self.upsample = True
        
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

    def track(self, data_packet):
        """ main thread - update map """
        tstamp, image, intrinsic = data_packet['timestamp'], data_packet['rgb'], data_packet['intrinsic']
        with torch.no_grad():
            # check there is enough motion
            depth = None if 'depth' not in list(data_packet.keys()) else data_packet['depth']
            self.filterx.track(tstamp, image, depth, intrinsic)
            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        del self.frontend

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
        
    
    