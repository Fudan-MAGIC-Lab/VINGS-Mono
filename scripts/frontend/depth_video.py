from __future__ import annotations

import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value

from frontend.droid_net import cvx_upsample
import frontend.geom.projective_ops as pops

from frontend.multi_sensor import MultiSensorState
from scipy.spatial.transform import Rotation
import copy
import logging
import frontend.geoFunc.trans as trans
from lietorch import SE3

try:
    import gtsam
    from gtsam.symbol_shorthand import B, V, X
except ModuleNotFoundError:
    gtsam = None
    B = V = X = None

# cfg['use_uncertainty'] = True -  -  -  -  -  -  -  -  -  -  -  -
# Maybe gtsam_nerfslam? 🤔
if gtsam is not None:
    from gtsam import (HessianFactor)
    from gtsam import Values
    from gtsam import PriorFactorPose3, NonlinearFactorGraph, GaussianFactorGraph
    from gtsam import Pose3
else:
    HessianFactor = Values = PriorFactorPose3 = NonlinearFactorGraph = GaussianFactorGraph = Pose3 = None
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

import droid_backends as droid_backends_nerfslam
if gtsam is not None:
    from vings_utils.gtsam_utils import gtsam_pose_to_torch
else:
    gtsam_pose_to_torch = None


def BA2GTSAM(H: np.ndarray, v: np.ndarray, Tbc: gtsam.Pose3):
    A = -Tbc.inverse().AdjointMap()
    # A = -np.eye(6,6)
    A = np.concatenate([A[3:6,:],A[0:3,:]],axis=0)
    ss = H.shape[0]//6
    J = np.zeros_like(H)
    for i in range(ss):
       J[(i*6):(i*6+6),(i*6):(i*6+6)] = A
    JT = J.T
    return np.matmul(np.matmul(JT,H),J),np.matmul(JT,v)

def CustomHessianFactor(values: gtsam.Values, H: np.ndarray, v: np.ndarray):
    info_expand = np.zeros([H.shape[0]+1,H.shape[1]+1])
    info_expand[0:-1,0:-1] = H
    info_expand[0:-1,-1] = v
    info_expand[-1,-1] = 0.0 # This is meaningless.
    h_f = gtsam.HessianFactor(values.keys(),[6]*len(values.keys()),info_expand)
    l_c = gtsam.LinearContainerFactor(h_f,values)
    return l_c

class DepthVideo:
    def __init__(self, cfg, image_size=[480, 640], buffer=1024, save_pkl = True, stereo=False, upsample = True, device="cuda:0"):
                
        # current keyframe count
        self.cfg = cfg
        self.use_shared_memory = bool(self.cfg.get('use_shared_memory', False))
        buffer_size = buffer + 1

        def maybe_share(tensor):
            return tensor.share_memory_() if self.use_shared_memory else tensor

        self.device = self.cfg['device']['tracker']
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = maybe_share(torch.zeros(buffer_size, device="cuda", dtype=torch.float64))
        self.images = torch.zeros(buffer_size, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = maybe_share(torch.zeros(buffer_size, device="cuda", dtype=torch.bool))
        self.red = maybe_share(torch.zeros(buffer_size, device="cuda", dtype=torch.bool))
        self.poses = maybe_share(torch.zeros(buffer_size, 7, device="cuda", dtype=torch.float))
        self.poses[:, -1] = 1
        self.disps = maybe_share(torch.ones(buffer_size, ht//8, wd//8, device="cuda", dtype=torch.float))
        self.disps_sens = maybe_share(torch.zeros(buffer_size, ht//8, wd//8, device="cuda", dtype=torch.float))
        self.disps_up   = maybe_share(torch.zeros(buffer_size, ht, wd, device="cuda", dtype=torch.float))
        self.intrinsics = maybe_share(torch.zeros(buffer_size, 4, device="cuda", dtype=torch.float))

        # TTD 2024/09/24
        self.depths_cov    = maybe_share(100.0 * torch.ones(buffer_size, ht//8, wd//8, device="cuda", dtype=torch.float))
        self.depths_cov_up = maybe_share(100.0 * torch.ones(buffer_size, ht, wd, device="cuda", dtype=torch.float))
        
        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = maybe_share(torch.zeros(buffer_size, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda"))
        self.nets = maybe_share(torch.zeros(buffer_size, 128, ht//8, wd//8, dtype=torch.half, device="cuda"))
        self.inps = maybe_share(torch.zeros(buffer_size, 128, ht//8, wd//8, dtype=torch.half, device="cuda"))

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        
        ### DBAFusion
        # for .pkl saving
        if '360' in self.cfg['dataset']['module']:
            SAVE_BUFFER_SIZE = 8000
        else:
            SAVE_BUFFER_SIZE = 2500
        
        self.disps_save = torch.ones(SAVE_BUFFER_SIZE, ht//8, wd//8, device="cpu", dtype=torch.float)
        self.poses_save = torch.zeros(SAVE_BUFFER_SIZE, 7, device="cpu", dtype=torch.float)
        self.poses_save[:, -1] = 1
        self.tstamp_save = torch.zeros(SAVE_BUFFER_SIZE, device="cpu", dtype=torch.float64)
        self.images_save = torch.zeros(SAVE_BUFFER_SIZE, ht//8, wd//8, 3, device="cpu", dtype=torch.float)
        if upsample:
            self.disps_up_save  = maybe_share(torch.zeros(SAVE_BUFFER_SIZE, ht, wd, device="cpu", dtype=torch.float))
            self.depths_cov_up_save = maybe_share(100.0 * torch.ones(SAVE_BUFFER_SIZE, ht, wd, device="cpu", dtype=torch.float))
            self.images_up_save = maybe_share(torch.zeros(SAVE_BUFFER_SIZE, ht, wd, 3, device="cpu", dtype=torch.float))
        self.count_save = 0
        self.save_pkl = save_pkl
        self.upsample_flag = upsample

        self.imu_enabled = False
        if self.cfg['mode'] == 'vio':
            if gtsam is None:
                raise ModuleNotFoundError(
                    "gtsam is required for VIO mode. Use mode=vo or install gtsam with Python>=3.9."
                )
            self.state = MultiSensorState()
            self.ignore_imu = False
        elif self.cfg['mode'] == 'vo':
            self.state = None
            self.ignore_imu = True
        else:
            raise ValueError('Invalid mode')

        self.last_t0 = 0
        self.last_t1 = 0
        self.cur_graph = None
        self.cur_result = None
        self.marg_factor = None
        self.prior_factor = []
        self.prior_factor_map = {}
        self.cur_ii = None
        self.cur_jj = None
        self.cur_target = None
        self.cur_weight = None
        self.cur_eta = None

        self.xyz_ref = []
        
        # extrinsics, need to be set in the main .py
        self.Ti1c = None  # shape = (4,4)
        self.Tbc = None   # gtsam.Pose3
        self.tbg = None   # shape = (3)

        self.reinit = False
        self.vi_init_t1 = -1
        self.vi_init_time = 0.0
        self.gnss_init_t1 = -1
        self.gnss_init_time = 0.0
        self.ten0 = None
        
        
        self.height_dsf, self.width_dsf = self.ht//8, self.wd//8
        
        self.init_pose_sigma =np.array([0.1, 0.1, 0.0001, 0.0001,0.0001,0.0001])
        self.init_bias_sigma =np.array([1.0,1.0,1.0, 0.1, 0.1, 0.1])

        
        # Remember to Delete.
        self.visual_only_init = False
        
    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """
        #       mask.shape = [1, 8, 576, 43, 77]
        # self.disps.shape = [80, 43, 77]
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()
        depths_cov_up = cvx_upsample(self.depths_cov[ix].unsqueeze(-1), mask)
        self.depths_cov_up[ix] = depths_cov_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask
    
    def reproject_comp(self, ii, jj, xyz_comp):
        ii, jj = DepthVideo.format_indicies(ii,jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform_comp(Gs, self.disps[None], self.intrinsics[None], ii, jj, xyz_comp)

        return coords, valid_mask
    
    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def rm_new_gnss(self, t1):
        if (self.gnss_init_t1> 0 and self.state.gnss_valid[t1]) or self.state.odo_valid[t1]:
            graph_temp = gtsam.NonlinearFactorGraph()
            linear_point  = self.marg_factor.linearizationPoint()
            graph_temp.push_back(self.marg_factor)

            if self.state.gnss_valid[t1]:
                T1 = self.state.wTbs[t1]
                T0 = self.state.wTbs[t1-1]
                p = np.matmul(trans.Cen(self.ten0).T, self.state.gnss_position[t1] - self.ten0)
                n0pbg = self.state.wTbs[t1].rotation().rotate(self.tbg)
                p = p - n0pbg
                p = p - T1.translation() + T0.translation()
                if not linear_point.exists(X(t1-1)):
                    linear_point.insert(X(t1-1), self.cur_result.atPose3(X(t1-1)))
                gnss_factor = gtsam.GPSFactor(X(t1-1), p,\
                              gtsam.noiseModel.Robust.Create(\
                              gtsam.noiseModel.mEstimator.Cauchy(0.08),\
                  gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,5.0]))))
                graph_temp.push_back(gnss_factor)
            if self.state.odo_valid[t1]:
                v1 = np.matmul(self.state.wTbs[t1].rotation().matrix().T, self.state.vs[t1])
                v0 = np.matmul(self.state.wTbs[t1-1].rotation().matrix().T, self.state.vs[t1-1])
                v = self.state.odo_vel[t1] - v1 + v0
                if not linear_point.exists(X(t1-1)):
                    linear_point.insert(X(t1-1), self.cur_result.atPose3(X(t1-1)))
                if not linear_point.exists(V(t1-1)):
                    linear_point.insert(V(t1-1), self.cur_result.atVector(V(t1-1)))
                odo_factor = gtsam.VelFactor(X(t1-1),V(t1-1),v,gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0,2.0,2.0])))
                graph_temp.push_back(odo_factor)           
            
            h_factor = graph_temp.linearizeToHessianFactor(linear_point)
            self.marg_factor = gtsam.LinearContainerFactor(h_factor,linear_point)
            
    
    def set_prior(self, t0, t1):
        for i in range(t0,t0+2):
            self.prior_factor_map[i] = []
            init_pose_sigma = self.init_pose_sigma
            if len(self.init_pose_sigma.shape) > 1:
                init_pose_sigma = self.init_pose_sigma[i-t0]
            self.prior_factor_map[i].append(gtsam.PriorFactorPose3(X(i),\
                                         self.state.wTbs[i], \
                                         gtsam.noiseModel.Diagonal.Sigmas(init_pose_sigma)))
            if not self.ignore_imu:
                self.prior_factor_map[i].append(gtsam.PriorFactorConstantBias(B(i),\
                                             self.state.bs[i], \
                                             gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma)))
            self.last_t0 = t0
            self.last_t1 = t1

    
    
    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        # self.ba_raw(target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
        # self.ba_uncertainty_dbaf(target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
        # self.ba_uncertainty_nerfslam(target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
        if self.cfg['mode'] == 'vio':
            self.ba_raw(target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
        elif self.cfg['mode'] == 'vo':
            self.ba_raw(target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
            # self.ba_uncertainty_nerfslam(target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
        
        
    def ba_raw(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """
        with self.get_lock():
            
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # 1) visual-only BA
            # 2) multi-sensor BA
            if not self.imu_enabled: 
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                    target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
                '''
                ii, jj是对video.poses, disps等的索引;
                确实, 这里肯定会落后tracker;
                在 min(ii_last, jj_last)~min(ii, jj) 之间搞。
                '''
                for i in range(self.last_t0, min(ii.min().item(), jj.min().item())):
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save] = self.tstamp[i].clone().cpu()
                        self.disps_save[self.count_save] = self.disps[i].clone().cpu()
                        self.poses_save[self.count_save] = self.poses[i].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save] = self.disps_up[i].clone().cpu()
                            self.depths_cov_up_save[self.count_save] = self.depths_cov_up[i].clone().cpu()
                            self.images_up_save[self.count_save] = self.images[i,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                            # self.images_up_save[self.count_save] = self.images[i,:].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        
                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                        # self.images_save[self.count_save] = self.images[i,:,::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.count_save += 1
                        # i in [self.last_t0, min(ii.min().item(), jj.min().item())] 对应 []
                        # NOTE: "min(ii.min().item(), jj.min().item()) 对应 self.count_save"
                
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
                # TTD 2024/10/15
                # Dangerous Option, 上面的那些都是存了就不改了，俺这边存了还会改的;
                for i_temp in range(min(ii.min().item(), jj.min().item()), t1):
                    bias = i_temp-min(ii.min().item(), jj.min().item())
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save+bias] = self.tstamp[i_temp].clone().cpu()
                        self.disps_save[self.count_save+bias] = self.disps[i_temp].clone().cpu()
                        self.poses_save[self.count_save+bias] = self.poses[i_temp].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save+bias] = self.disps_up[i_temp].clone().cpu()
                            self.depths_cov_up_save[self.count_save+bias] = self.depths_cov_up[i_temp].clone().cpu()
                            self.images_up_save[self.count_save+bias] = self.images[i_temp,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.images_save[self.count_save+bias] = self.images[i_temp,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                self.count_save_bias = t1 - min(ii.min().item(), jj.min().item())
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
                
                self.last_t0 = min(ii.min().item(), jj.min().item())
                self.last_t1 = t1
            else:
                last_last_t0 = self.last_t0
                t0 = min(ii.min().item(), jj.min().item())
                Q, E = None, None
                
                """ marginalization """
                if self.last_t1!=t1 or self.last_t0 != t0:
                    if self.last_t0 > t0:
                        t0 = self.last_t0
                    elif self.last_t0 == t0:
                        t0 = self.last_t0
                    else:
                        marg_paras = []
                        # Construct a temporary factor graph (related to the old states) to obtain the marginalization information
                        graph = gtsam.NonlinearFactorGraph()
                        marg_idx = torch.logical_and(torch.greater_equal(self.cur_ii,self.last_t0),\
                                                    torch.less(self.cur_ii,t0))
                        marg_idx2 = torch.logical_and(torch.less(self.cur_ii,self.last_t1-2),\
                                                     torch.less(self.cur_jj,self.last_t1-2))
                        marg_idx = torch.logical_and(marg_idx,marg_idx2)

                        marg_ii = self.cur_ii[marg_idx]
                        marg_jj = self.cur_jj[marg_idx]
                        marg_t0 = self.last_t0 
                        marg_t1 = t0 + 1
                        if len(marg_ii) > 0:
                            marg_t0 = self.last_t0 
                            marg_t1 = torch.max(marg_jj).item()+1
                            marg_result = gtsam.Values()
                            for i in range(self.last_t0,marg_t1): # 就存一个优化完的结果？
                                if i < t0:
                                    marg_paras.append(X(i))
                                    if self.save_pkl:
                                        # save marginalized results
                                        self.tstamp_save[self.count_save] = self.tstamp[i].clone().cpu()
                                        self.disps_save[self.count_save] = self.disps[i].clone().cpu()
                                        self.poses_save[self.count_save] = self.poses[i].clone().cpu()
                                        if self.upsample_flag:
                                            self.disps_up_save[self.count_save] = self.disps_up[i].clone().cpu()
                                            self.depths_cov_up_save[self.count_save] = self.depths_cov_up[i].clone().cpu()
                                            self.images_up_save[self.count_save] = self.images[i,[2,1,0],...].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                                            
                                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                                        self.count_save += 1
                                marg_result.insert(X(i), self.cur_result.atPose3(X(i)))
                            
                            marg_target = self.cur_target[marg_idx]
                            marg_weight = self.cur_weight[marg_idx]
                            marg_eta = self.cur_eta[0:marg_t1-marg_t0]
    
                            bacore = droid_backends.BACore()
                            bacore.init(self.poses, self.disps, self.intrinsics[0], torch.zeros_like(self.disps_sens),
                                marg_target, marg_weight, marg_eta, marg_ii, marg_jj, marg_t0, marg_t1, itrs, lm, ep, motion_only)
                            H = torch.zeros([(marg_t1-marg_t0)*6,(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                            v = torch.zeros([(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                            
                            bacore.hessian(H,v)
                            
                            
                            for i in range(6): H[i,i] += 0.00025  # for stability

                            # Hg,vg = BA2GTSAM(H,v,self.Tbc)
                            Hgg = gtsam.BA2GTSAM(H,v,self.Tbc)
                            Hg = Hgg[0:(marg_t1-marg_t0)*6,0:(marg_t1-marg_t0)*6]
                            vg = Hgg[0:(marg_t1-marg_t0)*6,  (marg_t1-marg_t0)*6]
                            vis_factor = CustomHessianFactor(marg_result,Hg,vg)
                            
                            graph.push_back(vis_factor)

                        for i in range(self.last_t0,marg_t1):
                            if i < t0:
                                if X(i) not in marg_paras:
                                    marg_paras.append(X(i))
                                if not self.ignore_imu:
                                    marg_paras.append(V(i))
                                    marg_paras.append(B(i))
                                    graph.push_back(gtsam.gtsam.CombinedImuFactor(\
                                                X(i),V(i),X(i+1),V(i+1),B(i),B(i+1),\
                                                self.state.preintegrations[i]))
                                if self.gnss_init_t1 > 0:
                                    if self.state.gnss_valid[i]:
                                        p = np.matmul(trans.Cen(self.ten0).T, self.state.gnss_position[i] - self.ten0)
                                        n0pbg = self.state.wTbs[i].rotation().rotate(self.tbg)
                                        p = p - n0pbg
                                        gnss_factor = gtsam.GPSFactor(X(i), p,\
                                                      gtsam.noiseModel.Robust.Create(\
                                                      gtsam.noiseModel.mEstimator.Cauchy(0.08),\
                                          gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,5.0]))))
                                        graph.push_back(gnss_factor)
                                if self.state.odo_valid[i]:
                                    vb = self.state.odo_vel[i]
                                    odo_factor = gtsam.VelFactor(X(i),V(i),vb,gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0,2.0,2.0])))
                                    graph.push_back(odo_factor)
                        
                        keys = self.prior_factor_map.keys()
                        for i in sorted(keys):
                            if i < t0:
                                for iii in range(len(self.prior_factor_map[i])):
                                    graph.push_back(self.prior_factor_map[i][iii])
                            del self.prior_factor_map[i]
                        if not self.marg_factor == None:
                            graph.push_back(self.marg_factor)

                        self.marg_factor = gtsam.marginalizeOut(graph,self.cur_result,marg_paras)

                        # covariance inflation of IMU biases
                        if self.reinit == True:
                            all_keys = self.marg_factor.keys()
                            for i in range(len(all_keys)):
                                if all_keys[i] == B(t0):
                                    all_keys[i] = B(0)
                            graph = gtsam.NonlinearFactorGraph()
                            graph.push_back(self.marg_factor.rekey(all_keys))
                            b_l = gtsam.BetweenFactorConstantBias(B(0),B(t0),gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])),\
                                                                  gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma))
                            graph.push_back(b_l)
                            result_tmp = self.marg_factor.linearizationPoint()
                            result_tmp.insert(B(0),result_tmp.atConstantBias(B(t0)))
                            self.marg_factor = gtsam.marginalizeOut(graph,result_tmp,[B(0)])
                            self.reinit = False

                    self.last_t0 = t0
                    self.last_t1 = t1

                """ optimization """
                H = torch.zeros([(t1-t0)*6,(t1-t0)*6],dtype=torch.float64,device='cpu')
                v = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu')
                dx = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu') 

                bacore = droid_backends.BACore()
                active_index    = torch.logical_and(ii>=t0,jj>=t0)
                self.cur_ii     = ii[active_index]
                self.cur_jj     = jj[active_index]
                self.cur_target = target[active_index]
                self.cur_weight = weight[active_index]
                self.cur_eta    = eta[(t0-ii.min().item()):]
                # TTD 2024/10/22
                self.middleware_dict = {'ii': self.cur_ii, 'jj': self.cur_jj, 't0': t0, 't1': t1}

                bacore.init(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                    self.cur_target, self.cur_weight, self.cur_eta, self.cur_ii, self.cur_jj, t0, t1, itrs, lm, ep, motion_only)
                
                # TTD 2024/09/27
                Q, E = bacore.hessian_expand(torch.zeros_like(H), torch.zeros_like(v))
                
                self.cur_graph = gtsam.NonlinearFactorGraph()
                params = gtsam.LevenbergMarquardtParams()#;params.setMaxIterations(1)

                # imu factor
                if not self.ignore_imu:
                    for i in range(t0,t1):
                        if i > t0:
                            imu_factor = gtsam.gtsam.CombinedImuFactor(\
                                X(i-1),V(i-1),X(i),V(i),B(i-1),B(i),\
                                self.state.preintegrations[i-1])
                            self.cur_graph.add(imu_factor)

                # prior factor
                keys = self.prior_factor_map.keys()
                for i in sorted(keys):
                    if i >= t0 and i < t1:
                        for iii in range(len(self.prior_factor_map[i])):
                            self.cur_graph.push_back(self.prior_factor_map[i][iii])
                
                # marginalization factor
                if self.marg_factor is not None:
                    self.cur_graph.push_back(self.marg_factor)

                # GNSS factor
                if self.gnss_init_t1 > 0:
                    for i in range(t0,t1):
                        if self.state.gnss_valid[i]:
                            p = np.matmul(trans.Cen(self.ten0).T, self.state.gnss_position[i] - self.ten0)
                            n0pbg = self.state.wTbs[i].rotation().rotate(self.tbg)
                            p = p - n0pbg
                            gnss_factor = gtsam.GPSFactor(X(i), p,\
                                          gtsam.noiseModel.Robust.Create(\
                                                      gtsam.noiseModel.mEstimator.Cauchy(0.08),\
                                          gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,5.0]))))
                            self.cur_graph.push_back(gnss_factor)
                
                # Odo factor
                for i in range(t0,t1):
                    if self.state.odo_valid[i]:
                        vb = self.state.odo_vel[i]
                        odo_factor = gtsam.VelFactor(X(i),V(i),vb,gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0,2.0,2.0])))
                        self.cur_graph.push_back(odo_factor)

                """ multi-sensor DBA iterations """
                for iter in range(2):
                    if iter > 0:
                        self.cur_graph.resize(self.cur_graph.size()-1)
                    
                    bacore.hessian(H,v) # camera frame
                    
                    Hgg = gtsam.BA2GTSAM(H,v,self.Tbc)
                    Hg = Hgg[0:(t1-t0)*6,0:(t1-t0)*6]
                    vg = Hgg[0:(t1-t0)*6,(t1-t0)*6]

                    initial = gtsam.Values()
                    for i in range(t0,t1):
                        initial.insert(X(i), self.state.wTbs[i]) # the indice need to be handled
                    initial_vis = copy.deepcopy(initial)
                    vis_factor = CustomHessianFactor(initial_vis,Hg,vg)
                    self.cur_graph.push_back(vis_factor)
                    
                    if not self.ignore_imu:
                        for i in range(t0,t1):
                            initial.insert(B(i),self.state.bs[i])
                            initial.insert(V(i),self.state.vs[i])

                    optimizer = gtsam.LevenbergMarquardtOptimizer(self.cur_graph, initial, params)
                    self.cur_result = optimizer.optimize()

                    # retraction and depth update
                    for i in range(t0,t1):
                        p0 = initial.atPose3(X(i))
                        p1 = self.cur_result.atPose3(X(i))
                        xi = gtsam.Pose3.Logmap(p0.inverse()*p1)
                        dx[(i-t0)*6:(i-t0)*6+6] = torch.tensor(xi)
                        if not self.ignore_imu:
                            self.state.bs[i] = self.cur_result.atConstantBias(B(i))
                            self.state.vs[i] = self.cur_result.atVector(V(i))
                        self.state.wTbs[i] = self.cur_result.atPose3(X(i))
                    dx = torch.tensor(gtsam.GTSAM2BA(dx,self.Tbc))
                    dx_dz = bacore.retract(dx)
                    
                # FIXME: Check this carefully.
                '''
                H.shape: torch.Size([78, 78])
                v.shape: torch.Size([78])
                Q.shape: torch.Size([10, 3311]) | Q.shape[0]==torch.unique(ii).shape[0]
                E.shape: torch.Size([19, 6, 3311])
                ii.shape, jj.shape: torch.Size([62])
                t0, t1 =depths_cov 2, 14
                '''
                
                if Q is not None and Q.shape[0]==torch.unique(ii).shape[0]:
                    # 13x15 → 13x6
                    linear_factor_graph = self.cur_graph.linearize(self.cur_result)
                    H_linear_all, _ = linear_factor_graph.hessian()
                    H_linear = H_linear_all[-H.shape[0]:, -H.shape[0]:]
                    hw_dsf = self.height_dsf * self.width_dsf
                    try:
                        self.compute_covariances(H_linear, Q, E, hw_dsf, ii, jj, t0, t1)
                    except:
                        pass
                del bacore
            self.disps.clamp_(min=0.001)
    
    
    def compute_covariances(self, H, Q, E, hw_dsf, ii, jj, t0, t1):
        '''
        N: kf1-kf0
        Let's explore what are H, v, Q, E, w.
        H.shape = (78, 78)
        Q.shape = (4, 43*77) should be (13, 43*77)
        E.shape = (7, 6, 43*77)
        w.shape = (10, 43*77)
        ii, jj: (66), (66)
        t0, t1 = 4, 17
        pose_keys: length=7
        '''
        N = t1-t0
        L = None
        # H /= N
        try: L = torch.linalg.cholesky(torch.as_tensor(H, device=Q.device, dtype=torch.float)) # from double to float...
        except: pass
        if L is not None:
            identity = torch.eye(L.shape[0], device=L.device)
            L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
            if torch.isnan(L_inv).any(): raise ValueError("NANs in L_inv!!")
            sigma_gg = L_inv.transpose(-2,-1) @ L_inv
            P = N
            D = L.shape[0] // P
            assert D == 6
            sigma_gg = sigma_gg.view(P, D, P, D).permute(0,2,1,3) # P x P x D x D
            sigma_g = torch.diagonal(sigma_gg, dim1=0, dim2=1).permute(2,0,1).view(P, D, D) # P x D x D
            Ei = E[:P]
            Ejz = E[P:P+ii.shape[0]]
            M = Ejz.shape[0]
            assert M == ii.shape[0]
            kx, kk = torch.unique(ii, return_inverse=True)
            K = kx.shape[0]
            min_ii_jj = min(ii.min(),jj.min())
            Ej = torch.zeros(K, K, D, hw_dsf, device=ii.device)
            Ej[jj - min_ii_jj, ii - min_ii_jj] = Ejz
            Ej = Ej[t0-min_ii_jj:t1-min_ii_jj].view(P,K,D,hw_dsf) # Keep only the keyframes we are optimizing over, and remove the fixed ones, but add all the depth-maps...
            
            Ej[range(P), t0-min_ii_jj:t1-min_ii_jj, :, :] = Ei[range(P), :, :]
                
            E_sum = Ej
            E_sum = E_sum.view(P, K, D, hw_dsf)
            E_sum = E_sum.permute(0,2,1,3).reshape(P*D, K*hw_dsf)
            Q_ = Q.view(K*hw_dsf,1)
            F = torch.matmul(Q_ * E_sum.t(), L_inv) # K*HW x D*P
            F2 = torch.pow(F, 2)
            delta_cov = F2.sum(dim=-1) # K*HW
            z_cov = Q_.squeeze() + delta_cov # K*HW
            z_cov = z_cov.view(K, self.height_dsf, self.width_dsf) 
            # self.video.upsample(torch.unique(self.ii), upmask)
            self.depths_cov[kx] = z_cov / (self.disps[kx]**4+1e-6)
    
    
    def ba_uncertainty_dbaf(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """
        with self.get_lock():
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # 1) visual-only BA
            # 2) multi-sensor BA
            if (not self.imu_enabled) and (not self.visual_only_init): 
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                    target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
                '''
                ii, jj是对video.poses, disps等的索引;
                确实, 这里肯定会落后tracker;
                在 min(ii_last, jj_last)~min(ii, jj) 之间搞。
                '''
                for i in range(self.last_t0, min(ii.min().item(), jj.min().item())):
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save] = self.tstamp[i].clone().cpu()
                        self.disps_save[self.count_save] = self.disps[i].clone().cpu()
                        self.poses_save[self.count_save] = self.poses[i].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save] = self.disps_up[i].clone().cpu()
                            self.depths_cov_up_save[self.count_save] = self.depths_cov_up[i].clone().cpu()
                            self.images_up_save[self.count_save] = self.images[i,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                            # self.images_up_save[self.count_save] = self.images[i,:].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        
                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                        # self.images_save[self.count_save] = self.images[i,:,::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.count_save += 1
                        # i in [self.last_t0, min(ii.min().item(), jj.min().item())] 对应 []
                        # NOTE: "min(ii.min().item(), jj.min().item()) 对应 self.count_save"
                
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
                # TTD 2024/10/15
                # Dangerous Option, 上面的那些都是存了就不改了，俺这边存了还会改的;
                for i_temp in range(min(ii.min().item(), jj.min().item()), t1):
                    bias = i_temp-min(ii.min().item(), jj.min().item())
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save+bias] = self.tstamp[i_temp].clone().cpu()
                        self.disps_save[self.count_save+bias] = self.disps[i_temp].clone().cpu()
                        self.poses_save[self.count_save+bias] = self.poses[i_temp].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save+bias] = self.disps_up[i_temp].clone().cpu()
                            self.depths_cov_up_save[self.count_save+bias] = self.depths_cov_up[i_temp].clone().cpu()
                            self.images_up_save[self.count_save+bias] = self.images[i_temp,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.images_save[self.count_save+bias] = self.images[i_temp,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                self.count_save_bias = t1 - min(ii.min().item(), jj.min().item())
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
                
                self.last_t0 = min(ii.min().item(), jj.min().item())
                self.last_t1 = t1
            
            else:
                last_last_t0 = self.last_t0
                t0 = min(ii.min().item(), jj.min().item())
                Q, E = None, None
                
                """ marginalization """
                if self.last_t1!=t1 or self.last_t0 != t0:
                    if self.last_t0 >= t0:
                        t0 = self.last_t0
                    else:
                        marg_paras = []
                        # Construct a temporary factor graph (related to the old states) to obtain the marginalization information
                        graph = gtsam.NonlinearFactorGraph()
                        marg_idx = torch.logical_and(torch.greater_equal(self.cur_ii,self.last_t0),\
                                                    torch.less(self.cur_ii,t0))
                        marg_idx2 = torch.logical_and(torch.less(self.cur_ii,self.last_t1-2),\
                                                     torch.less(self.cur_jj,self.last_t1-2))
                        marg_idx = torch.logical_and(marg_idx,marg_idx2)

                        marg_ii = self.cur_ii[marg_idx]
                        marg_jj = self.cur_jj[marg_idx]
                        marg_t0 = self.last_t0 
                        marg_t1 = t0 + 1
                        if len(marg_ii) > 0:
                            marg_t0 = self.last_t0 
                            marg_t1 = torch.max(marg_jj).item()+1
                            marg_result = gtsam.Values()
                            for i in range(self.last_t0,marg_t1): # 就存一个优化完的结果？
                                if i < t0:
                                    marg_paras.append(X(i))
                                    if self.save_pkl:
                                        # save marginalized results
                                        self.tstamp_save[self.count_save] = self.tstamp[i].clone().cpu()
                                        self.disps_save[self.count_save] = self.disps[i].clone().cpu()
                                        self.poses_save[self.count_save] = self.poses[i].clone().cpu()
                                        if self.upsample_flag:
                                            self.disps_up_save[self.count_save] = self.disps_up[i].clone().cpu()
                                            self.depths_cov_up_save[self.count_save] = self.depths_cov_up[i].clone().cpu()
                                            self.images_up_save[self.count_save] = self.images[i,[2,1,0],...].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                                            
                                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                                        self.count_save += 1
                                marg_result.insert(X(i), self.cur_result.atPose3(X(i)))
                            
                            marg_target = self.cur_target[marg_idx]
                            marg_weight = self.cur_weight[marg_idx]
                            marg_eta = self.cur_eta[0:marg_t1-marg_t0]
    
                            bacore = droid_backends.BACore()
                            bacore.init(self.poses, self.disps, self.intrinsics[0], torch.zeros_like(self.disps_sens),
                                marg_target, marg_weight, marg_eta, marg_ii, marg_jj, marg_t0, marg_t1, itrs, lm, ep, motion_only)
                            H = torch.zeros([(marg_t1-marg_t0)*6,(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                            v = torch.zeros([(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                            
                            bacore.hessian(H,v)
                            
                            for i in range(6): H[i,i] += 0.00025  # for stability

                            # Hg,vg = BA2GTSAM(H,v,self.Tbc)
                            Hgg = gtsam.BA2GTSAM(H,v,self.Tbc)
                            Hg = Hgg[0:(marg_t1-marg_t0)*6,0:(marg_t1-marg_t0)*6]
                            vg = Hgg[0:(marg_t1-marg_t0)*6,  (marg_t1-marg_t0)*6]
                            vis_factor = CustomHessianFactor(marg_result,Hg,vg)
                            
                            graph.push_back(vis_factor)

                        for i in range(self.last_t0,marg_t1):
                            if i < t0:
                                if X(i) not in marg_paras:
                                    marg_paras.append(X(i))
                        
                        keys = self.prior_factor_map.keys()
                        for i in sorted(keys):
                            if i < t0:
                                for iii in range(len(self.prior_factor_map[i])):
                                    graph.push_back(self.prior_factor_map[i][iii])
                            del self.prior_factor_map[i]
                        if not self.marg_factor == None:
                            graph.push_back(self.marg_factor)

                        self.marg_factor = gtsam.marginalizeOut(graph,self.cur_result,marg_paras)

                        # covariance inflation of IMU biases
                        if self.reinit == True:
                            all_keys = self.marg_factor.keys()
                            for i in range(len(all_keys)):
                                if all_keys[i] == B(t0):
                                    all_keys[i] = B(0)
                            graph = gtsam.NonlinearFactorGraph()
                            graph.push_back(self.marg_factor.rekey(all_keys))
                            b_l = gtsam.BetweenFactorConstantBias(B(0),B(t0),gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])),\
                                                                  gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma))
                            graph.push_back(b_l)
                            result_tmp = self.marg_factor.linearizationPoint()
                            result_tmp.insert(B(0),result_tmp.atConstantBias(B(t0)))
                            self.marg_factor = gtsam.marginalizeOut(graph,result_tmp,[B(0)])
                            self.reinit = False

                    self.last_t0 = t0
                    self.last_t1 = t1

                """ optimization """
                H = torch.zeros([(t1-t0)*6,(t1-t0)*6],dtype=torch.float64,device='cpu')
                v = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu')
                dx = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu') 

                bacore = droid_backends.BACore()
                active_index    = torch.logical_and(ii>=t0,jj>=t0)
                self.cur_ii     = ii[active_index]
                self.cur_jj     = jj[active_index]
                self.cur_target = target[active_index]
                self.cur_weight = weight[active_index]
                self.cur_eta    = eta[(t0-ii.min().item()):]
                # TTD 2024/10/22
                self.middleware_dict = {'ii': self.cur_ii, 'jj': self.cur_jj, 't0': t0, 't1': t1}

                bacore.init(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                    self.cur_target, self.cur_weight, self.cur_eta, self.cur_ii, self.cur_jj, t0, t1, itrs, lm, ep, motion_only)
                
                # TTD 2024/09/27
                Q, E = bacore.hessian_expand(torch.zeros_like(H), torch.zeros_like(v))
                self.cur_graph = gtsam.NonlinearFactorGraph()
                params = gtsam.LevenbergMarquardtParams()#;params.setMaxIterations(1)
                
                # prior factor
                keys = self.prior_factor_map.keys()
                for i in sorted(keys):
                    if i >= t0 and i < t1:
                        for iii in range(len(self.prior_factor_map[i])):
                            self.cur_graph.push_back(self.prior_factor_map[i][iii])
                
                # marginalization factor
                if self.marg_factor is not None:
                    self.cur_graph.push_back(self.marg_factor)

                
                """ multi-sensor DBA iterations """
                for iter in range(2):
                    if iter > 0:
                        self.cur_graph.resize(self.cur_graph.size()-1)
                    
                    bacore.hessian(H,v) # camera frame
                    
                    Hgg = gtsam.BA2GTSAM(H,v,self.Tbc)
                    Hg = Hgg[0:(t1-t0)*6,0:(t1-t0)*6]
                    vg = Hgg[0:(t1-t0)*6,(t1-t0)*6]

                    initial = gtsam.Values()
                    for i in range(t0,t1):
                        initial.insert(X(i), self.state.wTbs[i]) # the indice need to be handled
                    initial_vis = copy.deepcopy(initial)
                    vis_factor = CustomHessianFactor(initial_vis,Hg,vg)
                    self.cur_graph.push_back(vis_factor)
                    
                    
                    optimizer = gtsam.LevenbergMarquardtOptimizer(self.cur_graph, initial, params)
                    self.cur_result = optimizer.optimize()

                    # retraction and depth update
                    for i in range(t0,t1):
                        p0 = initial.atPose3(X(i))
                        p1 = self.cur_result.atPose3(X(i))
                        xi = gtsam.Pose3.Logmap(p0.inverse()*p1)
                        dx[(i-t0)*6:(i-t0)*6+6] = torch.tensor(xi)
                        self.state.wTbs[i] = self.cur_result.atPose3(X(i))
                    dx = torch.tensor(gtsam.GTSAM2BA(dx,self.Tbc))
                    dx_dz = bacore.retract(dx)
                    
                # FIXME: Check this carefully.
                '''
                H.shape: torch.Size([78, 78])
                v.shape: torch.Size([78])
                Q.shape: torch.Size([10, 3311]) | Q.shape[0]==torch.unique(ii).shape[0]
                E.shape: torch.Size([19, 6, 3311])
                ii.shape, jj.shape: torch.Size([62])
                t0, t1 =depths_cov 2, 14
                '''
                
                if Q is not None and Q.shape[0]==torch.unique(ii).shape[0]:
                    # 13x15 → 13x6
                    linear_factor_graph = self.cur_graph.linearize(self.cur_result)
                    H_linear_all, _ = linear_factor_graph.hessian()
                    H_linear = H_linear_all[-H.shape[0]:, -H.shape[0]:]
                    hw_dsf = self.height_dsf * self.width_dsf
                    try:
                        self.compute_covariances(H_linear, Q, E, hw_dsf, ii, jj, t0, t1)
                    except:
                        pass
                del bacore
            self.disps.clamp_(min=0.001)
    
    
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -    
    def ba_uncertainty_nerfslam(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) 
        现在的问题可能是 “索引错误” or “gtsam” 版本错误; 
        """
        with self.get_lock():
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # 1) visual-only  BA
            if False and not self.visual_only_init: 
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                    target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)
                '''
                ii, jj是对video.poses, disps等的索引;
                确实, 这里肯定会落后tracker;
                在 min(ii_last, jj_last)~min(ii, jj) 之间搞。
                '''
                for i in range(self.last_t0, min(ii.min().item(), jj.min().item())):
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save] = self.tstamp[i].clone().cpu()
                        self.disps_save[self.count_save] = self.disps[i].clone().cpu()
                        self.poses_save[self.count_save] = self.poses[i].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save] = self.disps_up[i].clone().cpu()
                            self.depths_cov_up_save[self.count_save] = self.depths_cov_up[i].clone().cpu()
                            self.images_up_save[self.count_save] = self.images[i,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                            # self.images_up_save[self.count_save] = self.images[i,:].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        
                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                        # self.images_save[self.count_save] = self.images[i,:,::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.count_save += 1
                        # i in [self.last_t0, min(ii.min().item(), jj.min().item())] 对应 []
                        # NOTE: "min(ii.min().item(), jj.min().item()) 对应 self.count_save"
                
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
                # TTD 2024/10/15
                # Dangerous Option, 上面的那些都是存了就不改了，俺这边存了还会改的;
                for i_temp in range(min(ii.min().item(), jj.min().item()), t1):
                    bias = i_temp-min(ii.min().item(), jj.min().item())
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save+bias] = self.tstamp[i_temp].clone().cpu()
                        self.disps_save[self.count_save+bias] = self.disps[i_temp].clone().cpu()
                        self.poses_save[self.count_save+bias] = self.poses[i_temp].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save+bias] = self.disps_up[i_temp].clone().cpu()
                            self.depths_cov_up_save[self.count_save+bias] = self.depths_cov_up[i_temp].clone().cpu()
                            self.images_up_save[self.count_save+bias] = self.images[i_temp,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.images_save[self.count_save+bias] = self.images[i_temp,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                self.count_save_bias = t1 - min(ii.min().item(), jj.min().item())
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
                
                self.last_t0 = min(ii.min().item(), jj.min().item())
                self.last_t1 = t1
                
                self.disps.clamp_(0.001)
            
            # 2) multi-sensor BA
            else:
                N = t1 - t0
                kx = torch.unique(ii)

                kf_ids = [i+t0 for i in range(t1 - t0)]
                Xii = np.array([X(kf_id) for kf_id in kf_ids])
                
                if not hasattr(self, 'first_kf_id') or self.first_kf_id==t0:
                    self.first_kf_id = t0
                    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])) # rot, trans
                    pose_prior = PriorFactorPose3(X(t0), gtsam.Pose3(np.eye(4)), pose_noise)
                    prior_graph = NonlinearFactorGraph()
                    prior_graph.push_back(pose_prior)
                    initial_priors = prior_graph
                else:
                    initial_priors = None
                
                for _ in range(itrs):
                    x0 = Values()
                    linear_factor_graph = GaussianFactorGraph()
                    for i in range(N):
                        kf_id = i + t0
                        x0.insert(Xii[i], gtsam.Pose3(SE3(self.poses[kf_id]).inv().matrix().cpu().numpy()))
                # Check here to judge whether the index is error associated.
                H, v, Q, E, w = droid_backends_nerfslam.reduced_camera_matrix(
                                                                    self.poses,
                                                                    SE3(self.poses).inv().data,
                                                                    self.disps,
                                                                    self.intrinsics[0],
                                                                    torch.tensor([0., 0., 0., 0., 0., 0., 1.], dtype=torch.float32, device=self.poses.device),
                                                                    self.disps,
                                                                    target,
                                                                    weight,
                                                                    eta, 
                                                                    ii, jj, t0, t1)
                
                vision_factors = GaussianFactorGraph()
                H = torch.nn.functional.unfold(H[None,None], (6, 6), stride=6).permute(2,0,1).view(N, N, 6, 6)
                v = torch.nn.functional.unfold(v[None,None], (6, 1), stride=6).permute(2,0,1).view(N, 6)
                H[range(N), range(N)] /= N
                v[:] /= N
                upper_triangular_indices = torch.triu_indices(N, N)
                for i, j in zip(upper_triangular_indices[0], upper_triangular_indices[1]):
                    if i == j:
                        vision_factors.add(HessianFactor(Xii[i], H[i, i].cpu().numpy(), v[i].cpu().numpy(), 0.0))
                    else:
                        vision_factors.add(HessianFactor(Xii[i], Xii[j], H[i, i].cpu().numpy(), H[i, j].cpu().numpy(), v[i].cpu().numpy(), H[j, j].cpu().numpy(), v[j].cpu().numpy(), 0.0))
                
                linear_factor_graph.push_back(vision_factors)
                if initial_priors is not None: linear_factor_graph.push_back(initial_priors.linearize(x0))
                
                gtsam_delta = linear_factor_graph.optimize()
                last_state = x0.retract(gtsam_delta)    
                poses = gtsam.utilities.allPose3s(last_state)    
                pose_keys = poses.keys()
                
                for i, key in enumerate(pose_keys):
                    kf_idx  = gtsam.Symbol(key).index()
                    self.poses[kf_idx] = SE3(gtsam_pose_to_torch(poses.atPose3(key), device=self.poses.device, dtype=torch.float32)).inv().data
                
                xi_delta = torch.as_tensor(gtsam_delta.vector(), device=self.poses.device, dtype=torch.float).view(-1, 6)

                droid_backends_nerfslam.solve_depth(xi_delta, self.disps, Q, E, w, ii, jj, t0, t1)
                self.disps.clamp_(0.001)
                
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  
                # Hi! Let's update video.(disps, poses) and video.(disps_save, poses_save)
                '''
                ii, jj是对video.poses, disps等的索引;
                确实, 这里肯定会落后tracker;
                在 min(ii_last, jj_last)~min(ii, jj) 之间搞。
                '''
                for i in range(self.last_t0, min(ii.min().item(), jj.min().item())):
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save] = self.tstamp[i].clone().cpu()
                        self.disps_save[self.count_save] = self.disps[i].clone().cpu()
                        self.poses_save[self.count_save] = self.poses[i].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save] = self.disps_up[i].clone().cpu()
                            self.depths_cov_up_save[self.count_save] = self.depths_cov_up[i].clone().cpu()
                            self.images_up_save[self.count_save] = self.images[i,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                            # self.images_up_save[self.count_save] = self.images[i,:].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        
                        self.images_save[self.count_save] = self.images[i,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                        # self.images_save[self.count_save] = self.images[i,:,::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.count_save += 1
                        # i in [self.last_t0, min(ii.min().item(), jj.min().item())] 对应 []
                        # NOTE: "min(ii.min().item(), jj.min().item()) 对应 self.count_save"
                            
                # TTD 2024/10/15
                # Dangerous Option, 上面的那些都是存了就不改了，俺这边存了还会改的;
                for i_temp in range(min(ii.min().item(), jj.min().item()), t1):
                    bias = i_temp-min(ii.min().item(), jj.min().item())
                    if self.save_pkl:
                        # save marginalized results
                        self.tstamp_save[self.count_save+bias] = self.tstamp[i_temp].clone().cpu()
                        self.disps_save[self.count_save+bias] = self.disps[i_temp].clone().cpu()
                        self.poses_save[self.count_save+bias] = self.poses[i_temp].clone().cpu()
                        if self.upsample_flag:
                            self.disps_up_save[self.count_save+bias] = self.disps_up[i_temp].clone().cpu()
                            self.depths_cov_up_save[self.count_save+bias] = self.depths_cov_up[i_temp].clone().cpu()
                            self.images_up_save[self.count_save+bias] = self.images[i_temp,[2,1,0]].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?
                        self.images_save[self.count_save+bias] = self.images[i_temp,[2,1,0],::8,::8].permute(1,2,0).cpu() / 255.0 # might be "3::8, 3::8"?    
                self.count_save_bias = t1 - min(ii.min().item(), jj.min().item())
                # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
                
                self.last_t0 = min(ii.min().item(), jj.min().item())
                self.last_t1 = t1
                
                compute_covariances = True
                if compute_covariances:
                    hw_dsf = self.disps.shape[-2] * self.disps.shape[-1] 
                    self.compute_covariances_nerfslam(linear_factor_graph, ii, jj, E, Q, hw_dsf, t0, t1, pose_keys)

        
    def compute_covariances_nerfslam(self, linear_factor_graph, ii, jj, E, Q, HW, kf0, kf1, pose_keys):
        '''
        N: kf1-kf0
        Let's explore what are H, v, Q, E, w.
        H.shape = (42, 42)
        v.shape = (42)
        Q.shape = (10, 43*77)
        E.shape = (57, 6, 43*77)
        w.shape = (10, 43*77)
        ii, jj: (50), (50)
        kf0=14, kf1=21
        pose_keys: length=7
        '''
        H, v = linear_factor_graph.hessian()
        L = None
        N = kf1 - kf0
        try:
            L = torch.linalg.cholesky(torch.as_tensor(H, device=self.device, dtype=torch.float))# from double to float...
        except:
            pass
        if L is not None:
            identity = torch.eye(L.shape[0], device=L.device) # L has shape (PD,PD) 
            L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
            if torch.isnan(L_inv).any():
                print("NANs in L_inv!!")
                raise
            sigma_gg = L_inv.transpose(-2,-1) @ L_inv
            P = N
            D = L.shape[0] // P
            assert D == 6
            sigma_gg = sigma_gg.view(P, D, P, D).permute(0,2,1,3) # P x P x D x D
            sigma_g = torch.diagonal(sigma_gg, dim1=0, dim2=1).permute(2,0,1).view(P, D, D) # P x D x D

            Ei = E[:P]
            Ejz = E[P:P+ii.shape[0]]
            M = Ejz.shape[0]
            assert M == ii.shape[0]
            kx, kk = torch.unique(ii, return_inverse=True)
            K = kx.shape[0]
            
            # min_ii_jj = min(ii.min(),jj.min())
            # Dangerous Option. 2024/11/03
            min_ii_jj = max(ii.min(),jj.min())
            
            '''
            N: 8
            H.shape: (48, 48)
            L.shape: (48, 48)
            ii, jj.shape: (55)
            E.shape: (63, 6, 3311)
            '''
            Ej = torch.zeros(K, K, D, HW, device=Ejz.device)
            Ej[(jj - min_ii_jj), (ii - min_ii_jj)] = Ejz
            
            Ej = Ej[kf0-min_ii_jj:kf1-min_ii_jj].view(P,K,D,HW) # Keep only the keyframes we are optimizing over, and remove the fixed ones, but add all the depth-maps...
            
            Ej[range(P), kf0-min_ii_jj:kf1-min_ii_jj, :, :] = Ei[range(P), :, :]
                
            E_sum = Ej
            E_sum = E_sum.view(P, K, D, HW)
            E_sum = E_sum.permute(0,2,1,3).reshape(P*D, K*HW)
            Q_ = Q.view(K*HW,1)
            F = torch.matmul(Q_ * E_sum.t(), L_inv) # K*HW x D*P
            F2 = torch.pow(F, 2)
            delta_cov = F2.sum(dim=-1) # K*HW
            z_cov = Q_.squeeze() + delta_cov # K*HW
            z_cov = z_cov.view(K, self.height_dsf, self.width_dsf) 
            
            self.depths_cov[kx] = z_cov / self.disps[kx]**4
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -