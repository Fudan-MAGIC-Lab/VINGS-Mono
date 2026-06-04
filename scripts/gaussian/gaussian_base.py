import torch
import torch.nn as nn
import numpy as np
import random
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization import SparseGaussianAdam
from lietorch import SE3, SO3, RxSO3
import torch.nn.functional
from gaussian.cameras import get_camera
from gaussian.tf import TFer
from gaussian.gaussian_utils import distCUDA2, weighting_grad, get_gaussian_mask
from gaussian.general_utils import inverse_sigmoid
from abc import ABCMeta, abstractmethod
from gaussian.loss_utils import get_loss, get_pixel_mask, l1_loss
from gaussian.normal_utils import depth_propagate_normal
try:
    from gaussian.vis_utils import vis_rgbdnua, load_ply, calc_psnr
except ModuleNotFoundError as exc:
    _vis_utils_import_error = exc

    def _missing_vis_utils(*args, **kwargs):
        raise ModuleNotFoundError(
            "gaussian.vis_utils optional dependencies are missing (e.g., open3d)."
        ) from _vis_utils_import_error

    vis_rgbdnua = _missing_vis_utils
    load_ply = _missing_vis_utils
    calc_psnr = _missing_vis_utils
# from utils.gtsam_utils import matrix_to_tq
import copy
import time
import os
import cv2



class GaussianBase:
    def __init__(self, cfg):
        self.cfg = cfg

        self.tfer = TFer(cfg)
        self.dtype = torch.float32
        self.device = self.tfer.device

        self._xyz = torch.empty(0)
        self._rgb = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._global_scores = torch.empty(0) # Importance Score & Error Score.
        self._local_scores  = torch.empty(0)   # Importance Score & Error Score during training iters.
        self._stable_mask   = torch.empty(0)
        self._globalkf_id         = torch.empty(0)
        self._globalkf_max_scores = torch.empty(0)
        
        self.activate_dict = {'_scaling': torch.exp,
                              '_opacity': torch.sigmoid,
                              '_rotation': torch.nn.functional.normalize,
                              'inv_scaling': torch.log,
                              'inv_opacity': inverse_sigmoid}

        self.initialized_state = False
    
    def setup_optimizer(self):
        cfg = self.cfg
        lr_args = cfg['training_args']['lr']
        l = [
            {'params': [self._xyz], 'lr': lr_args['_xyz_lr'], "name": "_xyz"},
            {'params': [self._rgb], 'lr': lr_args['_rgb_lr'], "name": "_rgb"},
            {'params': [self._opacity], 'lr': lr_args['_opacity_lr'], "name": "_opacity"},
            {'params': [self._scaling], 'lr': lr_args['_scaling_lr'], "name": "_scaling"},
            {'params': [self._rotation], 'lr': lr_args['_rotation_lr'], "name": "_rotation"}
        ]
        self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
    
    def get_property(self, name):
        if name == '_xyz': y = self._xyz
        elif name == '_opacity': y = self.activate_dict['_opacity'](self._opacity)
        elif name == '_rotation': y = self.activate_dict['_rotation'](self._rotation)
        elif name == '_scaling': y = self.activate_dict['_scaling'](self._scaling)
        elif name == '_rgb': y = self._rgb
        elif name == '_zeros': y = torch.zeros_like(self._xyz[:, :2]).contiguous().requires_grad_(True)
        else: raise ValueError("Invalid property name: {}".format(name))
        return y

    def cat_tensors_to_optimizer(self, optimizer, tensors_dict):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group['name'] in ['_cold2w_mul_w2cnews', '_dnew_divide_dolds', '_c0old_to_c0news']: continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_tensors_from_optimizer(self, optimizer, prune_mask):
        valid_mask = ~prune_mask
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group['name'] in ['_cold2w_mul_w2cnews', '_dnew_divide_dolds', '_c0old_to_c0news']: continue
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    @abstractmethod
    def init_first_frame(self, batch):
        pass
        
    @abstractmethod
    def add_new_frame(self, processed_dict):
        pass
    
    def judge_new_frame(self, processed_dict):
        new_id_list = processed_dict['viz_out_idx_to_f_idx'].tolist()
        history_list = self.history_list
        exist_list = [(item in history_list) for item in new_id_list]
        if all(exist_list):
            return False, None
        else:
            new_id = None
            for e_id in range(len(exist_list)):
                if not exist_list[e_id]:
                    new_id = e_id
                    break
            new_added_dict = {}
            self.history_list.append(new_id_list[new_id])
            new_added_dict['pose'] = processed_dict['poses'][new_id]
            new_added_dict['idx'] = new_id_list[new_id]
            new_added_dict['depth'] = processed_dict['depths'][new_id]
            new_added_dict['image'] = processed_dict['images'][new_id]
            new_added_dict['cov'] = processed_dict['depths_cov'][new_id]
            new_added_dict['intrinsic'] = processed_dict['intrinsic']
            return True, new_added_dict

    def render_raw(self, w2c, intrinsic_dict, unopt_gaussian_mask = None):
        # Copy 2DGS.
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # (1) Setup raster_settings.
        camera = get_camera(w2c, intrinsic_dict)
        
        
        # TTD 2024/10/23 
        # We should use a random bias_per_patch for a test?
        # bias_per_patch = torch.zeros((int(camera.height), int(camera.width)), dtype=torch.int32, device="cuda")
        pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
            bg = torch.zeros(3, device=self.device),
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0, # Set None here will lead to TypeError.
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            # stable_mask = self._stable_mask if unopt_gaussian_mask is None else unopt_gaussian_mask,
            pixel_mask = pixel_mask,
            # pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # (2) Render.
        means3D        = self.get_property('_xyz') # (N, 3)
        means2D        = screenspace_points
        opacity        = self.get_property('_opacity') # (N, 1)
        scales         = self.get_property('_scaling') # (N, 3)
        rotations      = self.get_property('_rotation') # (N, 4)
        colors_precomp = self.get_property('_rgb') # (N, 3)
        self._zeros    = self.get_property('_zeros') # (N, 2)
        render_zeros   = self._zeros
        
        if 'refine_pose' in self.cfg.keys() and self.cfg['refine_pose']:
            means3D = w2c[:3,] @ means3D
            rotations = None
        
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            scores    = render_zeros,
            cov3D_precomp = None
        )
        
        render_alpha = allmap[1:2]
        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        rets = {}
        # (depth, accum, normal, (median_depth), dist)
        rets['radii']   = radii # (1, H, W)
        rets['accum']   = render_alpha # (1, H, W)
        rets['rgb']     = rendered_image # (3, H, W)
        rets['depth']   = render_depth_expected # (1, H, W)
        # transform normal from view space to world space
        rets['normal']  = (allmap[2:5].permute(1,2,0) @ (w2c[:3,:3])).permute(2,0,1) # (3, H, W)
        rets['dist']    = allmap[6:7] # (1, H, W)
        rets['surf_normal'] = depth_propagate_normal(rets['depth'].squeeze(0), self.tfer).permute(2,0,1) # (3, H, W)
        rets['surf_normal'] = ( rets['surf_normal'] .permute(1,2,0) @ (w2c[:3,:3]) ).permute(2,0,1)
        # rets['n_contrib']   = allmap[7:8]
        
        return rets
    
    
    def render(self, w2c, intrinsic_dict, unopt_gaussian_mask = None, w2c2=None):
        if w2c2 is None:
            w2c2 = w2c
        rets = self.render_raw(w2c, intrinsic_dict, unopt_gaussian_mask = None)
        # rets = self.render_opticalflow(w2c, w2c2, intrinsic_dict, unopt_gaussian_mask = None)
        return rets
    
    
    # TTD 2024/12/09
    # Tailored for loop detect in .
    # We select gaussian in 100m distance and render them, this process is with out grad.
    def render_indistance(self, w2c, intrinsic_dict, unopt_gaussian_mask = None, w2c2=None):
        
        with torch.no_grad():
        
            if w2c2 is None:
                w2c2 = w2c
            # rets = self.render_raw(w2c, intrinsic_dict, unopt_gaussian_mask = None)
            
            screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
            
            # (1) Setup raster_settings.
            camera = get_camera(w2c, intrinsic_dict)
            
            # TTD 2024/10/23 
            pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
            
            raster_settings = GaussianRasterizationSettings(
                image_height=int(camera.height),
                image_width=int(camera.width),
                tanfovx=camera.tanfovx,
                tanfovy=camera.tanfovy,
                # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
                bg = torch.zeros(3, device=self.device),
                scale_modifier=1.0,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform,
                sh_degree=0, # Set None here will lead to TypeError.
                campos=camera.camera_center,
                prefiltered=False,
                debug=False,
                # stable_mask = self._stable_mask if unopt_gaussian_mask is None else unopt_gaussian_mask,
                pixel_mask = pixel_mask,
                # pipe.debug
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            # (2) Render.
            means3D        = self.get_property('_xyz') # (N, 3)
            means2D        = screenspace_points
            opacity        = self.get_property('_opacity') # (N, 1)
            scales         = self.get_property('_scaling') # (N, 3)
            rotations      = self.get_property('_rotation') # (N, 4)
            colors_precomp = self.get_property('_rgb') # (N, 3)
            self._zeros    = self.get_property('_zeros') # (N, 2)
            
            
            indistance_mask = torch.linalg.norm(means3D - torch.linalg.inv(w2c)[:3, 3].unsqueeze(0), dim=1)<60
            means3D = means3D[indistance_mask]
            means2D = means2D[indistance_mask]
            opacity = opacity[indistance_mask]
            scales = scales[indistance_mask]
            rotations = rotations[indistance_mask]
            colors_precomp = colors_precomp[indistance_mask]
            self._zeros = self._zeros[indistance_mask]
            render_zeros   = self._zeros
            
            
            if 'refine_pose' in self.cfg.keys() and self.cfg['refine_pose']:
                means3D = w2c[:3,] @ means3D
                rotations = None
            
            rendered_image, radii, allmap = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                scores    = render_zeros,
                cov3D_precomp = None
            )
            
            render_alpha = allmap[1:2]
            # get expected depth map
            render_depth_expected = allmap[0:1]
            render_depth_expected = (render_depth_expected / render_alpha)
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
            
            rets = {}
            # (depth, accum, normal, (median_depth), dist)
            rets['radii']   = radii # (1, H, W)
            rets['accum']   = render_alpha # (1, H, W)
            rets['rgb']     = rendered_image # (3, H, W)
            rets['depth']   = render_depth_expected # (1, H, W)
            # transform normal from view space to world space
            rets['normal']  = (allmap[2:5].permute(1,2,0) @ (w2c[:3,:3])).permute(2,0,1) # (3, H, W)
            rets['dist']    = allmap[6:7] # (1, H, W)
            rets['surf_normal'] = depth_propagate_normal(rets['depth'].squeeze(0), self.tfer).permute(2,0,1) # (3, H, W)
            rets['surf_normal'] = ( rets['surf_normal'] .permute(1,2,0) @ (w2c[:3,:3]) ).permute(2,0,1)
        
        return rets
    
    
    
    def update_properties(self, new_dict):
        '''
        Run this when add/prune new gaussians.
        You should update optimizer and get new_dict from optimizer.
        '''
        self._xyz, self._rgb, self._opacity, self._scaling, self._rotation = new_dict['_xyz'], new_dict['_rgb'], new_dict['_opacity'], new_dict['_scaling'], new_dict['_rotation']
    
    def train_once_gaussian(self, batch, train_iters):
        # Get data from batch.
        # abs_frame_idx_list = batch["viz_out_idx_to_f_idx"]    # (N, 1)
        poses              = batch["poses"]                   # (N, 4, 4)
        images             = batch["images"]                  # (N, 344, 616, 3)
        depths             = batch["depths"]                  # (N, 344, 616, 1)
        depths_cov         = batch["depths_cov"]              # (N, 344, 616, 1) 
        intrinsic_dict     = batch["intrinsic"]               # {'fu', 'fv', 'cu', 'cv', 'H', 'W'}
        # pixel_masks        = batch["pixel_mask"]              # (N, 344, 616)
        
        for curr_iter in range(train_iters):
            
            self.wandber.log_time('forward_time')

            # curr_id = random.randint(0, poses.shape[0]-2) # vo_nerfslam
            curr_id = random.randint(0, poses.shape[0]-1)
            
            
            if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile'] and curr_iter == train_iters - 1:
                curr_id = max(0, poses.shape[0]-1)
            
            c2w = poses[curr_id]
            w2c = torch.linalg.inv(c2w)
            
            pred_dict = self.render(w2c, intrinsic_dict, None, w2c2=torch.linalg.inv(poses[min(curr_id+1, poses.shape[0]-1)]))
            gt_dict = {'rgb': images[curr_id].permute(2,0,1), 'depth': depths[curr_id].permute(2,0,1), 'uncert': depths_cov[curr_id].permute(2,0,1), 'c2w': c2w}
            gt_dict['depth_cov'] = depths_cov[curr_id].permute(2,0,1)
            
            self.wandber.log_time('forward_time')
            
            if self.cfg['use_sky']:
                gt_dict['sky_rgb'] = batch["sky_images"][curr_id].permute(2,0,1) # (3, H, W)
                pred_dict_sky = self.sky_model.render(w2c, intrinsic_dict)
                pred_dict['rgb'] = self.sky_model.fuse_rgb(pred_dict, pred_dict_sky)
            
            self.wandber.log_time('backward_time')
            pred_dict['time_idx'] = self.time_idx
            total_loss = get_loss(self.cfg, pred_dict, gt_dict)
            
            total_loss.backward()
            self.wandber.log_time('backward_time')

            # (1) Record Importance Score & Error Score. (2) Multiply weights by accumulate scores to avoid forgetting problem.
            self.wandber.log_time('record_time')
            _current_scores = self._zeros.grad.detach()
            self.add_records(_current_scores)
            # 2024/10/01, 🐖 mapper._globalkf_id 和 tracker.video.poses_save 的索引是一致的;
            replace_mask = self._globalkf_max_scores<_current_scores[:,0]
            self._globalkf_max_scores[replace_mask] = _current_scores[replace_mask,0]
            self._globalkf_id[replace_mask] = batch['global_kf_id'][curr_id]

            # TTD 2024/11/17
            weighting_grad(self, _current_scores, self._global_scores)
            
            self.wandber.log_time('record_time')
            
            self.wandber.log_time('step_time')
            radii = pred_dict['radii']
            radii[self._stable_mask] = 0
            self.optimizer.step(radii > 0, radii.shape[0])
            self.optimizer.zero_grad()
            self.wandber.log_time('step_time')

            if self.cfg['use_sky']:
                radii_sky = pred_dict_sky['radii']
                self.sky_model.optimizer.step(radii_sky > 0, radii_sky.shape[0])
                self.sky_model.optimizer.zero_grad()

            # self.wandber.log_time('Time_PerIter')

            # TTD 2024/12/29 dangerous option.
            if self.cfg.get('use_vis', False) and curr_iter == train_iters - 1:
                gt_dict['pose'] = c2w
                gt_dict['abs_frame_idx_list'] = batch["viz_out_idx_to_f_idx"]
                frame_id = batch["viz_out_idx_to_f_idx"][curr_id]
                if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                    self.vis_rgbdnua = vis_rgbdnua(self.cfg, frame_id, pred_dict, gt_dict, True)
                else:
                    vis_rgbdnua(self.cfg, frame_id, pred_dict, gt_dict)
                    
                self.wandber.log_once("num_of_gaussians", self._xyz.shape[0])
                self.wandber.log_once("psnr", calc_psnr(pred_dict['rgb'], gt_dict['rgb'], gt_dict['depth'].squeeze(0)>0).item())
            
            self.wandber.log_time('adcs_time')
            self.stablemask_control(curr_iter)
            # self.adaptive_densify_control(curr_iter, batch)

            self.storage_control(curr_iter, batch)
            
            self.wandber.log_time('adcs_time')
            
        self.time_idx += 1
    
    
    
    def train_once(self, batch, train_iters):
        
        # self.wandber.log_time('per_idx_time')
        
        self.train_once_gaussian(batch, train_iters)
        
        # self.wandber.log_time('per_idx_time')
        
    def run_only_mapping(self, processed_dict, return_vizout=False):
        if self.initialized_state:
            new_frame_added, new_added_dict = self.judge_new_frame(processed_dict)
            if new_frame_added:
                self.add_new_frame(new_added_dict)
                
                # TTD 2024/11/17
                if 'use_refine' in self.cfg.keys() and self.cfg['use_refine']:        
                   new_poses        = self.train_once_pose(processed_dict)
                   processed_dict['poses'] = new_poses
                
                # Excellent Option.
                self.train_once(processed_dict, self.cfg['training_args']['iters']) 
                
                if 'use_refine' in self.cfg.keys() and self.cfg['use_refine']:    
                    if return_vizout:
                        new_poses        = self.train_once_pose(processed_dict)
                        processed_dict['poses'] = new_poses
                        return processed_dict
                    
                
            # Dangerous Option.
            # self.train_once(processed_dict, self.cfg['training_args']['iters'])
        else:
            self.init_first_frame(processed_dict)
            self.initialized_state = True
            self.setup_optimizer()
            self.train_once(processed_dict, self.cfg['training_args']['iters'])
        
        return None
        
    def load_ply_ckpt(self, ckpt_path):
        property_dict_npy = load_ply(ckpt_path)
        self._xyz = nn.Parameter(torch.tensor(property_dict_npy['_xyz'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._rgb = nn.Parameter(torch.tensor(property_dict_npy['_rgb'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(property_dict_npy['_opacity'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(property_dict_npy['_scaling'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(property_dict_npy['_rotation'], dtype=torch.float32, device=self.device).contiguous().requires_grad_(True))
        self.setup_optimizer()
            
        # Please remember to set tfer before render.
        self.tfer.H, self.tfer.W = None, None
        self.tfer.cu, self.tfer.cv = None, None
        self.tfer.fu, self.tfer.fv = None, None
        self.initialized_state = True
    
    def save_pt_ckpt(self, ckpt_path):
        ckpt_dict = {
            '_xyz': self._xyz,
            '_rgb': self._rgb,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            '_opacity': self._opacity,
            '_global_scores': self._global_scores,
            '_local_scores': self._local_scores,
            '_stable_mask': self._stable_mask,
            '_globalkf_id': self._globalkf_id,
            '_globalkf_max_scores': self._globalkf_max_scores,
        }
        torch.save(ckpt_dict, ckpt_path)
    
    def load_pt_ckpt(self, ckpt_path):
        ckpt_dict = torch.load(ckpt_path)
        self._xyz                 = ckpt_dict['_xyz']
        self._rgb                 = ckpt_dict['_rgb']
        self._scaling             = ckpt_dict['_scaling']
        self._rotation            = ckpt_dict['_rotation']
        self._opacity             = ckpt_dict['_opacity']
        self._global_scores       = ckpt_dict['_global_scores']
        self._local_scores        = ckpt_dict['_local_scores']
        self._stable_mask         = ckpt_dict['_stable_mask']
        self._globalkf_id         = ckpt_dict['_globalkf_id']
        self._globalkf_max_scores = ckpt_dict['_globalkf_max_scores']
        
        self.setup_optimizer()
        # Please remember to set tfer before render.
        self.tfer.H, self.tfer.W = None, None
        self.tfer.cu, self.tfer.cv = None, None
        self.tfer.fu, self.tfer.fv = None, None
        self.initialized_state = True
    
    # TODO: Implement GaussianModel.run().
    def run(self, processed_dict, return_vizout=False):
        processed_dict_new = self.run_only_mapping(processed_dict, return_vizout)
        
        return processed_dict_new

