'''
Customize ADC(Adaptive Densify Control) for Gaussian model.
Version: 0.0
Date: 2021-07-04
Description: Vanilla 2DGS's policy.(with new clone opacity)
'''
from gaussian.gaussian_base import GaussianBase
import torch
import torch.nn as nn
from gaussian.gaussian_utils import distCUDA2, get_pointcloud, get_split_properties, get_u2_minus_u1
from gaussian.general_utils import inverse_sigmoid
from gaussian.wandb_utils import Wandber
from torch.autograd import Variable
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.cameras import get_camera
from gaussian.sky_utils import SkyModel
# TTD 2024/07/12
from lietorch import SE3, SO3
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from gaussian.normal_utils import depth_propagate_normal
# TTD 2024/11/17
from vings_utils.refineposes_utils import get_xyz_bias_multi, get_new_xyz_single
from gaussian.loss_utils import l1_loss, ssim_loss
from gaussian.mapping_budget import select_new_gaussian_budget
from gaussian.pruning_budget import cap_prune_mask_by_score, select_pruning_budget


class GaussianModel(GaussianBase):
    def __init__(self, cfg):
        super(GaussianModel, self).__init__(cfg)
        self.wandber = Wandber(cfg, self.cfg['output']['save_dir'].split('/')[-1])
        self.global_c2w = {}
        self.time_idx   = 0

    def init_first_frame(self, batch):
        '''
        (1) Reset self.tfer.
        (2) Get init pointcloud and relative attributes.
        '''
        depths = batch["depths"] # (N, 344, 616, 1)
        images          = batch["images"] # (N, 344, 616, 4)
        poses           = batch["poses"] # (N, 4, 4)
        depths_cov      = batch["depths_cov"] # (N, 344, 616, 1) 
        
        # (1) Reset self.tfer.
        self.tfer.H = batch['intrinsic']['H']
        self.tfer.W = batch['intrinsic']['W']
        self.tfer.fu, self.tfer.cu = batch['intrinsic']['fu'], batch['intrinsic']['cu']
        self.tfer.fv, self.tfer.cv = batch['intrinsic']['fv'], batch['intrinsic']['cv']
        
        # (2) Accumulate Point Cloud.
        pc_world_list = []
        pc_color_list = []
        pc_rots_list = []
        for idx in range(depths.shape[0]):
            pose = poses[idx] # (4, 4)
            depth = depths[idx] # (H, W)
            rgb = images[idx] # (H, W, 3)
            xyz, rgb, q = get_pointcloud(self.tfer, pose, rgb.permute(2, 0, 1), depth.permute(2, 0, 1), None, 50000) 
            pc_world_list.append(xyz)
            pc_color_list.append(rgb)
            pc_rots_list.append(q)

        # (3) Set self.history_list.
        self.history_list = batch['viz_out_idx_to_f_idx'].tolist()

        pc_world = torch.cat(pc_world_list, dim=0)# (N, 3)
        pc_world_color = torch.cat(pc_color_list, dim=0) # (N, 3)
        pc_world_rots  = torch.cat(pc_rots_list, dim=0) # (N, 4)
                
        dist2 = torch.clamp_min(distCUDA2(pc_world), 0.0000001)
        scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 2) # (N, 2)
        opacities = inverse_sigmoid(0.1 * torch.ones((pc_world.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz      = nn.Parameter(pc_world.contiguous().requires_grad_(True))
        self._rgb      = nn.Parameter(pc_world_color.contiguous().requires_grad_(True))
        self._scaling  = nn.Parameter(scales.contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(pc_world_rots.contiguous().requires_grad_(True))
        self._opacity  = nn.Parameter(opacities.contiguous().requires_grad_(True))
        self._local_scores  = torch.zeros((pc_world.shape[0], 2), dtype=torch.float32, device=self.device)
        self._global_scores = torch.zeros((pc_world.shape[0], 2), dtype=torch.float32, device=self.device)
        self._stable_mask   = torch.zeros(pc_world.shape[0], dtype=torch.bool, device=self.device)
        # TTD 2024/10/01, 记录每个gaussian到底属于哪个global_kf_id, 这个逃不掉的, 反正之后做loop closure肯定也要用。
        self._globalkf_id         = torch.zeros((pc_world.shape[0], ), dtype=torch.long, device=self.device)
        self._globalkf_max_scores = torch.zeros((pc_world.shape[0], ), dtype=torch.float32, device=self.device)

        if self.cfg['use_sky']:
            self.sky_model = SkyModel(self)
            self.sky_model.init_first_frame(batch)
    
    def add_new_frame(self, new_added_frame):
        new_added_pose = new_added_frame['pose'] # (4, 4)
        new_added_depth = new_added_frame['depth'] # (H, W, 1)
        new_added_color = new_added_frame['image'] # (H, W, 3)
        intrinsic_dict  = new_added_frame['intrinsic']
        new_added_c2w = new_added_pose # (4, 4)
        new_added_w2c = torch.inverse(new_added_c2w)
        with torch.no_grad():
            # Render Accumulation.
            rets = self.render(new_added_w2c, intrinsic_dict)
            pred_rgb   = rets['rgb']   # (3, H, W)
            pred_depth = rets['depth'] # (1, H, W)
            radii      = rets['radii'] # (1, H, W)

            # Delete pixels with large rgb error and in 1.5*gt_depth range.
            res_rgb = torch.abs(pred_rgb - new_added_color.permute(2, 0, 1)).sum(axis=0) # (H, W)
            loss_threshold   = 0.15
            delete_pixelmask = torch.bitwise_and((pred_depth.squeeze(0) < 1.5 * new_added_depth.squeeze(-1)), (res_rgb > loss_threshold)) # 
            
            proj_uv = self.tfer.transform(self.get_property('_xyz'), 'world', 'pixel', pose=new_added_c2w) # (P, 3), P = validdepth_mask.sum()
            visible_gaussianmask = (proj_uv[:, 0] > 0) & (proj_uv[:, 0] < self.tfer.H-1) & (proj_uv[:, 1] > 0) & (proj_uv[:, 1] < self.tfer.W-1) & (proj_uv[:, 2] > 0.01) # (P)
            
            delete_gaussianmask  = torch.zeros_like(visible_gaussianmask)
            delete_gaussianmask[visible_gaussianmask][delete_pixelmask[proj_uv[visible_gaussianmask,0].to(torch.long), proj_uv[visible_gaussianmask,1].to(torch.long)]] = True
            # delete_gaussianmask[visible_gaussianmask][proj_uv[visible_gaussianmask,2] > 1.5 * new_added_depth.squeeze(-1)[proj_uv[visible_gaussianmask,0].to(torch.int32), proj_uv[visible_gaussianmask,1].to(torch.int32)]] = False

            # Prune Gaussians have big radii.
            delete_gaussianmask[radii>25] = True
            
        new_dict = self.prune_tensors_from_optimizer(self.optimizer, delete_gaussianmask)
        self.update_properties(new_dict)
        self.update_records(mode="prune", prune_gaussianmask=delete_gaussianmask)

        with torch.no_grad():
            rets = self.render(new_added_w2c, intrinsic_dict)
            # Add Gaussians on area with "large rgb/depth error or have low accum".
            pred_accum  = rets['accum'] # (1, H, W)
            pred_depth  = rets['depth'] # (1, H, W)
            depth_error = torch.abs(pred_depth-new_added_depth.permute(2, 0, 1))
            rgb_error   = torch.abs(pred_rgb-new_added_color.permute(2, 0, 1)).sum(axis=0, keepdim=True)
            pred_accum[depth_error > 10*depth_error.median()] = 0.0
            pred_accum[rgb_error > 0.1] = 0.0

        # Get point cloud and concat it to GaussianModel.
        mapping_budget_cfg = self.cfg.get("mapping_budget", {})
        accum_threshold = float(mapping_budget_cfg.get("accum_threshold", self.cfg["adc_args"]["accum_thresh"]))
        uncovered_ratio = float((pred_accum < accum_threshold).float().mean().item())
        gaussian_count_before = int(self._xyz.shape[0])
        new_gaussian_budget = select_new_gaussian_budget(
            self.cfg,
            current_gaussians=gaussian_count_before,
            uncovered_ratio=uncovered_ratio,
        )
        new_added_pc, new_added_pc_color, unnorm_rots = get_pointcloud(
            self.tfer,
            new_added_c2w,
            new_added_color.permute(2, 0, 1),
            new_added_depth.permute(2, 0, 1),
            pred_accum,
            new_gaussian_budget,
        )
        num_pts = new_added_pc.shape[0]
        if mapping_budget_cfg.get("log", True):
            print(
                "[mapping_budget] "
                f"gaussians_before={gaussian_count_before} "
                f"uncovered_ratio={uncovered_ratio:.4f} "
                f"requested={new_gaussian_budget} "
                f"added={num_pts} "
                f"gaussians_after={gaussian_count_before + num_pts}"
            )

        dist2 = torch.clamp_min(distCUDA2(new_added_pc), 0.0000001)
        log_scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, 2)
        logit_opacities = inverse_sigmoid((0.8*torch.ones((num_pts, 1), device=new_added_pc.device))).to(torch.float)

        new_params = {
            '_xyz': new_added_pc,
            '_rgb': new_added_pc_color,
            '_scaling': log_scales,
            '_rotation': unnorm_rots,
            '_opacity': logit_opacities
        }
        
        self._xyz = torch.nn.Parameter(torch.cat((self._xyz, new_params['_xyz']), dim=0).requires_grad_(True))
        self._rgb = torch.nn.Parameter(torch.cat((self._rgb, new_params['_rgb']), dim=0).requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.cat((self._scaling, new_params['_scaling']), dim=0).requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.cat((self._rotation, new_params['_rotation']), dim=0).requires_grad_(True))
        self._opacity = torch.nn.Parameter(torch.cat((self._opacity, new_params['_opacity']), dim=0).requires_grad_(True))
        self.update_records(mode="densify", densify_gaussiannum=num_pts)
        
        if self.cfg['use_sky']:
            self.sky_model.add_new_frame(new_added_frame)
        
        self.setup_optimizer()
    
    def add_records(self, _current_scores):
        self._local_scores[:, 0]  += _current_scores[:, 0]
        self._global_scores[:, 0] += _current_scores[:, 0]
        largeerror_mask = _current_scores[:, 1] > self._local_scores[:, 1]
        self._local_scores[largeerror_mask, 1]  = _current_scores[largeerror_mask, 1]
        self._global_scores  = torch.clamp(self._global_scores, 0, 1e4)
    
    def update_records(self, mode=None, densify_gaussiannum=None, prune_gaussianmask=None):
        if mode == "densify":
            self._local_scores  = torch.cat((self._local_scores, torch.zeros((densify_gaussiannum, 2), dtype=torch.float32, device=self.device)), dim=0)
            self._global_scores = torch.cat((self._global_scores, torch.zeros((densify_gaussiannum, 2), dtype=torch.float32, device=self.device)), dim=0)
            self._stable_mask   = torch.cat((self._stable_mask, torch.zeros((densify_gaussiannum, ), dtype=torch.bool, device=self.device)), dim=0)        
            self._globalkf_id         = torch.cat((self._globalkf_id, torch.zeros((densify_gaussiannum, ), dtype=torch.long, device=self.device)), dim=0)
            self._globalkf_max_scores = torch.cat((self._globalkf_max_scores, torch.zeros((densify_gaussiannum, ), dtype=torch.float32, device=self.device)), dim=0)
            
        elif mode == "prune":
            self._local_scores  = self._local_scores[~prune_gaussianmask]
            self._global_scores = self._global_scores[~prune_gaussianmask]
            self._stable_mask   = self._stable_mask[~prune_gaussianmask.reshape(-1)]
            self._globalkf_id         = self._globalkf_id[~prune_gaussianmask.reshape(-1)]
            self._globalkf_max_scores = self._globalkf_max_scores[~prune_gaussianmask.reshape(-1)]
        else:
            assert False, "Invalid mode."
    
    def stablemask_control(self, current_iter):
        if (current_iter == self.cfg['training_args']['iters'] - 1) and \
           (self.time_idx+1) % self.cfg['training_args']['num_keyframe'] == 0:
            # Unstable → Stable ClassA: Gaussians whose "_local_scores[:, 0]" have no change during last num_iters.
            unstable2stable_mask = (~self._stable_mask) & (self._local_scores[:,0] < 1e-4)
            self._stable_mask[unstable2stable_mask] = True
            # Unstable → Stable ClassB: Gaussians whose "_local_scores[:,  ]" > th and "_local_score[:, 1]" < th.
            # Stable → Unstable: Gaussians whose "_local_scores[:, 1]" becomes too large.
            # Dangerous Option, Stable2Unstable.
            # stable2unstable_mask = (self._local_scores[:,1]>0.5) & (self._local_scores[:,0]>0.1) & (self._stable_mask)
            stable2unstable_mask = (self._local_scores[:,1]>0.3) & (self._local_scores[:,0]>0.05) & (self._stable_mask)
            self._stable_mask[stable2unstable_mask] = False
            
            self._local_scores *= 0.0
            

    def adaptive_densify_control(self, current_iter, batch):
        pass
        '''
        ERROR_THRESHOLD = 1e4 # 0.004
        # Split Gaussians: gaussian-level, where _local_scores[:, 1]/_local_scores[:, 0] > th.
        if (current_iter == self.cfg['training_args']['iters'] - 1) and \
           (self.time_idx+1) % 2 == 0:
            # avg_error_scores   = self._local_scores[:, 1] / (self._local_scores[:, 0]+1e-4)
            # split_gaussianmask = (avg_error_scores > 0.3) & (~self._stable_mask) & (self._local_scores[:, 0] > 0.1)
            split_gaussianmask = (self._local_scores[:, 1] > ERROR_THRESHOLD) & (~self._stable_mask)
            subgaussian_dict   = get_split_properties(self, split_gaussianmask)

            new_dict = self.prune_tensors_from_optimizer(self.optimizer, split_gaussianmask)
            self.update_properties(new_dict)
            self.update_records(mode="prune", prune_gaussianmask=split_gaussianmask)

            new_dict = self.cat_tensors_to_optimizer(self.optimizer, subgaussian_dict)
            self.update_properties(new_dict)
            self.update_records(mode="densify", densify_gaussiannum=subgaussian_dict['_xyz'].shape[0])
            # print("Split Gaussians: ", split_gaussianmask.sum().item())

        # Densify Gaussians: pixel-level, where RGB error or Depth error is large.
        '''
        
    # TODO: Iter over all history frames.
    def storage_control(self, current_iter, batch):
        pruning_budget = select_pruning_budget(
            self.cfg,
            current_gaussians=int(self._xyz.shape[0]),
            time_idx=self.time_idx,
        )
        if (current_iter == self.cfg['training_args']['iters'] - 1) and \
           pruning_budget["should_prune"]:
            # (self.time_idx+1) % self.cfg['training_args']['num_keyframe'] == 0:
            # Rerender on whole keyframe list and prune unstable gaussians whose _local_scores[:, 0] < 1.0.
            temp_importance_scores = torch.zeros_like(self._local_scores[:, 0]) # (P, )
            intrinsic_dict = batch["intrinsic"]
            for kf_idx in range(batch["poses"].shape[0]):
                c2w, gt_rgb = batch["poses"][kf_idx], batch["images"][kf_idx].permute(2, 0, 1) # (4, 4), (3, H, W)
                pred_rgb = self.render(torch.linalg.inv(c2w), intrinsic_dict)['rgb']
                (torch.abs(pred_rgb-gt_rgb)[:, gt_rgb.sum(axis=0)>0]).mean().backward()
                temp_importance_scores += self._zeros.grad.detach()[:, 0]
                self.optimizer.zero_grad()
                self._zeros.grad.zero_()
            # prune_gaussianmask = (temp_importance_scores > 0.1) & (~self._stable_mask) & (temp_importance_scores < 0.8)    
            # prune_gaussianmask = (temp_importance_scores > 0.05) & (~self._stable_mask) & (temp_importance_scores < 0.8)
            # Ablation TTD 2024/12/04
            prune_gaussianmask = (
                (temp_importance_scores > pruning_budget["low_threshold"])
                & (~self._stable_mask)
                & (temp_importance_scores < pruning_budget["high_threshold"])
            )
            max_prune_count = int(self._xyz.shape[0] * pruning_budget["max_prune_ratio"])
            prune_gaussianmask = cap_prune_mask_by_score(
                prune_gaussianmask,
                temp_importance_scores,
                max_prune_count=max_prune_count,
            )
            prune_count = int(prune_gaussianmask.sum().item())
            if pruning_budget["enabled"] and self.cfg.get("pruning_budget", {}).get("log", True):
                print(
                    "[pruning_budget] "
                    f"enabled={int(pruning_budget['enabled'])} "
                    f"stage={pruning_budget['stage']} "
                    f"interval={pruning_budget['interval']} "
                    f"gaussians_before={int(self._xyz.shape[0])} "
                    f"low={pruning_budget['low_threshold']:.4f} "
                    f"high={pruning_budget['high_threshold']:.4f} "
                    f"max_ratio={pruning_budget['max_prune_ratio']:.4f} "
                    f"pruned={prune_count}"
                )
            new_dict = self.prune_tensors_from_optimizer(self.optimizer, prune_gaussianmask)
            self.update_properties(new_dict)
            self.update_records(mode="prune", prune_gaussianmask=prune_gaussianmask)
            
            # Ablation TTD 2024/12/04
            # print("Prune Gaussians: ", prune_gaussianmask.sum().item())
            

    def render_refine(self, w2c, left_part, intrinsic_dict, unopt_gaussian_mask = None):
        # Copy 2DGS.
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # (1) Setup raster_settings.
        camera = get_camera(w2c, intrinsic_dict)
        
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
            u2_minus_u1 = torch.zeros_like(self._xyz[..., :2])
            # pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # (2) Render.
        means3D_old    = self.get_property('_xyz') # (N, 3)
        means2D        = screenspace_points
        opacity        = self.get_property('_opacity') # (N, 1)
        scales         = self.get_property('_scaling') # (N, 3)
        rotations_old  = self.get_property('_rotation') # (N, 4)
        colors_precomp = self.get_property('_rgb') # (N, 3)
        self._zeros    = self.get_property('_zeros') # (N, 2)
        render_zeros   = self._zeros
        
        # -  -  -  -  -  -  -  -  -  -  -  -  -
        # Get new means3D and rotations.
        # cold2w_mul_w2cnew.shape = (1, 7)
        # with torch.no_grad():
        #     cold2w_mul_w2cnew[:, 3:] = cold2w_mul_w2cnew[:, 3:] / torch.norm(cold2w_mul_w2cnew[:, 3:])
        # cold2w_mul_w2cnew_matrix = SE3(cold2w_mul_w2cnew).matrix().squeeze(0) # (4, 4)
        # means3D   = torch.matmul(means3D_old, (cold2w_mul_w2cnew_matrix[:3,:3]).T) + cold2w_mul_w2cnew_matrix[:3,-1].unsqueeze(0) # (N, 3)
        # means3D = means3D_old + cold2w_mul_w2cnew_matrix[:3,-1].unsqueeze(0) # (N, 3)
        # rotations = (SO3.InitFromVec(rotations_old) * SO3.InitFromVec(cold2w_mul_w2cnew[:, 3:]).inv()).vec()
        
        means3D   = means3D_old @ left_part[:3,:3].T + left_part[:3,3].unsqueeze(0) # (N, 3)
        # means3D   = (means3D_old@c2w_refine_matrix[:3,:3]+torch.linalg.inv(c2w_refine_matrix)[:3,-1].unsqueeze(0)-w2c[:3,-1].unsqueeze(0)) @ w2c[:3,:3]
        rotations = rotations_old
        # -  -  -  -  -  -  -  -  -  -  -  -  -
        
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
    
    def render_opticalflow(self, w2c1, w2c2, intrinsic_dict, unopt_gaussian_mask = None):
        # Copy 2DGS.
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # (1) Setup raster_settings.
        camera = get_camera(w2c1, intrinsic_dict)
        
        pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
        
        # Calculate (u2 - u1), hope projection dosen't cost a long time.
        u2_minus_u1 = get_u2_minus_u1(w2c1, w2c2, self.get_property('_xyz'), self.tfer) # (N, 2)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
            # bg=torch.zeros(3, device=self.device) if (random.random()>0.5) else torch.ones(3, device=self.device),
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
            u2_minus_u1 = u2_minus_u1, # TTD 2024/10/09
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
            means3D = w2c1[:3,] @ means3D
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
        rets['normal']  = (allmap[2:5].permute(1,2,0) @ (w2c1[:3,:3])).permute(2,0,1) # (3, H, W)
        rets['dist']    = allmap[6:7] # (1, H, W)
        rets['surf_normal'] = depth_propagate_normal(rets['depth'].squeeze(0), self.tfer).permute(2,0,1) # (3, H, W)
        rets['surf_normal'] = ( rets['surf_normal'] .permute(1,2,0) @ (w2c1[:3,:3]) ).permute(2,0,1)
        # rets['n_contrib']   = allmap[7:8]
        rets['optical_flow'] = allmap[7:9] # vu, (2, H, W)
        
        return rets
    
    
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # TTD 2024/11/16
    def train_once_pose_v1(self, batch):
        '''
        注意两点,这个优化pose分别体现在 渲染位姿 和 其他帧的 gaussians;
        We have global_kf_id for each gaussian, so can we optimize poses(except first k poses to keep sequent) using global kf_id after training one batch?
        (1) We need to record a independent poses-timestamps in mapper. 
        '''
        poses              = batch["poses"]                   # (N, 4, 4)
        images             = batch["images"]                  # (N, 344, 616, 3)
        depths             = batch["depths"]                  # (N, 344, 616, 1)
        depths_cov         = batch["depths_cov"]              # (N, 344, 616, 1) 
        intrinsic_dict     = batch["intrinsic"]               # {'fu', 'fv', 'cu', 'cv', 'H', 'W'}
        batch_global_kf_id = batch['global_kf_id']
        
        # STEP 1 Prepare variables & Setup optimizer.
        localbatch_globalkf_id_mask = []
        for i in range(batch_global_kf_id.shape[0]):
            localbatch_globalkf_id_mask.append(self._globalkf_id==batch_global_kf_id[i])
        
        optimizing_c2c_num = poses.shape[0]
        optimize_c2c_tqs   = torch.zeros(7, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_tqs[:, -1] += 1.0
        optimize_c2c_tqs   = nn.Parameter(optimize_c2c_tqs, requires_grad=True)
        
        if '_c2c_lr' in self.cfg['training_args']['lr'].keys():
            c2c_lr = self.cfg['training_args']['lr']['_c2c_lr']
        else:
            c2c_lr = 0.0001
        self.pose_optimizer = torch.optim.Adam([optimize_c2c_tqs], lr=c2c_lr) # 0.002
        
        
        # localbatch_gaussians_mask = torch.bitwise_and(self._globalkf_id>=batch['global_kf_id'].min(), self._globalkf_id<=batch['global_kf_id'].max())
        
        # STEP 2 Get new gaussians' _xyz and render.
        train_pose_iters = 20
        for curr_iter in range(train_pose_iters):
            curr_id = random.randint(0, poses.shape[0]-1)
            c2w     = poses[curr_id]
            
            # new_xyz = get_new_xyz(self, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, curr_id)
            xyz_bias = get_xyz_bias_multi(self, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, self._xyz)
            new_xyz = self._xyz + xyz_bias
            new_xyz = get_new_xyz_single(new_xyz, poses[curr_id], optimize_c2c_tqs[curr_id])
            
            
            w2c = torch.linalg.inv(c2w)
            camera = get_camera(w2c, intrinsic_dict)
            pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
            raster_settings = GaussianRasterizationSettings(image_height=int(camera.height), image_width=int(camera.width),
                                tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
                                bg = torch.zeros(3, device=self.device), scale_modifier=1.0,
                                viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
                                sh_degree=0, campos=camera.camera_center,
                                prefiltered=False, debug=False,
                                pixel_mask = pixel_mask)
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
            means2D        = screenspace_points
            opacity        = self.get_property('_opacity') # (N, 1)
            scales         = self.get_property('_scaling') # (N, 3)
            rotations_old  = self.get_property('_rotation') # (N, 4)
            colors_precomp = self.get_property('_rgb') # (N, 3)
            self._zeros    = self.get_property('_zeros') # (N, 2)
            render_zeros   = self._zeros
            
            rendered_image, radii, allmap = rasterizer(
                                                means3D = new_xyz,
                                                means2D = means2D,
                                                shs = None,
                                                colors_precomp = colors_precomp,
                                                opacities = opacity,
                                                scales = scales,
                                                rotations = rotations_old,
                                                scores    = render_zeros,
                                                cov3D_precomp = None
                                                )
            
            valid_mask = depths[curr_id].squeeze(0).squeeze(-1) > 0 # (H, W)
            gt_rgb     = images[curr_id].permute(2,0,1) # (3, H, W)
                        
            rgb_loss   = 0.8 * l1_loss(rendered_image, gt_rgb, valid_mask) +\
                        0.2 * (1.0 - ssim_loss(rendered_image, gt_rgb, valid_mask))
                        
            rgb_loss.backward()
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad()
        
        rectified_poses = torch.matmul(SE3(optimize_c2c_tqs.detach()).matrix(), poses)
        
        self._xyz = (self._xyz.detach() + xyz_bias.detach()).requires_grad_(True)
        self.setup_optimizer()
        self.optimizer.zero_grad()
        
        return rectified_poses
    
        
    def train_once_pose_v2(self, batch):
        '''
        注意两点,这个优化pose分别体现在 渲染位姿 和 其他帧的 gaussians;
        We have global_kf_id for each gaussian, so can we optimize poses(except first k poses to keep sequent) using global kf_id after training one batch?
        (1) We need to record a independent poses-timestamps in mapper. 
        '''
        poses              = batch["poses"]                   # (N, 4, 4)
        images             = batch["images"]                  # (N, 344, 616, 3)
        depths             = batch["depths"]                  # (N, 344, 616, 1)
        depths_cov         = batch["depths_cov"]              # (N, 344, 616, 1) 
        intrinsic_dict     = batch["intrinsic"]               # {'fu', 'fv', 'cu', 'cv', 'H', 'W'}
        batch_global_kf_id = batch['global_kf_id']
        
        # STEP 1 Prepare variables & Setup optimizer.
        localbatch_globalkf_id_mask = []
        for i in range(batch_global_kf_id.shape[0]):
            localbatch_globalkf_id_mask.append(self._globalkf_id==batch_global_kf_id[i])
        
        optimizing_c2c_num = poses.shape[0]
        
        optimize_c2c_ts = torch.zeros(3, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs = torch.zeros(4, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs[:, -1] += 1.0
        
        
        optimize_c2c_ts   = torch.zeros(3, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs   = torch.zeros(4, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs[:, -1] += 1.0
        
        optimize_c2c_ts   = nn.Parameter(optimize_c2c_ts, requires_grad=True)
        optimize_c2c_qs   = nn.Parameter(optimize_c2c_qs, requires_grad=True)
        
        if '_c2c_lr' in self.cfg['training_args']['lr'].keys():
            c2c_lr = self.cfg['training_args']['lr']['_c2c_lr']
        else:
            c2c_lr = 0.005
        param_groups = [
            {'params': optimize_c2c_ts, 'lr': c2c_lr},  # 前3列，学习率为 0.001
            {'params': optimize_c2c_qs, 'lr': 0.0}     # 后4列，学习率为 0
        ]
        self.pose_optimizer = torch.optim.Adam(param_groups)
        
        # localbatch_gaussians_mask = torch.bitwise_and(self._globalkf_id>=batch['global_kf_id'].min(), self._globalkf_id<=batch['global_kf_id'].max())
        
        # STEP 2 Get new gaussians' _xyz and render.
        train_pose_iters = 20
        for curr_iter in range(train_pose_iters):
            
            optimize_c2c_tqs = torch.cat([optimize_c2c_ts, optimize_c2c_qs], dim=1)
            
            curr_id = random.randint(0, poses.shape[0]-1)
            c2w     = poses[curr_id]
            
            # new_xyz = get_new_xyz(self, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, curr_id)
            xyz_bias = get_xyz_bias_multi(self, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, self._xyz)
            new_xyz = self._xyz + xyz_bias
            new_xyz = get_new_xyz_single(new_xyz, poses[curr_id], optimize_c2c_tqs[curr_id])
            
            
            w2c = torch.linalg.inv(c2w)
            camera = get_camera(w2c, intrinsic_dict)
            pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
            raster_settings = GaussianRasterizationSettings(image_height=int(camera.height), image_width=int(camera.width),
                                tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
                                bg = torch.zeros(3, device=self.device), scale_modifier=1.0,
                                viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
                                sh_degree=0, campos=camera.camera_center,
                                prefiltered=False, debug=False,
                                pixel_mask = pixel_mask)
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
            means2D        = screenspace_points
            opacity        = self.get_property('_opacity') # (N, 1)
            scales         = self.get_property('_scaling') # (N, 3)
            rotations_old  = self.get_property('_rotation') # (N, 4)
            colors_precomp = self.get_property('_rgb') # (N, 3)
            self._zeros    = self.get_property('_zeros') # (N, 2)
            render_zeros   = self._zeros
            
            rendered_image, radii, allmap = rasterizer(
                                                means3D = new_xyz,
                                                means2D = means2D,
                                                shs = None,
                                                colors_precomp = colors_precomp,
                                                opacities = opacity,
                                                scales = scales,
                                                rotations = rotations_old,
                                                scores    = render_zeros,
                                                cov3D_precomp = None
                                                )
            
            valid_mask = torch.bitwise_and(depths[curr_id].squeeze(0).squeeze(-1) > 0, depths[curr_id].squeeze(0).squeeze(-1) < 10) # (H, W)
            gt_rgb     = images[curr_id].permute(2,0,1) # (3, H, W)
                        
            rgb_loss   = 0.8 * l1_loss(rendered_image, gt_rgb, valid_mask) +\
                        0.2 * (1.0 - ssim_loss(rendered_image, gt_rgb, valid_mask))
                        
            rgb_loss.backward()
            
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad()
        
        optimize_c2c_tqs = torch.cat([optimize_c2c_ts.detach(), optimize_c2c_qs.detach()], dim=1)
        rectified_poses = torch.matmul(SE3(optimize_c2c_tqs).matrix(), poses)
        
        self._xyz = (self._xyz.detach() + xyz_bias.detach()).requires_grad_(True)
        self.setup_optimizer()
        self.optimizer.zero_grad()
        
        return rectified_poses
    
    
    # Train once pose cur.
    def train_once_pose_ablationcurpose(self, batch):
        '''
        注意两点,这个优化pose分别体现在 渲染位姿 和 其他帧的 gaussians;
        We have global_kf_id for each gaussian, so can we optimize poses(except first k poses to keep sequent) using global kf_id after training one batch?
        (1) We need to record a independent poses-timestamps in mapper. 
        '''
        poses              = batch["poses"]                   # (N, 4, 4)
        images             = batch["images"]                  # (N, 344, 616, 3)
        depths             = batch["depths"]                  # (N, 344, 616, 1)
        depths_cov         = batch["depths_cov"]              # (N, 344, 616, 1) 
        intrinsic_dict     = batch["intrinsic"]               # {'fu', 'fv', 'cu', 'cv', 'H', 'W'}
        batch_global_kf_id = batch['global_kf_id']
        
        # STEP 1 Prepare variables & Setup optimizer.
        localbatch_globalkf_id_mask = []
        for i in range(batch_global_kf_id.shape[0]):
            localbatch_globalkf_id_mask.append(self._globalkf_id==batch_global_kf_id[i])
        
        optimizing_c2c_num = poses.shape[0]
        
        optimize_c2c_ts = torch.zeros(3, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs = torch.zeros(4, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs[:, -1] += 1.0
        
        
        optimize_c2c_ts   = torch.zeros(3, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs   = torch.zeros(4, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(optimizing_c2c_num, 1)
        optimize_c2c_qs[:, -1] += 1.0
        
        optimize_c2c_ts   = nn.Parameter(optimize_c2c_ts, requires_grad=True)
        optimize_c2c_qs   = nn.Parameter(optimize_c2c_qs, requires_grad=True)
        
        if '_c2c_lr' in self.cfg['training_args']['lr'].keys():
            c2c_lr = self.cfg['training_args']['lr']['_c2c_lr']
        else:
            c2c_lr = 0.005
        param_groups = [
            {'params': optimize_c2c_ts, 'lr': c2c_lr},  # 前3列，学习率为 0.001
            {'params': optimize_c2c_qs, 'lr': 0.0}     # 后4列，学习率为 0
        ]
        self.pose_optimizer = torch.optim.Adam(param_groups)
        
        # localbatch_gaussians_mask = torch.bitwise_and(self._globalkf_id>=batch['global_kf_id'].min(), self._globalkf_id<=batch['global_kf_id'].max())
        
        # STEP 2 Get new gaussians' _xyz and render.
        train_pose_iters = 20
        for curr_iter in range(train_pose_iters):
            
            optimize_c2c_tqs = torch.cat([optimize_c2c_ts, optimize_c2c_qs], dim=1)
            
            curr_id = random.randint(0, poses.shape[0]-1)
            c2w     = poses[curr_id]
            
            # new_xyz = get_new_xyz(self, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, curr_id)
            # xyz_bias = get_xyz_bias_multi(self, poses, optimize_c2c_tqs, localbatch_globalkf_id_mask, self._xyz)
            new_xyz = self._xyz
            new_xyz = get_new_xyz_single(new_xyz, poses[curr_id], optimize_c2c_tqs[curr_id])
            
            
            w2c = torch.linalg.inv(c2w)
            camera = get_camera(w2c, intrinsic_dict)
            pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
            raster_settings = GaussianRasterizationSettings(image_height=int(camera.height), image_width=int(camera.width),
                                tanfovx=camera.tanfovx, tanfovy=camera.tanfovy,
                                bg = torch.zeros(3, device=self.device), scale_modifier=1.0,
                                viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
                                sh_degree=0, campos=camera.camera_center,
                                prefiltered=False, debug=False,
                                pixel_mask = pixel_mask)
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            screenspace_points = torch.zeros_like(self._xyz, dtype=self.dtype, requires_grad=True, device="cuda") + 0
            means2D        = screenspace_points
            opacity        = self.get_property('_opacity') # (N, 1)
            scales         = self.get_property('_scaling') # (N, 3)
            rotations_old  = self.get_property('_rotation') # (N, 4)
            colors_precomp = self.get_property('_rgb') # (N, 3)
            self._zeros    = self.get_property('_zeros') # (N, 2)
            render_zeros   = self._zeros
            
            rendered_image, radii, allmap = rasterizer(
                                                means3D = new_xyz,
                                                means2D = means2D,
                                                shs = None,
                                                colors_precomp = colors_precomp,
                                                opacities = opacity,
                                                scales = scales,
                                                rotations = rotations_old,
                                                scores    = render_zeros,
                                                cov3D_precomp = None
                                                )
            
            # valid_mask = depths[curr_id].squeeze(0).squeeze(-1) > 0 # (H, W)
            valid_mask = torch.bitwise_and(depths[curr_id].squeeze(0).squeeze(-1) > 0, depths[curr_id].squeeze(0).squeeze(-1) < 10) # (H, W)
            gt_rgb     = images[curr_id].permute(2,0,1) # (3, H, W)
                        
            rgb_loss   = 0.8 * l1_loss(rendered_image, gt_rgb, valid_mask) +\
                        0.2 * (1.0 - ssim_loss(rendered_image, gt_rgb, valid_mask))
                        
            rgb_loss.backward()
            
            self.pose_optimizer.step()
            self.pose_optimizer.zero_grad()
        
        optimize_c2c_tqs = torch.cat([optimize_c2c_ts.detach(), optimize_c2c_qs.detach()], dim=1)
        rectified_poses = torch.matmul(SE3(optimize_c2c_tqs).matrix(), poses)
        
        self._xyz = (self._xyz.detach()).requires_grad_(True)
        self.setup_optimizer()
        self.optimizer.zero_grad()
        
        return rectified_poses
    
    
    
    # TTD 2024/11/30
    def train_once_pose(self, batch):
        # TTD 2024/12/05
        # Ablation on pose refinement.
        # rectified_poses = self.train_once_pose_v2(batch)
        rectified_poses = self.train_once_pose_ablationcurpose(batch)
        return rectified_poses
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
