from __future__ import annotations

import torch
import numpy as np
from lietorch import SE3
from vings_utils.gtsam_utils import matrix_to_tq
import cv2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frontend_vo.vio_slam import VioSLAM

def tq_to_matrix(tqs: torch.Tensor):
    return SE3(tqs.cpu()).matrix()

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
def judge_and_package_v0(dba_fusion, intrinsics):
    '''
    The most ugly method, but it works on kitti.
    '''
    DEVICE = dba_fusion.video.disps.device
    # 确实，感觉像是这里的问题，因为只要roll了就一定会发生变化，但是其实可能并没有新增关键帧以及优化；
    if dba_fusion.frontend.new_frame_added:
        activate_kf_id = torch.arange(max(dba_fusion.video.count_save-8, 0), dba_fusion.video.count_save)
        if activate_kf_id.shape[0]>0:
            tstamps = dba_fusion.video.tstamp_save[activate_kf_id]
            rgbs   = dba_fusion.video.images_up_save[activate_kf_id][...,[2,1,0]]     # (N, H, W, 3)
            depths = 1./(dba_fusion.video.disps_up_save[activate_kf_id] + 1e-6).unsqueeze(-1) # (N, H, W, 1)
            depths_cov = dba_fusion.video.depths_cov_up_save[activate_kf_id].unsqueeze(-1) # (N, H, W, 1)
            
            N_frames = depths_cov.shape[0]
            cov_median = torch.tensor(np.median(depths_cov.cpu().numpy().reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1)
            
            zero_mask = torch.bitwise_or(depths > dba_fusion.cfg['middleware']['max_depth'], depths_cov>dba_fusion.cfg['middleware']['cov_times']*(cov_median))
            # zero_mask = depths > dba_fusion.cfg['middleware']['max_depth']
            depths[zero_mask] = 0.0
            # depths_cov[zero_mask] = 0.0
            
            w2c_tqs    = dba_fusion.video.poses_save[activate_kf_id]
            c2ws       = torch.linalg.inv(tq_to_matrix(w2c_tqs))
            intrinsic  = {'fu': intrinsics[1].to(DEVICE), 'fv':intrinsics[0].to(DEVICE), 'cu':intrinsics[3].to(DEVICE), 'cv':intrinsics[2].to(DEVICE), 'H':depths.shape[1], 'W':depths.shape[2]}
            rgbs[(depths.squeeze(-1)==0)] = 0.0
            pixel_mask = torch.ones_like(depths.squeeze(-1), dtype=torch.bool)
            viz_out = {'images': rgbs.to(DEVICE), 'depths': depths.to(DEVICE), 'depths_cov': depths_cov.to(DEVICE), 'poses': c2ws.to(DEVICE), 'viz_out_idx_to_f_idx': tstamps, 'intrinsic': intrinsic}
            viz_out['pixel_mask']   = pixel_mask
            # 2024/10/01 Add dict_f_id_to_global_kf_id.
            viz_out['global_kf_id'] = activate_kf_id.to(torch.long).to(DEVICE)
            viz_out['activate_kf_id'] = activate_kf_id.to(torch.long).to(DEVICE)
        else:
            viz_out = None
    else:
        viz_out = None
    return viz_out


def judge_and_package_v0_kitti360unsync(dba_fusion, intrinsics):
    DEVICE = dba_fusion.video.disps.device
    # 确实，感觉像是这里的问题，因为只要roll了就一定会发生变化，但是其实可能并没有新增关键帧以及优化；
    if dba_fusion.frontend.new_frame_added:
        activate_kf_id = torch.arange(max(dba_fusion.video.count_save-8, 0), dba_fusion.video.count_save)
        if activate_kf_id.shape[0]>0:
            tstamps = dba_fusion.video.tstamp_save[activate_kf_id]
            rgbs   = dba_fusion.video.images_up_save[activate_kf_id][...,[2,1,0]]     # (N, H, W, 3)
            depths = 1./(dba_fusion.video.disps_up_save[activate_kf_id] + 1e-6).unsqueeze(-1) # (N, H, W, 1)
            depths_cov = dba_fusion.video.depths_cov_up_save[activate_kf_id].unsqueeze(-1) # (N, H, W, 1)
            
            N_frames = depths_cov.shape[0]
            cov_median = torch.tensor(np.median(depths_cov.cpu().numpy().reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1)
            
            zero_mask = torch.bitwise_or(depths > dba_fusion.cfg['middleware']['max_depth'], depths_cov>dba_fusion.cfg['middleware']['cov_times']*(cov_median))
            # zero_mask = depths > dba_fusion.cfg['middleware']['max_depth']
            depths[zero_mask] = 0.0
            # depths_cov[zero_mask] = 0.0
            
            w2c_tqs    = dba_fusion.video.poses_save[activate_kf_id]
            c2ws       = torch.linalg.inv(tq_to_matrix(w2c_tqs))
            rgbs[(depths.squeeze(-1)==0)] = 0.0
            pixel_mask = torch.ones_like(depths.squeeze(-1), dtype=torch.bool)
            
            
            u_scale = dba_fusion.cfg['intrinsic']['new_H']/dba_fusion.cfg['intrinsic']['H']
            
            new_H = int(u_scale*dba_fusion.cfg['frontend']['image_size'][0])
            new_cu = new_H / 2
            
            intrinsic  = {'fu': intrinsics[1].to(DEVICE), 'fv':intrinsics[0].to(DEVICE), 'cu':new_cu, 'cv':intrinsics[2].to(DEVICE), 'H':new_H, 'W':depths.shape[2]}
            
            viz_out = {'images': rgbs[:, -new_H:].to(DEVICE), 'depths': depths[:, -new_H:].to(DEVICE), 'depths_cov': depths_cov[:, -new_H:].to(DEVICE), 'poses': c2ws.to(DEVICE), 'viz_out_idx_to_f_idx': tstamps, 'intrinsic': intrinsic}
            viz_out['pixel_mask']   = pixel_mask[:, -new_H:]
            # 2024/10/01 Add dict_f_id_to_global_kf_id.
            viz_out['global_kf_id'] = activate_kf_id.to(torch.long).to(DEVICE)
            viz_out['activate_kf_id'] = activate_kf_id.to(torch.long).to(DEVICE)
        else:
            viz_out = None
    else:
        viz_out = None
    return viz_out



def judge_and_package_v1(dba_fusion, intrinsics):
    DEVICE = dba_fusion.video.disps.device
    # 确实，感觉像是这里的问题，因为只要roll了就一定会发生变化，但是其实可能并没有新增关键帧以及优化；
    if dba_fusion.frontend.new_frame_added:
        
        # -    -    -    -    -    -    -    -    -    -    -    -    
        # 没任何用，纯粹是为了索引
        t0 = max(1, dba_fusion.frontend.graph.ii.min().item()+1)
        m  = (dba_fusion.frontend.graph.ii_inac >= t0 - dba_fusion.frontend.graph.inac_range) & (dba_fusion.frontend.graph.jj_inac >= t0 - dba_fusion.frontend.graph.inac_range)
        ii = torch.cat([dba_fusion.frontend.graph.ii_inac[m], dba_fusion.frontend.graph.ii], 0)
        jj = torch.cat([dba_fusion.frontend.graph.jj_inac[m], dba_fusion.frontend.graph.jj], 0)
        t1 = min(max(ii.max().item(), jj.max().item())+1, ii.shape[0])
        valid_localkf_id = torch.sort(torch.unique(ii[torch.arange(t0, t1)]))[0]
        local_to_global_bias      = dba_fusion.frontend.video.count_save - min(ii.min().item(), jj.min().item())
        dba_fusion.local_to_global_bias = local_to_global_bias
        # -    -    -    -    -    -    -    -    -    -    -    -    
        
        activate_kf_id = torch.arange(max(dba_fusion.video.count_save-8, 0), dba_fusion.video.count_save)
        if activate_kf_id.shape[0]>0:
            tstamps = dba_fusion.video.tstamp_save[activate_kf_id]
            rgbs   = dba_fusion.video.images_up_save[activate_kf_id][...,[2,1,0]]     # (N, H, W, 3)
            depths = 1./(dba_fusion.video.disps_up_save[activate_kf_id] + 1e-6).unsqueeze(-1) # (N, H, W, 1)
            depths_cov = dba_fusion.video.depths_cov_up_save[activate_kf_id].unsqueeze(-1) # (N, H, W, 1)
            
            N_frames = depths_cov.shape[0]
            cov_median = torch.tensor(np.median(depths_cov.cpu().numpy().reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1)
            
            zero_mask = torch.bitwise_or(depths > dba_fusion.cfg['middleware']['max_depth'], depths_cov>dba_fusion.cfg['middleware']['cov_times']*(cov_median))
            # zero_mask = depths > dba_fusion.cfg['middleware']['max_depth']
            depths[zero_mask] = 0.0
            # depths_cov[zero_mask] = 0.0
            depths_cov[depths==0] = 0
            
            w2c_tqs    = dba_fusion.video.poses_save[activate_kf_id]
            c2ws       = torch.linalg.inv(tq_to_matrix(w2c_tqs))
            intrinsic  = {'fu': intrinsics[1], 'fv':intrinsics[0], 'cu':intrinsics[3], 'cv':intrinsics[2], 'H':depths.shape[1], 'W':depths.shape[2]}
            rgbs[(depths.squeeze(-1)==0)] = 0.0
            pixel_mask = torch.ones_like(depths.squeeze(-1), dtype=torch.bool)
            viz_out = {'images': rgbs.to(DEVICE), 'depths': depths.to(DEVICE), 'depths_cov': depths_cov.to(DEVICE), 'poses': c2ws.to(DEVICE), 'viz_out_idx_to_f_idx': tstamps, 'intrinsic': intrinsic}
            viz_out['pixel_mask']   = pixel_mask
            # 2024/10/01 Add dict_f_id_to_global_kf_id.
            viz_out['global_kf_id'] = activate_kf_id.to(torch.long).to(DEVICE)
        else:
            viz_out = None
    else:
        viz_out = None
    return viz_out


def judge_and_package_v2(dba_fusion, intrinsics):
    '''
    v2比v1更超前, v1那里是只有当ii,jj图优化不涉及的时候才去将其进行替换;
    Mapping Relation: (ii) → ()
    '''
    DEVICE = dba_fusion.video.disps.device
    # 确实，感觉像是这里的问题，因为只要roll了就一定会发生变化，但是其实可能并没有新增关键帧以及优化；
    if dba_fusion.frontend.new_frame_added:
        
        if dba_fusion.cfg['mode']=='vo' or dba_fusion.cfg['mode']=='vio':
            t0 = max(1, dba_fusion.frontend.graph.ii.min().item()+1)
            m  = (dba_fusion.frontend.graph.ii_inac >= t0 - dba_fusion.frontend.graph.inac_range) & (dba_fusion.frontend.graph.jj_inac >= t0 - dba_fusion.frontend.graph.inac_range)
            ii = torch.cat([dba_fusion.frontend.graph.ii_inac[m], dba_fusion.frontend.graph.ii], 0)
            jj = torch.cat([dba_fusion.frontend.graph.jj_inac[m], dba_fusion.frontend.graph.jj], 0)
            t1 = max(ii.max().item(), jj.max().item()) + 1
            t1 = min(t1, ii.shape[0])
            valid_localkf_id = torch.sort(torch.unique(ii[torch.arange(t0, t1)]))[0]
            local_to_global_bias      = dba_fusion.frontend.video.count_save - min(ii.min().item(), jj.min().item())
            dba_fusion.local_to_global_bias = local_to_global_bias
            localkf_id_to_globalkf_id = valid_localkf_id + local_to_global_bias
        
        else:
            assert False, "Invalid mode."
        
        
        if valid_localkf_id.shape[0] > 0:
            tstamps = dba_fusion.video.tstamp[valid_localkf_id]
            rgbs   = dba_fusion.video.images[valid_localkf_id].permute(0,2,3,1)[...,[0,1,2]]/255.0 # (N, H, W, 3)
            depths = 1./(dba_fusion.video.disps_up[valid_localkf_id] + 1e-6).unsqueeze(-1)         # (N, H, W, 1)
            depths_cov = dba_fusion.video.depths_cov_up[valid_localkf_id].unsqueeze(-1)            # (N, H, W, 1)
            N_frames = depths_cov.shape[0]
            cov_median = torch.tensor(np.median(depths_cov.cpu().numpy().reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1)
            zero_mask = torch.bitwise_or(depths > dba_fusion.cfg['middleware']['max_depth'], depths_cov>dba_fusion.cfg['middleware']['cov_times']*(cov_median))
            # zero_mask = depths > dba_fusion.cfg['middleware']['max_depth']
            depths[zero_mask] = 0.0
            # depths_cov[depths==0] = depths_cov[depths>0].max()
            
            # depths_cov[zero_mask] = 0.0
            w2c_tqs    = dba_fusion.video.poses[valid_localkf_id]
            c2ws       = torch.linalg.inv(tq_to_matrix(w2c_tqs))
            intrinsic  = {'fu': intrinsics[1], 'fv':intrinsics[0], 'cu':intrinsics[3], 'cv':intrinsics[2], 'H':depths.shape[1], 'W':depths.shape[2]}
            rgbs[(depths.squeeze(-1)==0)] = 0.0
            pixel_mask = torch.ones_like(depths.squeeze(-1), dtype=torch.bool)
            viz_out = {'images': rgbs.to(DEVICE), 'depths': depths.to(DEVICE), 'depths_cov': depths_cov.to(DEVICE), 'poses': c2ws.to(DEVICE), 'viz_out_idx_to_f_idx': tstamps, 'intrinsic': intrinsic}
            viz_out['pixel_mask']   = pixel_mask
            # 2024/10/01 Add dict_f_id_to_global_kf_id.
            viz_out['global_kf_id'] = localkf_id_to_globalkf_id.to(torch.long).to(DEVICE)
        else:
            viz_out = None
    else:
        viz_out = None
    return viz_out


def judge_and_package_v3(dba_fusion, intrinsics):
    '''
    v2比v1更超前, v1那里是只有当ii,jj图优化不涉及的时候才去将其进行替换;
    Mapping Relation: (ii) → ()
    '''
    DEVICE = dba_fusion.video.disps.device
    # 确实，感觉像是这里的问题，因为只要roll了就一定会发生变化，但是其实可能并没有新增关键帧以及优化；
    if dba_fusion.frontend.new_frame_added:
        t0 = max(1, dba_fusion.frontend.graph.ii.min().item()+1)
        m  = (dba_fusion.frontend.graph.ii_inac >= t0 - dba_fusion.frontend.graph.inac_range) & (dba_fusion.frontend.graph.jj_inac >= t0 - dba_fusion.frontend.graph.inac_range)
        ii = torch.cat([dba_fusion.frontend.graph.ii_inac[m], dba_fusion.frontend.graph.ii], 0)
        jj = torch.cat([dba_fusion.frontend.graph.jj_inac[m], dba_fusion.frontend.graph.jj], 0)
        t1 = max(ii.max().item(), jj.max().item()) + 1
        t1 = min(t1, ii.shape[0])
        if t1 < t0: return None
        try:
            valid_localkf_id = torch.sort(torch.unique(ii[torch.arange(t0, t1)]))[0][:-1]
        except:
            pass
        local_to_global_bias      = dba_fusion.frontend.video.count_save - min(ii.min().item(), jj.min().item())
        dba_fusion.local_to_global_bias = local_to_global_bias
        localkf_id_to_globalkf_id = valid_localkf_id + local_to_global_bias
        
        if valid_localkf_id.shape[0] > 0:
            tstamps = dba_fusion.video.tstamp[valid_localkf_id]
            rgbs   = dba_fusion.video.images[valid_localkf_id].permute(0,2,3,1)[...,[0,1,2]]/255.0 # (N, H, W, 3)
            depths = 1./(dba_fusion.video.disps_up[valid_localkf_id] + 1e-6).unsqueeze(-1)         # (N, H, W, 1)
            depths_cov = dba_fusion.video.depths_cov_up[valid_localkf_id].unsqueeze(-1)            # (N, H, W, 1)
            N_frames = depths_cov.shape[0]
            cov_median = torch.tensor(np.median(depths_cov.cpu().numpy().reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1)
            zero_mask = torch.bitwise_or(depths > dba_fusion.cfg['middleware']['max_depth'], depths_cov>dba_fusion.cfg['middleware']['cov_times']*(cov_median))
            # zero_mask = depths > dba_fusion.cfg['middleware']['max_depth']
            depths[zero_mask] = 0.0
            depths_cov[depths==0] = depths_cov[depths>0].max()
            
            # depths_cov[zero_mask] = 0.0
            w2c_tqs    = dba_fusion.video.poses[valid_localkf_id]
            c2ws       = torch.linalg.inv(tq_to_matrix(w2c_tqs))
            intrinsic  = {'fu': intrinsics[1], 'fv':intrinsics[0], 'cu':intrinsics[3], 'cv':intrinsics[2], 'H':depths.shape[1], 'W':depths.shape[2]}
            rgbs[(depths.squeeze(-1)==0)] = 0.0
            pixel_mask = torch.ones_like(depths.squeeze(-1), dtype=torch.bool)
            viz_out = {'images': rgbs.to(DEVICE), 'depths': depths.to(DEVICE), 'depths_cov': depths_cov.to(DEVICE), 'poses': c2ws.to(DEVICE), 'viz_out_idx_to_f_idx': tstamps, 'intrinsic': intrinsic}
            viz_out['pixel_mask']   = pixel_mask
            # 2024/10/01 Add dict_f_id_to_global_kf_id.
            viz_out['global_kf_id'] = localkf_id_to_globalkf_id.to(torch.long).to(DEVICE)

            viz_out['valid_localkf_id'] = valid_localkf_id
        else:
            viz_out = None
    else:
        viz_out = None
    return viz_out




# TTD 2024/11/06
def judge_and_package_nerfslam(vio_slam: VioSLAM, intrinsics):
    
    if vio_slam.viz_out is not None:
        
        DEVICE = vio_slam.visual_frontend.cam0_idepths.device
        # Get viz_out.
        # viz_index, = torch.where(vio_slam.visual_frontend.viz_idx)
        # w2cs_tq    = torch.index_select(vio_slam.visual_frontend.gt_poses, 0, viz_index)
        # c2ws       = torch.linalg.inv(tq_to_matrix(w2cs_tq))
        # rgbs       = torch.index_select(vio_slam.visual_frontend.cam0_images, 0, viz_index)
        # depths     = 1./ (torch.index_select(vio_slam.visual_frontend.cam0_idepths_up, 0, viz_index)+1e-6)
        # depths_cov = torch.index_select(vio_slam.visual_frontend.cam0_depths_cov, 0, viz_index)
        # tstamps    = vio_slam.visual_frontend.cam0_timestamps[vio_slam.visual_frontend.viz_idx].to(torch.long)
        rgbs       = vio_slam.viz_out['cam0_images'].permute(0,2,3,1) / 255.0
        depths     = (1.0/vio_slam.viz_out['cam0_idepths_up'][..., None])
        depths_cov = vio_slam.viz_out['cam0_depths_cov_up'].unsqueeze(-1)
        c2ws       = torch.linalg.inv(SE3(vio_slam.viz_out['cam0_poses']).matrix())
        tstamps    = vio_slam.viz_out['viz_out_idx_to_f_idx']
        calibs     = vio_slam.viz_out["calibs"]
        
        N_frames = depths.shape[0]
        cov_median = torch.tensor(np.median(depths_cov.reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1, 1)
        zero_mask = torch.bitwise_or(depths_cov>(cov_median*vio_slam.cfg['middleware']['cov_times']), depths>vio_slam.cfg['middleware']['max_depth'])
        depths[zero_mask] = 0
        rgbs[zero_mask.squeeze(-1)] = 0
        
        
        camera_model = calibs[0].camera_model
        intrinsic    = {'fv': camera_model[0], 'fu': camera_model[1], 'cv': camera_model[2], 'cu': camera_model[3], 
                         'H': depths.shape[1], 'W': depths.shape[2]}
        # intrinsic  = {'fu': intrinsics[1], 'fv':intrinsics[0], 'cu':intrinsics[3], 'cv':intrinsics[2], 'H':depths.shape[1], 'W':depths.shape[2]}
        
        
        viz_out = {'images': rgbs.to(DEVICE), 'depths': depths.to(DEVICE), 'depths_cov': depths_cov.to(DEVICE), 
                   'poses': c2ws.to(DEVICE), 'viz_out_idx_to_f_idx': tstamps, 'intrinsic': intrinsic, 'global_kf_id': vio_slam.viz_out['viz_out_idx_to_f_idx']}
        '''
        batch["poses"]                   # (N, 4, 4)
        batch["images"]                  # (N, 344, 616, 3)
        batch["depths"]                  # (N, 344, 616, 1)
        batch["depths_cov"]              # (N, 344, 616, 1) 
        batch["intrinsic"]               # {'fu', 'fv', 'cu', 'cv', 'H', 'W'}
        '''

    else:
        viz_out = None
    return viz_out


def judge_and_package(dba_fusion, intrinsics):
    # Serve for KITTI.
    if not dba_fusion.cfg['mode'] == 'vo_nerfslam':
        if 'kitti360_unsync' not in dba_fusion.cfg['dataset']['module']:
            viz_out = judge_and_package_v3(dba_fusion, intrinsics)
        else:
            viz_out = judge_and_package_v0_kitti360unsync(dba_fusion, intrinsics)
            
        # viz_out = judge_and_package_v1(dba_fusion, intrinsics)
        # viz_out = judge_and_package_v0(dba_fusion, intrinsics)
        
    else:
        viz_out = judge_and_package_nerfslam(dba_fusion, intrinsics)
    return viz_out
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

def retrieve_to_tracker_v0(viz_out, dba_fusion, mapper):
    valid_localkf_id = viz_out['activate_kf_id'].cpu()
    dba_fusion.video.poses_save[valid_localkf_id] *= 0 
    dba_fusion.video.poses_save[valid_localkf_id] += matrix_to_tq(torch.linalg.inv(viz_out['poses'])).cpu()
    

def retrieve_to_tracker_v1(viz_out, dba_fusion, mapper):
    localkf_id_to_globalkf_id = viz_out['global_kf_id']
    valid_localkf_id = localkf_id_to_globalkf_id - dba_fusion.local_to_global_bias
    # Update dba_fusion.frontend.video.poses[valid_localkf_id], (dba_fusion.video.disps).
    dba_fusion.frontend.video.poses[valid_localkf_id] = matrix_to_tq(torch.linalg.inv(viz_out['poses'])).to(torch.float32)
    
    if hasattr(mapper, 'dnew_divide_dold'):
        dba_fusion.frontend.video.disps[valid_localkf_id]    /= mapper.dnew_divide_dold
        dba_fusion.frontend.video.disps_up[valid_localkf_id] /= mapper.dnew_divide_dold
        # dba_fusion.frontend.video.disps[valid_localkf_id.min():valid_localkf_id.max()]    /= mapper.dnew_divide_dold
        # dba_fusion.frontend.video.disps_up[valid_localkf_id.min():valid_localkf_id.max()] /= mapper.dnew_divide_dold
        del mapper.dnew_divide_dold
        
    # Check whether our relative idx are correct. √
    # torch.save({'dbaf_rgbs': dba_fusion.frontend.video.images[valid_localkf_id], 'vizout_rgbs': viz_out['images']}, '/data/wuke/workspace/VINGS-Mono/debug/check_relativeidx.pt')


def retrieve_to_tracker_v3(viz_out, dba_fusion, mapper):
    valid_localkf_id = viz_out['valid_localkf_id']
    dba_fusion.video.poses[valid_localkf_id] *= 0 
    dba_fusion.video.poses[valid_localkf_id] += matrix_to_tq(torch.linalg.inv(viz_out['poses']))
    
    

def retrieve_to_tracker(viz_out, dba_fusion, mapper):
    # retrieve_to_tracker_v0(viz_out, dba_fusion, mapper)
    # retrieve_to_tracker_v1(viz_out, dba_fusion, mapper)
    retrieve_to_tracker_v3(viz_out, dba_fusion, mapper)
    


# TTD 2024/11/06
class Fake_Calib:
    def __init__(self, resize_intrinsics):
        '''
        resize_intrinsics, (4, ), [fv, fu, cv, cu]
        '''
        self.body_T_cam = np.eye(4)
        self.camera_model = torch.tensor([resize_intrinsics[0], resize_intrinsics[1], resize_intrinsics[2], resize_intrinsics[3]])

def datapacket_to_nerfslam(our_data_dict, idx):
    '''
    > Change ours' dataset output format to nerfslam's input format.
    > We had better implement this ugly operation inside dataset class's __get_item__.
    > Namely tracker.track(input).
    > (1) NeRF-SLAM's frontend's input:
        {'data':
            'images': (1, 344, 616, 3), maxvalue=255
            'depths': (1, 344, 616, 1), 
            'is_last_frame': False
            'calibs': np.array([Fake_Calib(dba_fusion)])}
    > (2) Ours' input: 
        {'timestamp': float
               'rgb': (1, 3, H, W), maxvalue=255
         'intrinsic': (4, )}
    '''
    
    
    nerfslam_tracker_input_dict = {'data': {'k': np.arange(idx,idx+1),
                                   't_cams': np.array([idx]),
                                   'is_last_frame': False, 
                                   'images': np.array(our_data_dict['rgb'].permute(0,2,3,1).cpu()), # 这里check下255哈;
                                   'poses': np.eye(4)[np.newaxis,...],
                                   'calibs': np.array([Fake_Calib(our_data_dict['intrinsic'])])},
                                   'intrinsic': our_data_dict['intrinsic'],
                                   'global_kf_id': None}

    
    return nerfslam_tracker_input_dict
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   