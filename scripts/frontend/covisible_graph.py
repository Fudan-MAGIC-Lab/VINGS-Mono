import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
from lietorch import SE3
from frontend.modules.corr import CorrBlock, AltCorrBlock
import frontend.geom.projective_ops as pops
import matplotlib.pyplot as plt
import cv2
from contextlib import nullcontext
from frontend.depth_video import DepthVideo
import matplotlib.cm as cm
import matplotlib

class CovisibleGraph:
    def __init__(self, video: DepthVideo, update_op, device="cuda:0", corr_impl="volume", config = None):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = config['frontend']['max_factors']
        self.corr_impl = corr_impl
        self.upsample = True

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        self.far_threshold  = config['frontend']['far_threshold']
        self.inac_range     = config['frontend']['inac_range']
        self.mask_threshold = config['frontend']['mask_threshold']
        self.img_count = 0
        self.profiler = None
        self.profiler_frame_idx = None

        self.skip_edge       = config['frontend']['skip_edge']
        self.frontend_window = config['frontend']['frontend_window']
        
        # simple online visualization
        self.show_covisible_graph = False
        self.show_oldest_disparity = False
        self.show_flow_and_weight = False

    def _profile(self, stage):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.time(stage, self.profiler_frame_idx)

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """
        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """


        with self.video.get_lock():
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

            self.video.tstamp[ix] = self.video.tstamp[ix+1] # BUG fix

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)

    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False, marg = False):
        """ run update operator on factor graph """

        # self.video.logger.info('update')

        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0) # 1,2,4,48,96

        with self._profile("covisible_graph_corr"):
            corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj, self.upsample)
        
        '''
        self.video.disps.shape: [80, 43, 77]
                  upmask.shape: [1, 8, 576, 43, 77]
        '''
        
        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)
            
        # self.video.logger.info('ba')

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            if self.upsample: 
                self.damping[torch.unique(self.ii)] = damping

            if use_inactive: # use_inactive = True
                m = (self.ii_inac >= t0 - self.inac_range) & (self.jj_inac >= t0 - self.inac_range)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)
            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            # Some real-time visualization for debugging
            # 1) Disparity
            if self.show_oldest_disparity:
                disp_show_front = self.video.disps[self.ii[0]].cpu().numpy()
                disp_show_front = cv2.resize(disp_show_front,[disp_show_front.shape[1]*8,disp_show_front.shape[0]*8],interpolation =  cv2.INTER_NEAREST)
                disp_show_front= disp_show_front.astype(np.float32)
    
                normalizer = matplotlib.colors.Normalize(vmin=-0.2, vmax=1.0)
                mapper = cm.ScalarMappable(norm=normalizer,cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_show_front)[:, :, :3] * 255).astype(np.uint8)
                colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                cv2.imshow('colormapped_im',colormapped_im)
                cv2.waitKey(1)

            # 2) Optical flow and weight
            if self.show_flow_and_weight:
                rgb = self.video.images[torch.max(self.ii)].cpu().numpy().transpose(1,2,0)
                new_flow_id = torch.where(torch.logical_and(self.ii==torch.max(self.ii),self.jj==torch.max(self.ii)-5))[0][0].item()
                weight_cpu = weight[0,new_flow_id].cpu().numpy().astype(np.float32)
                weight_cpu = np.linalg.norm(weight_cpu,axis=2)
                normalizer = matplotlib.colors.Normalize(vmin=-0.0, vmax=1.5)
                mapper = cm.ScalarMappable(norm=normalizer,cmap='jet')
                colormapped_im = (mapper.to_rgba(weight_cpu)[:, :, :3] * 255).astype(np.uint8)
                colormapped_im = cv2.cvtColor(colormapped_im,cv2.COLOR_RGB2BGR)
                colormapped_im = cv2.resize(colormapped_im,[rgb.shape[1],rgb.shape[0]])
                colormapped_im = cv2.addWeighted(rgb,0.5,colormapped_im,0.5,0)
                absflow = (self.target[0,new_flow_id] - self.coords0).cpu().numpy()
                for iii in range(0,absflow.shape[0],4):
                    for jjj in range(0,absflow.shape[1],4):
                        colormapped_im = cv2.line(colormapped_im, (jjj * 8,iii * 8),(int(round((jjj-absflow[iii,jjj,0])* 8)) ,int(round((iii-absflow[iii,jjj,1]) * 8))),(255,255,255),1,cv2.LINE_AA)

                cv2.imshow('weight_cpu',colormapped_im)
                self.img_count += 1

            # 3) Covisible graph
            if self.show_covisible_graph:
                i0 = min(ii)
                i1 = max(ii)
                ppp = SE3(self.video.poses[i0:(i1+1)]).inv().matrix()[:,0:3,3].cpu().numpy()
                # [:,:3].cpu().numpy()
                scale = max(max(ppp[:,0]) - min(ppp[:,0]),max(ppp[:,1]) - min(ppp[:,1]))
                ppp[:,0] -= np.mean(ppp[:,0])
                ppp[:,1] = -(ppp[:,1]- np.mean(ppp[:,1]))
                ppp *= max(round(1/scale * 200 / 50)*50,50)
                ppp += 500
                mmm = np.zeros([1000,1000],dtype=np.uint8)
                for iii in range(0,i1+1-i0):
                    mmm = cv2.circle(mmm,(int(round(ppp[iii,0])),int(round(ppp[iii,1]))),4,255,0)
                for iii in range(self.ii_inac[m].shape[0]):
                    iiii = self.ii_inac[m][iii]-i0
                    jjjj = self.jj_inac[m][iii]-i0
                    mmm = cv2.line(mmm,(int(round(ppp[iiii,0])),int(round(ppp[iiii,1]))),(int(round(ppp[jjjj,0])),int(round(ppp[jjjj,1]))),128,1)
                for iii in range(self.ii.shape[0]):
                    iiii = self.ii[iii]-i0
                    jjjj = self.jj[iii]-i0
                    mmm = cv2.line(mmm,(int(round(ppp[iiii,0])),int(round(ppp[iiii,1]))),(int(round(ppp[jjjj,0])),int(round(ppp[jjjj,1]))),255,1)
                cv2.imshow('window',mmm)

            ## Tricks for better performance
            # 1) downweight far points
            # TTD 2024/11/03
            if (self.far_threshold > 0 and self.video.imu_enabled) or (self.mask_threshold > 0 and self.video.visual_only_init):
                disp_mask = (self.video.disps < self.far_threshold)
                mask = disp_mask[ii, :, :]
                weight[:, mask] /= 1000.0
            
            # 2) downweight far points
            # TTD 2024/11/03
            if (self.mask_threshold > 0 and self.video.imu_enabled) or (self.mask_threshold > 0 and self.video.visual_only_init):
                pose0 = SE3(self.video.poses[ii])
                pose1 = SE3(self.video.poses[jj])
                pose01 = pose0*pose1.inv()
                mask = torch.norm(pose01.translation()[:,:3],dim=1) < self.mask_threshold
                weight[:,mask,:,:,:] /= 1000.0
            
            # 3) downweight edges related to the newest frame
            downweight_newframe = True
            if downweight_newframe:
                weight[:,ii==max(ii)] /= 10.0
                weight[:,jj==max(jj)] /= 4.0

            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # Dense bundle adjustment
            with self._profile("covisible_graph_ba"):
                self.video.ba(target, weight, damping, ii, jj, t0, t1,
                    itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
            if self.upsample:
                '''
                upmask.shape = [1, 8, 576, 43, 77]
                '''
                with self._profile("covisible_graph_upsample"):
                    self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)
        
        cc = ii.shape[0]

        # Opportunistic "skip" edges in the graph
        if self.skip_edge:
            if torch.max(ii) - torch.min(ii) == self.frontend_window - 1:
                jj_add = torch.min(ii) + torch.tensor(self.skip_edge)
                jj_add = jj_add[jj_add>0]
                ii_add = torch.zeros_like(jj_add) + torch.max(ii)
                jj = torch.cat([jj,jj_add])
                ii = torch.cat([ii,ii_add])

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        es = []
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                if (i-t0)*(t-t1) + (j-t1) >=0:
                    d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if k >= cc:
                continue

            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf
        
        if ii.shape[0] > cc:
            ix = torch.argsort(d[cc:ii.shape[0]])
            if d[cc + ix[0]] < thresh and  d[cc + ix[0]]  > 0:
                es.append((ii[cc+ix[0]],jj[cc+ix[0]]))
                es.append((jj[cc+ix[0]],ii[cc+ix[0]]))

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
