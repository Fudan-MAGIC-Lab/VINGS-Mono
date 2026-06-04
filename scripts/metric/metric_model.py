import torch
import cv2
import numpy as np
import sys
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
# from metric.metric3d import Metric3D_Model

class Metric_Model:
    def __init__(self, cfg, u_scale=None, v_scale=None):
        self.cfg = cfg
        device = self.cfg['device']['tracker']
        
        ''' ZoeDepth, cost time: 5.06s.
        repo = "isl-org/ZoeDepth"
        # Online.
        # self.predictor = torch.hub.load(repo, "ZoeD_N", pretrained=True).to(device)
        # Offline, git clone https://github.com/isl-org/ZoeDepth
        self.predictor = torch.hub.load("/data/wuke/workspace/ZoeDepth", "ZoeD_N", source="local", device=device, pretrained=True)
        '''
        
        '''
        Metric3D
        '''
        ckpt_path = REPO_ROOT / 'ckpts' / 'metric_depth_vit_small_800k.pth'
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Metric checkpoint not found: {ckpt_path}. "
                "Download metric_depth_vit_small_800k.pth into ckpts/ before enabling metric."
            )
        # self.predictor = Metric(checkpoint='/data/wuke/workspace/droid_metric/weights/metric_depth_vit_small_800k.pth', model_name='v2-S')
        self.predictor = Metric(checkpoint=ckpt_path, model_name='v2-S')
        if u_scale is None:
            # u_scale, v_scale = self.cfg['frontend']['image_size'][0]/self.cfg['intrinsic']['H'], self.cfg['frontend']['image_size'][1]/self.cfg['intrinsic']['W']
            u_scale, v_scale = 1.0, 1.0
            # self.intr  = np.array([cfg['intrinsic']['fv'], cfg['intrinsic']['fu'], cfg['intrinsic']['cv'], cfg['intrinsic']['cu']])
            self.intr  = np.array([cfg['intrinsic']['fv']*v_scale, cfg['intrinsic']['fu']*u_scale, cfg['intrinsic']['cv']*v_scale, cfg['intrinsic']['cu']*u_scale])
        else:
            self.intr  = np.array([cfg['intrinsic']['fv']*v_scale, cfg['intrinsic']['fu']*u_scale, cfg['intrinsic']['cv']*v_scale, cfg['intrinsic']['cu']*u_scale])
        self.d_max = 300.0
    
    def predict(self, img):
        '''
        img: (3, H, W), np.array
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
        if isinstance(img, torch.Tensor):
            img_numpy = img.cpu().permute(1,2,0).numpy()
        depth = self.predictor(rgb_image=img_numpy, intrinsic=self.intr, d_max=self.d_max)
        depth = cv2.resize(depth, (img_numpy.shape[1], img_numpy.shape[0]))
        depth = torch.tensor(depth, device=img.device, dtype=torch.float32)
        return depth
    
