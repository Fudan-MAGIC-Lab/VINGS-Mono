from __future__ import annotations

from lietorch import SE3
import torch
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import gtsam
except ModuleNotFoundError:
    gtsam = None
def tq_to_matrix(tqs: torch.Tensor):
    """
    Convert a tensor of TQs to a matrix.
    """
    return SE3(tqs.cpu()).matrix()

def matrix_to_tq(matrix: torch.Tensor):
    """
    Convert a matrix to a tensor of TQs.
    """
    t = matrix[:, :3, -1].detach() # (N, 3)
    q = torch.tensor(Rotation.from_matrix(matrix[:, :3, :3].detach().cpu().numpy()).as_quat(), device=t.device) # (N, 4)
    
    return torch.concat([t, q], dim=1) # (N, 7)


def gtsam_pose_to_torch(pose: gtsam.Pose3, device, dtype):
    if gtsam is None:
        raise ModuleNotFoundError("gtsam is required for gtsam_pose_to_torch.")
    t = pose.translation()
    # q = pose.rotation().quaternion()
    # TTD 2024/04/21
    q = [pose.rotation().toQuaternion().w(),
         pose.rotation().toQuaternion().x(),
         pose.rotation().toQuaternion().y(),
         pose.rotation().toQuaternion().z()]
    
    return torch.tensor([t[0], t[1], t[2], q[1], q[2], q[3], q[0]], device=device, dtype=dtype)
