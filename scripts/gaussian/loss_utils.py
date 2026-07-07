import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from gaussian.normal_utils import depth_propagate_normal

def l1_loss(network_output, gt, mask=None):
    '''
    network_output.shape = (3, H, W)
    mask.shape           = (H, W)
    '''
    if mask is not None:
        return torch.abs((network_output - gt)[:, mask]).mean()
    else:
        return torch.abs((network_output - gt)).mean()

def weighted_l1_loss(network_output, gt, mask=None, weight=None):
    '''
    network_output.shape = (3, H, W)
    mask.shape           = (H, W)
    '''
    return (torch.abs((network_output - gt)[:, mask])*(weight[:, mask])).mean()
    


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, valid_mask, window, window_size, channel, size_average=True):

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map[:, valid_mask].mean()
    else:
        return ssim_map[:, valid_mask].mean(1).mean(1).mean(1)   


def ssim_img(img1, img2):
    channel = img1.size(-3)
    window_size = 11
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map

def ssim_loss(img1, img2, valid_mask, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, valid_mask, window, window_size, channel, size_average)


# Copy from MonoDepth2: monodepth2-master/layers.py
def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness.
    disp.shape=(1, 375, 1242), im.shape=(3, 375, 1242)
    """
    grad_disp_x = torch.abs(disp[:, :, :-1] - disp[:, :, 1:])
    grad_disp_y = torch.abs(disp[:, :-1, :] - disp[:, 1:, :])
    grad_img_x = torch.mean(torch.abs(img[:, :, :-1] - img[:, :, 1:]), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :-1, :] - img[:, 1:, :]), 0, keepdim=True)
    
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    
    
    grad_disp_x, grad_disp_y = grad_disp_x[grad_disp_x>0], grad_disp_y[grad_disp_y>0]
    return grad_disp_x.mean() + grad_disp_y.mean()


def get_loss(cfg, pred_dict, gt_dict):

    sky_mask = gt_dict['rgb'].sum(axis=0) == 0.0 # (H, W)
    valid_mask = torch.bitwise_and(~sky_mask, gt_dict['depth'].sum(axis=0) > 0.0) # (H, W)
    pixel_mask = pred_dict.get('pixel_mask')
    if pixel_mask is not None:
        valid_mask = torch.bitwise_and(valid_mask, pixel_mask)

    ssim_timeidx_weight   = 0.2
    normal_timeidx_weight = 1.0
    accum_timeidx_weight  = 1.0
    
    if not cfg['use_sky']:
        Ll1 = l1_loss(pred_dict['rgb'], gt_dict['rgb'], valid_mask)
        rgb_loss = 0.8 * Ll1 + ssim_timeidx_weight * (1.0 - ssim_loss(pred_dict['rgb'], gt_dict['rgb'], valid_mask))
    else:
        sky_valid_mask = torch.ones_like(valid_mask) if pixel_mask is None else pixel_mask
        Ll1 = l1_loss(pred_dict['rgb'], gt_dict['sky_rgb'], sky_valid_mask)
        rgb_loss = 0.8 * Ll1 + ssim_timeidx_weight * (1.0 - ssim_loss(pred_dict['rgb'], gt_dict['sky_rgb'], sky_valid_mask))
    
    rend_normal = pred_dict['normal']
    surf_normal = pred_dict['surf_normal']
    normal_loss = (1 - (rend_normal * surf_normal).sum(dim=0)[valid_mask]).mean() * normal_timeidx_weight
    alpha_loss = (pred_dict['accum'][:, sky_mask]).mean() * accum_timeidx_weight
    
    # depth_loss = l1_loss(pred_dict["depth"], gt_dict['depth'], valid_mask)
    weight     = 1./gt_dict['depth_cov'] # torch.log(1+gt_dict['depth'])
    depth_loss = weighted_l1_loss(pred_dict["depth"], gt_dict['depth'], valid_mask, weight)
    
    dist_loss  = (pred_dict['dist'][:, valid_mask]).mean()
    
    
    disp = 1/(pred_dict["depth"]+1e-5)
    disp[:, ~valid_mask] *= 0
    
    # if 'smooth_loss' in cfg['training_args']['loss_weights'].keys() and cfg['training_args']['loss_weights']['smooth_loss'] > 0:
    #     smooth_loss = get_smooth_loss(disp, gt_dict['rgb'])
    # else:
    #     smooth_loss = 0.0

    loss_cfg = cfg['training_args']['loss_weights']
    total_loss = loss_cfg['rgb_loss'] * rgb_loss + \
                 loss_cfg['normal_loss'] * normal_loss + \
                 loss_cfg['alpha_loss'] * alpha_loss + \
                 loss_cfg['depth_loss'] * depth_loss + \
                 loss_cfg['dist_loss'] * dist_loss

    return total_loss


def get_pixel_mask(pred_dict, gt_dict):
    
    # 选Loss大的;
    '''
    with torch.no_grad():
        tile_size = 16
        h, w = pred_dict['rgb'].shape[-2:]
        l1_loss   = torch.sum(torch.abs(pred_dict['rgb']-gt_dict['rgb']), dim=0) # (H, W)
        # All-zeros-Padding l1_loss to an integer multiple of 16. (Don't use zero padding.)
        pad_h = (tile_size - (h % tile_size)) % tile_size
        pad_w = (tile_size - (w % tile_size)) % tile_size
        l1_loss_padded = torch.nn.functional.pad(l1_loss, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        patches = l1_loss_padded.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size)
        patches = patches.contiguous().view(-1, tile_size, tile_size)  # (num_patches, 16, 16)
        medians = patches.reshape(-1, tile_size*tile_size).median(dim=1, keepdim=True).values
        medians = medians.reshape(-1, 1, 1)
        medians[medians<1e-4] = 1e-4
        masks = (patches >= medians)
        
        h_tile = l1_loss_padded.shape[0] // tile_size
        w_tile = l1_loss_padded.shape[1] // tile_size
        masks = masks.view(h_tile, w_tile, tile_size, tile_size)
        masks = masks.permute(0, 2, 1, 3).contiguous()  # 调整维度顺序
        high_loss_mask = masks.view(h_tile * tile_size, w_tile * tile_size)
        high_loss_mask = high_loss_mask[:h, :w]
        
        high_loss_mask = torch.ones_like(high_loss_mask)
    '''
    
    # 随机选一半呢;
    height, width = pred_dict['rgb'].shape[-2:]
    mask = torch.ones(height * width, dtype=torch.bool).cuda()
    mask[: len(mask) // 4] = False
    mask = mask[torch.randperm(mask.size(0))]
    high_loss_mask = mask.reshape(height, width)
    high_loss_mask = torch.ones_like(high_loss_mask)
    high_loss_mask[:height//2,:] = False
    
    return high_loss_mask # (H, W)