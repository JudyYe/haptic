import torch
import pytorch3d.transforms.rotation_conversions as rot_cvt
from pytorch3d.transforms.rotation_conversions import matrix_to_rotation_6d, matrix_to_axis_angle
from pytorch3d.renderer.cameras import look_at_rotation

def scale_matrix(scale, homo=True):
    """
    :param scale: (..., 3)
    :return: scale matrix (..., 4, 4)
    """
    dims = scale.size()[0:-1]
    if scale.size(-1) == 1:
        scale = scale.expand(*dims, 3)
    mat = torch.diag_embed(scale, dim1=-2, dim2=-1)
    if homo:
        mat = rt_to_homo(mat)
    return mat


def rt_to_homo(rot=None, t=None, s=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :param s: (..., 1)
    :return: (N, 4, 4) [R, t; 0, 1] sRX + t
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] += 1
    mat = torch.cat([mat, zeros], dim=-2)
    if s is not None:
        s = scale_matrix(s)
        mat = torch.matmul(mat, s)

    return mat


def axis_angle_t_to_matrix(axisang=None, t=None, s=None, homo=True):
    """
    :param axisang: (N, 3)
    :param t: (N, 3)
    :return: (N, 4, 4)
    """
    if axisang is None:
        axisang = torch.zeros_like(t)
    if t is None:
        t = torch.zeros_like(axisang)
    rot = rot_cvt.axis_angle_to_matrix(axisang)
    if homo:
        return rt_to_homo(rot, t, s)
    else:
        return rot




def inverse_rt_v2(mat=None, return_mat=True, ignore_scale=True):
    """
    [R, t] --> [R.T, -R.T + t]
    :param se3:
    :param mat:
    :param return_mat:
    :return
    """
    rot, trans, _ = homo_to_rt(mat, ignore_scale=ignore_scale)
    inv_mat = rt_to_homo(rot.transpose(-1, -2),  
        (-rot.transpose(-1, -2)) @ trans.unsqueeze(-1))
    if return_mat:
        return inv_mat
    else:
        return matrix_to_se3(inv_mat)



def homo_to_rt(mat, ignore_scale=False):
    """
    :param (N, 4, 4) [R, t; 0, 1]
    :return: rot: (N, 3, 3), t: (N, 3), s: (N, 3)
    """
    mat, _ = torch.split(mat, [3, mat.size(-2) - 3], dim=-2)
    rot_scale, trans = torch.split(mat, [3, 1], dim=-1)
    if ignore_scale:
        rot = rot_scale
        scale = None
    else:
        rot, scale = mat_to_scale_rot(rot_scale)

    trans = trans.squeeze(-1)
    return rot, trans, scale



def mat_to_scale_rot(mat):
    """s*R to s, R
    
    Args:
        mat ( ): (..., 3, 3)
    Returns:
        rot: (..., 3, 3)
        scale: (..., 3)
    """
    sq = torch.matmul(mat, mat.transpose(-1, -2))
    scale_flat = torch.sqrt(torch.diagonal(sq, dim1=-1, dim2=-2))  # (..., 3)
    scale_inv = scale_matrix(1/scale_flat, homo=False)
    rot = torch.matmul(mat, scale_inv)
    return rot, scale_flat



def matrix_to_se3(mat: torch.Tensor, rtn_scale=True) -> torch.Tensor:
    """
    :param mat: transformation matrix in shape of (N, 4, 4)
    :return: tensor in shape of (N, 9) rotation param (6) + translation (3)
    """
    rot, trans, scale = homo_to_rt(mat)
    rot = matrix_to_rotation_6d(rot)
    if rtn_scale:
        se3 = torch.cat([rot, trans, scale], dim=-1)
    else:
        se3 = torch.cat([rot, trans], dim=-1)
    return se3


def azel_to_rot_v2(azel, homo=False, t=None):
    # unit vector location to azimuth az, elevation el
    # Calculate the unit vector components
    azim, elev = azel.split([1, 1], -1)  # (..., 1)
    elev = -elev
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)

    xyz = torch.cat([x, y, z], -1)
    rot = look_at_rotation(xyz, up=((0, 1, 0),), device=azel.device)
    rot = rot.transpose(-1, -2)
    
    if homo:
        rot = rt_to_homo(rot, t)
    return rot