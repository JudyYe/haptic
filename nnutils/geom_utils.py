import torch


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
