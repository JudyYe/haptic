import torch
import torch.nn as nn

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        assert gt_keypoints_2d.ndim == 3 and pred_keypoints_2d.ndim == 3
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()


class AccNormLoss(nn.Module):
    def __init__(self, loss_type='diffnorm') -> None:
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred_kpts, gt_kpts):
        """

        :param pred_kpts: (B, T, J, 3)
        :param gt_kpts: (B, T, J, 3)
        """
        pred_vel = pred_kpts[:, 1:, ... , :3] - pred_kpts[:, :-1, ..., :3]  # (B, T-1, J, 3)
        gt_vel = gt_kpts[:, 1:, ..., :3] - gt_kpts[:, :-1, ..., :3]

        pred_acc = pred_vel[:, 1: ] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        
        # you can compute norm of diff, or diff of norm
        if self.loss_type == 'diffnorm':
            pred_norm = torch.linalg.norm(pred_acc, dim=-1)
            gt_norm = torch.linalg.norm(gt_acc, dim=-1)
            loss = (pred_norm - gt_norm).abs().sum()
        elif self.loss_type == 'normdiff':
            diff_norm = torch.linalg.norm(pred_acc - gt_acc, dim=-1)
            loss = diff_norm.sum()
        return loss
    
class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0, mode='xyz'):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        if pelvis_id is not None:
            if mode == 'xyz':
                pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
                gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
            elif mode == 'xy':
                pred_keypoints_3d[..., :2] = pred_keypoints_3d[..., :2] - pred_keypoints_3d[:, pelvis_id, :2].unsqueeze(dim=1)
                gt_keypoints_3d[..., :2] = gt_keypoints_3d[..., :2] - gt_keypoints_3d[:, pelvis_id, :2].unsqueeze(dim=1)

        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        MANO parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute MANO parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param, gt_param))
        return loss_param.sum()
