import os
import os.path as osp
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import wandb
from einops import rearrange
from yacs.config import CfgNode

from jutils import image_utils
from jutils.visualizer import Visualizer

from ..utils import MeshRenderer, SkeletonRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from . import MANO
from .backbones import create_backbone
from .discriminator import Discriminator
from .heads import build_mano_head
from .losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss

log = get_pylogger(__name__)



class HAPTIC(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        self.new_keys = []
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])

        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

        self.viz = Visualizer()
        self.share_backbone = self.cfg.MODEL.get("SHARE_BACKBONE", True)
        if self.share_backbone:
            self.backbone_bg = self.backbone
        else:
            raise NotImplementedError(f"Not implemented {self.share_backbone}")

    def get_parameters(self):
        all_params = list(self.mano_head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """

        if self.cfg.TRAIN.STAGE == 'ft':
            param_groups = [{'params': filter(lambda v: v.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]    
        else:
            raise NotImplementedError(f"Stage {self.cfg.TRAIN.STAGE} not implemented")

        optimizer = torch.optim.AdamW(params=param_groups,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc        

    def forward(self, batch: Dict, num_frames=-1) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False, num_frames=num_frames)
        
    def forward_step(self, batch: Dict, train: bool = False, num_frames=-1) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        if num_frames == -1:
            x = batch['img']  # (B, T, C, H, W)
            B, T, = x.shape[:2]
            batch_size = B*T
            x = x.view(B*T, *x.shape[2:])
        elif num_frames == 1:
            x = batch['img']
            T = 1
            batch_size = B = x.shape[0]

        conditioning_feats = self.backbone(x[:,:,:,32:-32])

        vid = batch['orig_img']  # (B, T, C, H, W)
        # vid_feats torch.Size([64, 384, 16, 16])
        vid = vid.view(B*T, *vid.shape[-3:])
        if self.share_backbone: 
            vid = vid[:,:,:,32:-32]
        vid_feats = self.backbone_bg(vid) # (B*T, C, H, W)
        vid_feats = rearrange(vid_feats, '(b t) c h w -> b t c h w', b=B, t=T)        

        pred_mano_params, pred_cam, depth, _ = self.mano_head(conditioning_feats, vid_feats, num_frames=T)
        # output: (B*T, ...)

        output = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}

        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        
        # TODO: compromised full cam. since we do not modify 2D alignment')
        full_cam = torch.cat([pred_cam[:, :2] ,depth], -1)
        output['pred_depth'] = depth
        output['full_cam_t'] = full_cam
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        # additional: 
        output["pred_keypoints_3d_full_cam"] = output['pred_keypoints_3d'] + full_cam.unsqueeze(1)
        output["pred_vertices_full_cam"] = output['pred_vertices'] + full_cam.unsqueeze(1)
        return output
    
    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        # Get annotations
        B, T = batch['img'].shape[:2]
        batch['keypoints_2d']  = rearrange(batch['keypoints_2d'], 'b t j xy -> (b t) j xy')
        batch['keypoints_3d'] = rearrange(batch['keypoints_3d'], 'b t j xyz -> (b t) j xyz')
        loss = self.compute_loss_frame(batch, output, train)
        
        # full projection
        gt_keypoints_3d = rearrange(batch['keypoints_3d'], '(b t) j xyz -> b t j xyz', b=B, t=T)
        pred_keypoints_3d = rearrange(output['pred_keypoints_3d_full_cam'], '(b t) j xyz -> b t j xyz', b=B, t=T)
        
        gt_keypoints_3d[..., :3] = gt_keypoints_3d[..., :3] - gt_keypoints_3d[:, 0:1, 0:1, :3]  # origin: T=0, J=0
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, 0:1, 0:1, :]

        loss_keypoints_3d_full_cam = self.keypoint_3d_loss(
            rearrange(pred_keypoints_3d, 'b t j xyz -> (b t) j xyz'), 
            rearrange(gt_keypoints_3d, 'b t j xyz -> (b t) j xyz'),
            pelvis_id=0, mode='none')
        # loss_keypoints_2d_full_cam = self.keypoint_2d_loss(output['pred_keypoints_2d_full_cam'], gt_keypoints_2d)

        losses = output['losses']
        losses['loss_keypoints_3d_full_cam'] = loss_keypoints_3d_full_cam.detach()
        # losses['loss_keypoints_2d_full_cam'] = loss_keypoints_2d_full_cam.detach()
        loss += self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_FULL_CAM'] * loss_keypoints_3d_full_cam 
                # self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_FULL_CAM'] * loss_keypoints_2d_full_cam

        if self.cfg.LOSS_WEIGHTS.get('DEPTH0_REG', 0) > 0:       
            depth0 = output['pred_depth'][::T]
            losses['loss_1st_depth'] = depth0.norm(dim=-1).mean()
            loss += self.cfg.LOSS_WEIGHTS['DEPTH0_REG'] * losses['loss_1st_depth']

        # acc_norm of full keypoints3d
        return loss
    
    def compute_loss_frame(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_mano_params['hand_pose'].shape[0]
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']  # this is normalized to [-0.5, 0.5]
        gt_keypoints_3d = batch['keypoints_3d']
        gt_mano_params = batch['mano_params']
        has_mano_params = batch['has_mano_params']
        is_axis_angle = batch['mano_params_is_axis_angle']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # Compute loss on MANO parameters
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_mano_params[k]
            loss_mano_params[k] = self.mano_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt.reshape(batch_size))

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach()
        output['losses'] = losses

        return loss

    @pl.utilities.rank_zero.rank_zero_only
    def wandb_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True, fname=None, pref=None) -> None:
        super().wandb_logging(batch, output, step_count, train, write_to_summary_writer, fname, pref)
        mode = 'train' if train else 'val'
        if pref is not None:
            mode = pref

        B, T, = batch['img'].shape[:2]
    
        J = 21
        # just visualize depth 
        kpts3d_gt = batch['keypoints_3d'].reshape(B, T, J, 4) # in camera frame  # (B, T, J, 3)
        depth_gt = kpts3d_gt[..., :3] - kpts3d_gt[:, 0:1, 0:1, :3]  # origin: T=0, J=0   
        depth_gt = depth_gt[:, :, 0, 2]  # (B, T)

        BT = B*T
        depth_pred = output['pred_keypoints_3d_full_cam'].reshape(B, T, J, 3)
        depth_pred = depth_pred - depth_pred[:, 0:1, 0:1, :3]
        depth_pred = depth_pred[:, :, 0, 2]  # (B, T)
        
        fig = self.viz.plt_depths([depth_gt, depth_pred], ['gt', 'pred'])
        if fname is None:
            fname = osp.join(self.logger.save_dir, f'images/depth_{mode}_{self.global_step:08d}.jpg')
        else:
            fname += '.jpg'
        os.makedirs(osp.dirname(fname), exist_ok=True)
        fig.savefig(fname)
        if write_to_summary_writer:
            self.logger.log_metrics({f'{mode}_vis/depth': wandb.Image(fname)}, self.global_step)
            fname = fname.replace('.jpg', '_bg.png').replace('depth_', '')
            images = batch["orig_img"].reshape(BT, *batch["orig_img"].shape[-3:])[:16]
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
            image_utils.save_images(images, fname[:-4], col=4)
            self.logger.log_metrics({f'{mode}_vis/bg': wandb.Image(fname)}, self.global_step)
    
    @pl.utilities.rank_zero_only
    def wandb_logging_one_frame(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True, fname=None, pref=None) -> None:
        mode = 'train' if train else 'val'
        if pref is not None:
            mode = pref
        
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            for loss_name, val in losses.items():
                self.logger.log_metrics({mode +'/' + loss_name: val.detach().item()}, step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        C, H, W = images.shape[-3:]
        images = images.reshape(-1, C, H, W)
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if fname is None:
            fname = osp.join(self.logger.save_dir, f'images/{mode}_{self.global_step:08d}')
        image_utils.save_images(predictions[None], fname)
        print('save to ', fname)
        if write_to_summary_writer:
            self.logger.log_metrics({f'{mode}_vis/predictions': wandb.Image(fname + '.png')}, step_count)
        return predictions

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        image_utils.save_images(predictions[None], osp.join(self.logger.save_dir, f'images/pred_{self.global_step:08d}'),)
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions


    def training_step_discriminator(self, batch: Dict,
                                    hand_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            hand_pose (torch.Tensor): Regressed hand pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = hand_pose.shape[0]
        gt_hand_pose = batch['hand_pose']
        gt_betas = batch['betas']
        gt_bs = len(gt_hand_pose)

        gt_rotmat = aa_to_rotmat(gt_hand_pose.view(-1,3)).view(gt_bs, -1, 3, 3)
        disc_fake_out = self.discriminator(hand_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        if "single" in joint_batch:
            single_batch = joint_batch['single']
        else:
            single_batch = None

        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_mano_params = output['pred_mano_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)

        # image-based batch
        if single_batch is not None and self.cfg.LOSS_WEIGHTS.SINGLE > 0:
            output_single = self.forward_step(single_batch, train=True, num_frames=1)
            pred_mano_params_single = output_single['pred_mano_params']
            loss_single = self.compute_loss_frame(single_batch, output_single, train=True)
            
            loss = loss + self.cfg.LOSS_WEIGHTS.SINGLE * loss_single
            output["losses"].update({f"loss_single/{k}": v for k,v in output_single['losses'].items()})

        losses = output['losses']
        if self.global_step % 100 == 0 or self.global_step < 10:
            print(f"Step {self.global_step}, {self.cfg.expname}")
            for k, v in losses.items():
                print(k, v)

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            if batch['img'].ndim == 5:
                batch_size = batch['img'].shape[0] * batch['img'].shape[1]
                pred_mano = pred_mano_params['hand_pose']
                pred_betas = pred_mano_params['betas']
                if single_batch is not None  and self.cfg.LOSS_WEIGHTS.SINGLE > 0:
                    pred_mano_single = pred_mano_params_single['hand_pose']
                    pred_betas_single = pred_mano_params_single['betas']
                    pred_mano = torch.cat([pred_mano, pred_mano_single], 0)
                    pred_betas = torch.cat([pred_betas, pred_betas_single], 0)
                    batch_size = pred_mano.shape[0]
            else:
                pred_mano = pred_mano_params['hand_pose']
                pred_betas = pred_mano_params['betas']
            disc_out = self.discriminator(
                pred_mano.reshape(batch_size, -1), 
                pred_betas.reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            print(batch['imgname'], batch['imgname_rel'])
            # save input, output and loss, losses
            fname = osp.join(self.logger.save_dir, f'images/nan_{self.global_step:08d}.pkl')
            with open(fname, 'wb') as f:
                import pickle
                pickle.dump({'batch': batch, 'output': output, 'loss': loss, 'losses': losses}, f)

            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            # loss_disc = self.training_step_discriminator(mocap_batch, pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1), optimizer_disc)
            loss_disc = self.training_step_discriminator(
                mocap_batch, 
                pred_mano, 
                pred_betas, optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.wandb_logging(batch, output, self.global_step, train=True)
            if single_batch is not None and self.cfg.LOSS_WEIGHTS.SINGLE > 0:
                self.wandb_logging_one_frame(single_batch, output_single, self.global_step, train=True, pref='train_single')

        # self.log_('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.logger.log_metrics({'train_loss/Total': output['losses']['loss']}, self.global_step)
        self.logger.log_metrics({f'train_loss/{k}': v for k,v in output['losses'].items() if 'loss' in k}, self.global_step)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.wandb_logging(batch, output, self.global_step, train=False)

        return output

