import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder, TransformerDecoderVid


def build_mano_head(cfg):
    mano_head_type = cfg.MODEL.MANO_HEAD.get("TYPE", "haptic")
    if mano_head_type == "transformer_decoder":
        return MANOTransformerDecoderHead(cfg)
    elif mano_head_type == 'trans_vid':
        return MANOTransformerDecoderHeadVid(cfg)
    else:
        raise ValueError("Unknown MANO head type: {}".format(mano_head_type))


class MANOTransformerDecoderHead(nn.Module):
    """Cross-attention based MANO Transformer decoder"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get("JOINT_REP", "6d")
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = (
            cfg.MODEL.MANO_HEAD.get("TRANSFORMER_INPUT", "zero") == "mean_shape"
        )
        print(f"Input to transformer is {self.input_is_mean_shape}")
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = transformer_args | dict(
            cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER
        )
        self.transformer = TransformerDecoder(**transformer_args)
        dim = transformer_args["dim"]
        self.token_dim = dim
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.MANO_HEAD.get("INIT_DECODER_XAVIER", False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(cfg.MANO.MEAN_PARAMS)
        init_hand_pose = torch.from_numpy(
            mean_params["pose"].astype(np.float32)
        ).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(
            0
        )
        init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_hand_pose", init_hand_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # TODO: Convert init_hand_pose to aa rep if needed
        if self.joint_rep_type == "aa":
            raise NotImplementedError

        pred_hand_pose = init_hand_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_hand_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.cfg.MODEL.MANO_HEAD.get("IEF_ITERS", 1)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_hand_pose, pred_betas, pred_cam], dim=1)[
                    :, None, :
                ]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C)

            # Readout from token_out
            pred_hand_pose = self.decpose(token_out) + pred_hand_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_hand_pose_list.append(pred_hand_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            "6d": rot6d_to_rotmat,
            "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]

        pred_mano_params_list = {}
        pred_mano_params_list["hand_pose"] = torch.cat(
            [
                joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :]
                for pbp in pred_hand_pose_list
            ],
            dim=0,
        )
        pred_mano_params_list["betas"] = torch.cat(pred_betas_list, dim=0)
        pred_mano_params_list["cam"] = torch.cat(pred_cam_list, dim=0)
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(
            batch_size, self.cfg.MANO.NUM_HAND_JOINTS + 1, 3, 3
        )

        pred_mano_params = {
            "global_orient": pred_hand_pose[:, [0]],
            "hand_pose": pred_hand_pose[:, 1:],
            "betas": pred_betas,
        }
        return pred_mano_params, pred_cam, pred_mano_params_list


class MANOTransformerDecoderHeadVid(MANOTransformerDecoderHead):
    def __init__(self, cfg):
        super().__init__(cfg)
        aux_dim = cfg.MODEL.MANO_HEAD.get("AUX_DIM", 0)
        self.decaux = nn.Linear(self.token_dim, aux_dim)
        if cfg.MODEL.MANO_HEAD.get("INIT_DECODER_XAVIER", False):
            nn.init.xavier_uniform_(self.decaux.weight, gain=0.01)
        init_aux = torch.zeros(1, aux_dim)
        self.register_buffer("init_aux", init_aux)

        transformer_args = dict(
            num_tokens=1, # cfg.MODEL.NUM_FRAMES,
            token_dim=(self.npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
            cfg=cfg.MODEL,
        )
        transformer_args = transformer_args | dict(
            cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER
        )
        if not cfg.MODEL.get('SHARE_BACKBONE', True):
            transformer_args["global_context_dim"] = 384
        self.transformer = TransformerDecoderVid(**transformer_args)

    def forward(self, x, global_context, num_frames=1, **kwargs):
        """

        :param x: (BT, C, H, W)
        :param global_context: (BT, C, H, W)
        :raises NotImplementedError: _description_
        :return: _description_
        """
        T = num_frames
        batch_size = x.shape[0]
        B = batch_size // T
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        global_context = einops.rearrange(global_context, "b t c h w -> (b t) (h w) c")

        # global_context = self.pos_enc_frame(global_context)
        # (B, T, H*W, C)

        init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        init_aux = self.init_aux.expand(batch_size, -1)

        # TODO: Convert init_hand_pose to aa rep if needed
        if self.joint_rep_type == "aa":
            raise NotImplementedError

        pred_hand_pose = init_hand_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_aux = init_aux

        pred_hand_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        pred_aux_list = []
        for i in range(self.cfg.MODEL.MANO_HEAD.get("IEF_ITERS", 1)):
            # Input token to transformer is zero token
            token = torch.zeros(B*T, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x, temporal_context=global_context, num_frames=num_frames)
            # (B, T, C)
            token_out = token_out.view(batch_size, -1)

            # Readout from token_out
            pred_hand_pose = self.decpose(token_out) + pred_hand_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_aux = self.decaux(token_out) + pred_aux

            pred_hand_pose_list.append(pred_hand_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)
            pred_aux_list.append(pred_aux)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            "6d": rot6d_to_rotmat,
            "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]

        pred_mano_params_list = {}
        pred_mano_params_list["hand_pose"] = torch.cat(
            [
                joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :]
                for pbp in pred_hand_pose_list
            ],
            dim=0,
        )
        pred_mano_params_list["betas"] = torch.cat(pred_betas_list, dim=0)
        pred_mano_params_list["cam"] = torch.cat(pred_cam_list, dim=0)
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(
            batch_size, self.cfg.MANO.NUM_HAND_JOINTS + 1, 3, 3
        )

        pred_mano_params = {
            "global_orient": pred_hand_pose[:, [0]],
            "hand_pose": pred_hand_pose[:, 1:],
            "betas": pred_betas,
        }
        return pred_mano_params, pred_cam, pred_aux, pred_mano_params_list

