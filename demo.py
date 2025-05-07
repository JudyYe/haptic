
import imageio
import json
import os
import os.path as osp
import pickle
from collections import defaultdict
from glob import glob

import torch
from haptic.datasets.seq2clip import split_to_list_dl
from hydra import main
from omegaconf import OmegaConf
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes
from tqdm import tqdm

import haptic.models.haptic as haptic_module
from haptic.utils.renderer import (
    cam_crop_to_full_w_depth,
    cam_crop_to_full_w_pp,
)
from nnutils import geom_utils, mesh_utils, model_utils
from nnutils.hand_utils import ManopthWrapper
from nnutils.visualizer import (
    draw_world,
    Visualizer,
    vis_in_cam,
    vis_in_world,
    vis_quad,
)

device = "cuda:0"


def load_haptic_model(ckpt_path, device, load_weight=True):
    cfg_path = ckpt_path.split("/checkpoints")[0] + "/config.yaml"
    cfg = OmegaConf.load(cfg_path)

    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    class_ = getattr(haptic_module, cfg.MODEL.get("TARGET", "HAPTIC"))
    model = class_(cfg)
    if load_weight:
        model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model


@main(config_path="haptic/configs_hydra", config_name="demo", version_base=None)
def demo(cfg):
    if not cfg.dry:
        infer_seq(cfg)
    print('visualizing...')
    vis_seq(cfg)


@torch.no_grad()
def infer_seq(cfg):
    hand_wrapper = ManopthWrapper().to(device)
    model = load_haptic_model(cfg.ckpt, device)
    seq_list, _ = get_seq_list(cfg)

    overlap = 1 if model.cfg.MODEL.NUM_FRAMES > 1 else 0

    depth0 = 0
    rescale_factor = 2
    for s, seq in enumerate(tqdm(seq_list, desc="seq")):
        seq_dl = split_to_list_dl(
            model.cfg,
            seq,
            model.cfg.MODEL.NUM_FRAMES,
            overlap,
            box_mode=cfg.box_mode,
            load_depth=cfg.depth_mode == "zoe",
            rescale_factor=rescale_factor,
        )
        if cfg.num > 0 and s >= cfg.num:
            break
        for b, bs in enumerate(tqdm(seq_dl, desc="within seq")):
            bs = model_utils.to_cuda(bs, device)
            pred = model(bs)

            if b == 0:
                depth0 = get_depth_by_weak2full(
                    pred["pred_cam"][0:1],
                    bs["intr"][0, 0:1],
                    bs["img_size"][0, 0:1],
                    bs["box_center"][0, 0:1],
                    bs["box_size"][0, 0:1],
                )

            depth0, pred["pred_depth"] = integrate_depth(depth0, pred)

            cHands_pred, cJoints_pred, cam_full = cvt2camera_space(
                pred, bs, hand_wrapper, cfg.depth_mode
            )
            # cTw (world2camera) is identity, it could also come from camera estimation (MegaSAM) / on-device SLAM
            cTw = bs["cTw"][0]
            wTc = geom_utils.inverse_rt_v2(cTw)
            wHands_pred = mesh_utils.apply_transform(cHands_pred, wTc)
            wJoints_pred = mesh_utils.apply_transform(cJoints_pred, wTc)

            t0 = 0 if b == 0 else overlap

            # save
            to_save = {
                "right": bs["right"],
                "intr": bs["intr"][0],
                "img_file": [e[0] for e in bs["img_file"]],
                "img_size": bs["img_size"][0],
                "wHands": wHands_pred.verts_padded(),
                "wJoints": wJoints_pred,
                "cHands": cHands_pred.verts_padded(),
                "cJoints": cJoints_pred,
                "cam_full": cam_full,
                "hand_pose": pred["pred_mano_params"]["hand_pose"],
                "global_orient": pred["pred_mano_params"]["global_orient"],
                "betas": pred["pred_mano_params"]["betas"],
            }
            save_as_frames(
                to_save,
                [osp.basename(e[0]) for e in bs["name"]],
                t0,
                osp.join(cfg.exp_dir, cfg.eval_folder, seq["seq"]),
            )

            if cfg.save_crop:
                cHands_crop = get_cHands_crop(pred, hand_wrapper)
                cameras_crop = get_camera_crop(pred, bs)

                H, W = bs["img"].shape[-2:]
                crop_save = {
                    "cHands_crop": cHands_crop.verts_padded(),
                    "focal_crop": cameras_crop.focal_length,
                    "img_crop": bs["img"].reshape(-1, 3, H, W),
                    "right": torch.cat(bs["right"], 0).reshape(-1),
                }
                save_as_frames(
                    crop_save,
                    [osp.basename(e[0]) for e in bs["name"]],
                    t0,
                    osp.join(cfg.exp_dir, cfg.eval_folder + "_crop", seq["seq"]),
                )


@torch.no_grad()
def vis_seq(cfg):
    wrapper = ManopthWrapper().to(device)
    print("todo")

    # wrapper = ManopthWrapper(cfg.mano_dir).to(device)
    vis = Visualizer()
    seq_list, _ = get_seq_list(cfg)
    for s, seq in enumerate(tqdm(seq_list, desc="seq")):
        if cfg.num > 0 and s >= cfg.num:
            break

        vis_dir = osp.join(
            cfg.exp_dir, cfg.eval_folder + "_vis", f"{seq['seq']}", "{0}"
        )

        query_pattern = osp.join(seq["seq"], "*.pkl")
        pred_dir = osp.join(cfg.exp_dir, cfg.eval_folder)

        pred_list = sorted(glob(osp.join(pred_dir, query_pattern)))
        print(osp.join(pred_dir, query_pattern))

        if cfg.eval_T < 0:
            cfg.eval_T = 100000000
        img_size_fix = None

        jts_pred_list = []
        hand_pred_list = []
        intr_list = []
        img_size_list = []
        img_file_list = []

        crop_list = defaultdict(list)
        for pp in range(0, len(pred_list)):
            p = pred_list[pp]

            with open(p, "rb") as f:
                pred = pickle.load(f)
            right = pred.get("right", [True])[0]
            flip = not right

            jts_pred_list.append(pred["cJoints"])
            hand_pred_list.append(pred["cHands"])

            intr_list.append(pred["intr"].cpu())

            # for homogenous on wham
            if pred["img_size"].shape[0] == 0:
                pred["img_size"] = img_size_fix
            elif pred["img_size"].ndim >= 2 and pred["img_size"].shape[1] > 2:
                pred["img_size"] = img_size_fix = pred["img_size"][:, :2]

            pred["img_size"] = pred["img_size"].reshape(1, 2).cpu()
            img_size_list.append(pred["img_size"])
            img_file_list.append(osp.join(seq["img_dir"], pred["img_file"][0]))

            if cfg.fig_crop:
                crop_file = p.replace(cfg.eval_folder, cfg.eval_folder + "_crop")
                with open(crop_file, "rb") as f:
                    crop = pickle.load(f)
                crop_list["cHands_crop"].append(crop["cHands_crop"])
                crop_list["focal_crop"].append(crop["focal_crop"])
                crop_list["img_crop"].append(crop["img_crop"])
                crop_list["right"].append(crop["right"])

        jts_pred = torch.cat(jts_pred_list, 0)
        hand_pred = torch.cat(hand_pred_list, 0)
        intr = torch.cat(intr_list, 0)
        img_size = torch.cat(img_size_list, 0)
        for k, v in crop_list.items():
            crop_list[k] = torch.cat(v, 0)
        if cfg.fig_crop:
            tt = len(crop_list["cHands_crop"])
            cHands_crop = Meshes(
                crop_list["cHands_crop"].to(device),
                faces=wrapper.hand_faces.repeat(tt, 1, 1),
            ).to(device)
            cHands_crop.textures = mesh_utils.pad_texture(cHands_crop, "blue")
            img_crop = crop_list["img_crop"]
            cameras = PerspectiveCameras(focal_length=crop_list["focal_crop"]).to(
                device
            )
            vis_quad(
                cHands_crop,
                img_crop,
                cameras,
                vis_dir.format("crop"),
                crop_list["right"],
            )

        max_img_size = img_size.max(0)[0]
        down = (max_img_size.max() / 512).ceil().int().item()
        if down > 1:
            img_size = img_size // down
            intr[..., :2, :] /= down

        W, H = img_size.split([1, 1], -1)
        W, H = W.squeeze(-1), H.squeeze(-1)
        cameras = Visualizer.get_cameras(intr, H, W)

        h_pred = hand_pred - jts_pred[0:1, 0:1]
        h_jts_pred = jts_pred - jts_pred[0:1, 0:1]

        video_np = draw_world(pred_list, wrapper)
        save_file = osp.join(vis_dir.format('trail') + '.gif')
        imageio.mimwrite(save_file, video_np, fps=10, loop=0)

        vis_in_world(
            None,
            h_pred,
            wrapper,
            vis_dir.format("side1"),
            cfg=cfg,
            root=h_jts_pred[..., 0:1, :],
            az=60,
            el=0,
            flip=flip,
        )
        vis_in_cam(
            None,
            hand_pred,
            cameras,
            img_size,
            img_file_list,
            wrapper,
            vis_dir.format('cam'),
            cfg=cfg,
            flip=flip,
            root=jts_pred[..., 0:1, :],
        )


def get_seq_list(cfg):
    print('TODO refractor ')
    if cfg.data.name.startswith("arc"):
        from dataset.arctic import parse_arctic_long_seq

        data_file = osp.join(
            cfg.data.data_dir, "haptic_format", f"v{cfg.data.vid}_val.data.pyd"
        )
        seq_list = parse_arctic_long_seq(
            cfg.data,
            data_file,
        )
        gt_dir = "/is/cluster/fast/yye/data/arctic/gt_results"
    if cfg.data.name == "dexycb":
        from dataset.arctic import parse_arctic_long_seq

        data_file = osp.join("/is/cluster/fast/yye/data/dexycb", "s0_val.data.pyd")
        seq_list = parse_arctic_long_seq(
            cfg.data,
            data_file,
        )
        gt_dir = "/is/cluster/fast/yye/data/dexycb/gt_results"
    elif cfg.data.name == "custom":
        from nnutils.det_utils import parse_det_seq

        seq_list = parse_det_seq(cfg.data, cfg)
        gt_dir = ""

    return seq_list, gt_dir


def save_as_frames(data, name_list, t0, save_dir):
    data = model_utils.to_cuda(data, "cpu")
    for t in range(t0, len(name_list)):
        d = {k: v[t : t + 1] for k, v in data.items()}
        save_file = osp.join(save_dir, f"{name_list[t]}.pkl")
        os.makedirs(osp.dirname(save_file), exist_ok=True)
        with open(save_file, "wb") as f:
            pickle.dump(d, f)


def get_depth_by_weak2full(pred_cam, intr, img_size, box_center, box_size):
    """
    :param pred_cam: (B, 3)
    :param intr: (N, 4, 4? )
    :param img_size: (N, 2) in WH!! (Not HW!!!)
    :param box_center: _description_
    :param box_size: _description_
    :returns: (N, )
    """
    W, H = img_size[..., 0], img_size[..., 1]
    cam_full = cam_crop_to_full_w_pp(pred_cam, intr, H, W, box_center, box_size)
    return cam_full[..., 2]  # z


def integrate_depth(depth0, pred):
    offset = pred["pred_depth"][0:1]
    pred["pred_depth"] = pred["pred_depth"] - offset + depth0
    depth0 = pred["pred_depth"][-1:]  # (BT=1, )
    return depth0, pred["pred_depth"]


def cvt2camera_space(result_list, bs, wrapper, depth_mode="pred"):
    pHand_pred, pJoint_pred = wrapper(
        None,
        geom_utils.matrix_to_axis_angle(
            result_list["pred_mano_params"]["hand_pose"]
        ).reshape(-1, 45),
        geom_utils.matrix_to_axis_angle(
            result_list["pred_mano_params"]["global_orient"]
        ).reshape(-1, 3),
        th_betas=result_list["pred_mano_params"]["betas"],
    )
    W, H = bs["img_size"][0].split([1, 1], -1)
    W, H = W.squeeze(-1), H.squeeze(-1)

    box_center = bs["box_center"][0].clone()
    flip = not bs["right"][0]
    if flip:
        box_center[..., 0] = W - box_center[..., 0]
    if depth_mode == "pred":
        cam_full = cam_crop_to_full_w_depth(
            result_list["pred_cam"],
            bs["intr"][0],
            H,
            W,
            box_center,
            bs["box_size"][0],
            result_list["pred_depth"].squeeze(1),
        )

    elif depth_mode == "weak2full":
        cam_full = cam_crop_to_full_w_pp(
            result_list["pred_cam"],
            bs["intr"][0],
            H,
            W,
            box_center,
            bs["box_size"][0],
        )
    else:
        raise NotImplementedError(f"depth_mode {depth_mode} not implemented")

    cTp = geom_utils.axis_angle_t_to_matrix(torch.zeros_like(cam_full), cam_full)
    cHand_pred = mesh_utils.apply_transform(pHand_pred, cTp)
    cJoint_pred = mesh_utils.apply_transform(pJoint_pred, cTp)

    if flip:
        verts = cHand_pred.verts_padded()
        verts[..., 0] = -verts[..., 0]
        faces = cHand_pred.faces_padded()
        faces = torch.flip(faces, [-1])
        cHand_pred = Meshes(verts, faces)
        cJoint_pred[..., 0] = -cJoint_pred[..., 0]
    return cHand_pred, cJoint_pred, cam_full


def get_camera_crop(out, batch):
    size = batch["img"].shape[-1]
    focal_length = out["focal_length"]
    focal_length = focal_length / size * 2
    cameras = PerspectiveCameras(focal_length, device=device)
    return cameras


def get_cHands_crop(out, hand_wrapper):
    tsl = out["pred_cam_t"]
    cTp = geom_utils.axis_angle_t_to_matrix(torch.zeros_like(tsl), tsl)
    pHands = Meshes(
        out["pred_vertices"], hand_wrapper.hand_faces.repeat(len(tsl), 1, 1)
    ).to(device)
    pHands.textures = mesh_utils.pad_texture(pHands, "blue")
    cHands = mesh_utils.apply_transform(pHands, cTp)
    return cHands


if __name__ == "__main__":
    demo()
