import copy
import os
import os.path as osp
import random
from typing import Any, Dict, List

import braceexpand
import numpy as np
import torch
from PIL import Image
from yacs.config import CfgNode

from .dataset import Dataset
from .utils import expand_to_aspect_ratio, get_aug_list, get_example


def expand(s):
    return os.path.expanduser(os.path.expandvars(s))


def expand_urls(urls: str | List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls


FLIP_KEYPOINT_PERMUTATION = list(np.arange(21).astype(np.int32))

DEFAULT_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255.0 * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256


class ClipDataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        dataset_file: str,
        label_dir: str,
        img_dir: str,
        train: bool = True,
        rescale_factor=2,
        prune: Dict[str, Any] = {},
        **kwargs,
    ):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super().__init__()
        self.train = train
        self.cfg = cfg

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(self.cfg.MODEL.IMAGE_STD)
        self.rescale_factor = rescale_factor

        self.use_dino = not cfg.MODEL.get("SHARE_BACKBONE", True)
        self.keep_ratio = not self.use_dino or cfg.DATASETS.BG_CONFIG.get(
            "KEEP_RATIO", True
        )
        print(f"Keep ratio: {self.keep_ratio} use dino: {self.use_dino}")

        self.img_dir = img_dir

        clip_list = np.load(dataset_file, allow_pickle=True)
        self.label_list = [osp.join(label_dir, f"{e}") for e in clip_list["labelname"]]
        self.total_frames = cfg.MODEL.NUM_FRAMES
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

    def cvt_label_to_data(self, data):
        rtn = {}

        valid = np.ones(len(data["center"]), dtype=bool)
        W, H = Image.open(os.path.join(self.img_dir, data["imgname"][0])).size
        for c, center in enumerate(data["center"]):
            if center[0] < 0 or center[1] < 0:
                valid[c] = False
            if center[0] > W or center[1] > H:
                valid[c] = False
            if data["scale"][c][0] < 1e-2 or data["scale"][c][1] < 1e-2:
                valid[c] = False
        rtn["valid"] = valid

        rtn["imgname"] = data["imgname"]
        T = len(rtn["imgname"])

        num_pose = 3 * (self.cfg.MANO.NUM_HAND_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        rtn["center"] = data["center"]
        rtn["scale"] = (
            np.array(data["scale"]).reshape(len(rtn["center"]), -1) / 2
        )  # / 200.0
        if rtn["scale"].shape[1] == 1:
            rtn["scale"] = np.tile(rtn["scale"], (1, 2))
        assert rtn["scale"].shape == (len(rtn["center"]), 2)

        try:
            rtn["right"] = data["right"]
        except KeyError:
            rtn["right"] = np.ones(len(rtn["imgname"]), dtype=np.float32)

        # Get gt MANO parameters, if available
        try:
            rtn["hand_pose"] = data["hand_pose"].astype(np.float32)
            rtn["has_hand_pose"] = data["has_hand_pose"].astype(np.float32)
        except KeyError:
            rtn["hand_pose"] = np.zeros((T, num_pose), dtype=np.float32)
            rtn["has_hand_pose"] = np.zeros(T, dtype=np.float32)
        try:
            rtn["betas"] = data["betas"].astype(np.float32)
            rtn["has_betas"] = data["has_betas"].astype(np.float32)
        except KeyError:
            rtn["betas"] = np.zeros((T, 10), dtype=np.float32)
            rtn["has_betas"] = np.zeros(T, dtype=np.float32)

        # Try to get 2d keypoints, if available
        try:
            hand_keypoints_2d = data["hand_keypoints_2d"]
        except KeyError:
            hand_keypoints_2d = np.zeros((T, 21, 3))

        # hand_keypoints_2d: (..., J,  3), if [..., -1] is 0, then set the row to 0
        inds = np.where(hand_keypoints_2d[..., -1] == 0)
        if len(inds[0]) > 0:
            hand_keypoints_2d[inds] = 0
        # find where hand_keypoints_2d (T, J, 3) is nan, set the row to 0 [t, j, :] = 0
        inds = np.where(np.isnan(hand_keypoints_2d))
        if len(inds[0]) > 0:
            hand_keypoints_2d[inds[0], inds[1], :] = 0

        rtn["keypoints_2d"] = hand_keypoints_2d

        try:
            hand_keypoints_3d = data["hand_keypoints_3d"].astype(np.float32)
        except KeyError:
            hand_keypoints_3d = np.zeros((T, 21, 4), dtype=np.float32)

        rtn["keypoints_3d"] = hand_keypoints_3d
        rtn["cTw"] = data["cTw"]
        rtn["focal"] = data["focal"]
        return rtn

    def __len__(self) -> int:
        return len(self.label_list)

    def skip_sample(self, idx):
        print(f"Skip sample {idx}, {self.label_list[idx]}")
        if self.train:
            return self.__getitem__(np.random.randint(0, len(self)))
        else:
            return self.__getitem__((idx + 1) % len(self))

    def get_frame_ids(self, label_list):
        S = len(label_list["imgname"])
        T = self.total_frames
        if self.train:
            dt = 1
            if (
                self.cfg.DATASETS.CONFIG.AUG_DT > 1
                and np.random.rand() < self.cfg.DATASETS.CONFIG.AUG_DT_RATE
            ):
                dt = random.randint(1, self.cfg.DATASETS.CONFIG.AUG_DT)  # 1, 2, ..., 5
            T = self.total_frames
            t0 = random.randint(0, S - T * dt)
        else:
            t0 = 0
            dt = 1

        prev_ids = np.arange(t0, t0 + T * dt, dt).astype(np.int32).tolist()

        return prev_ids

    def __getitem__(self, idx: int) -> Dict:
        # load label
        label_list = np.load(self.label_list[idx], allow_pickle=True)
        data = self.cvt_label_to_data(label_list)

        try:
            frame_ids = self.get_frame_ids(label_list)
        except ValueError:
            return self.skip_sample(idx)

        ids = frame_ids
        valid_list = data["valid"][ids]
        if not valid_list.all():
            return self.skip_sample(idx)
        items = []

        aug_list_fg = get_aug_list(self.cfg.DATASETS.CONFIG, len(ids))
        aug_list_bg = get_aug_list(
            self.cfg.DATASETS.BG_CONFIG, len(ids), self.cfg.DATASETS.BG_CONFIG.AUG_RATE
        )

        for i, aug, aug_bg in zip(ids, aug_list_fg, aug_list_bg):
            item = self._load_one(data, i, aug, aug_bg)
            items.append(item)
        # stack all
        item_all = collate_all(items)

        return item_all

    def _load_one(self, data, idx: int, aug=None, aug_bg=None) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file_rel = data["imgname"][idx].decode("utf-8")
        except AttributeError:
            image_file_rel = data["imgname"][idx]
        image_file = os.path.join(self.img_dir, image_file_rel)

        keypoints_2d = data["keypoints_2d"][idx].copy()
        keypoints_3d = data["keypoints_3d"][idx].copy()

        center = data["center"][idx].copy()
        center_x = center[0]
        center_y = center[1]
        scale = data["scale"][idx]
        right = data["right"][idx].copy()
        if self.rescale_factor == -1:
            BBOX_SHAPE = self.cfg.MODEL.get("BBOX_SHAPE", None)
            bbox_size = expand_to_aspect_ratio(
                scale * 200, target_aspect_ratio=BBOX_SHAPE
            ).max()
            bbox_expand_factor = bbox_size / ((scale * 200).max())
        else:
            bbox_expand_factor = self.rescale_factor
            bbox_size = bbox_expand_factor * scale.max() * 200
        hand_pose = data["hand_pose"][idx].copy().astype(np.float32)
        betas = data["betas"][idx].copy().astype(np.float32)

        has_hand_pose = data["has_hand_pose"][idx].copy()
        has_betas = data["has_betas"][idx].copy()

        mano_params = {
            "global_orient": hand_pose[:3],
            "hand_pose": hand_pose[3:],
            "betas": betas,
        }
        has_mano_params = {
            "global_orient": bool(has_hand_pose),
            "hand_pose": bool(has_hand_pose),
            "betas": bool(has_betas),
        }

        mano_params_is_axis_angle = {
            "global_orient": True,
            "hand_pose": True,
            "betas": False,
        }

        # augm_config = self.cfg.DATASETS.CONFIG
        augm_config = None
        # Crop image and (possibly) perform data augmentation
        (
            img_patch,
            keypoints_2d,
            keypoints_3d,
            mano_params,
            has_mano_params,
            img_size,
            trans,
            orig_img,
        ) = get_example(
            image_file,
            center_x,
            center_y,
            bbox_size,
            bbox_size,
            keypoints_2d,
            keypoints_3d,
            mano_params,
            has_mano_params,
            self.flip_keypoint_permutation,
            self.img_size,
            self.img_size,
            self.mean,
            self.std,
            self.train,
            right,
            augm_config,
            return_trans=True,
            return_orig_img=True,
            aug=aug,
        )

        H, W = img_size
        WH = max(H, W)
        if self.use_dino:
            bg_size = 224
        else:
            bg_size = self.img_size
        if self.keep_ratio:
            # default
            width = WH
            height = WH
        else:
            width = W
            height = H

        orig_img = get_example(
            image_file,
            W / 2,
            H / 2,
            width,
            height,
            keypoints_2d.copy(),
            keypoints_3d.copy(),
            mano_params.copy(),
            has_mano_params.copy(),
            self.flip_keypoint_permutation,
            bg_size,
            bg_size,
            self.mean,
            self.std,
            self.train,
            right,
            None,
            return_trans=True,
            return_orig_img=True,
            aug=aug_bg,
        )[0]

        item = {}
        orig_keypoints_2d = data["keypoints_2d"][idx].copy()

        item["img"] = img_patch
        item["orig_img"] = orig_img
        item["keypoints_2d"] = keypoints_2d.astype(np.float32)
        item["keypoints_3d"] = keypoints_3d.astype(np.float32)
        item["orig_keypoints_2d"] = orig_keypoints_2d
        item["box_center"] = data["center"][idx].copy()
        item["box_size"] = bbox_size
        item["bbox_expand_factor"] = bbox_expand_factor
        item["img_size"] = 1.0 * img_size[::-1].copy()
        item["mano_params"] = mano_params
        item["has_mano_params"] = has_mano_params
        item["mano_params_is_axis_angle"] = mano_params_is_axis_angle
        item["imgname"] = image_file
        item["imgname_rel"] = image_file_rel
        item["idx"] = idx
        item["_scale"] = scale
        item["right"] = bool(data["right"][idx].copy())

        return item


def pad_mask_seq(T, seqs_to_mask):
    mask = np.ones((T,), dtype=bool)
    mask[T - len(seqs_to_mask[0]) :] = False

    for i, seq in enumerate(seqs_to_mask):
        seq = np.concatenate([seq, np.zeros((T - len(seq), *seq.shape[1:]))], axis=0)
        seqs_to_mask[i] = seq
    return mask, seqs_to_mask


def collate_all(item_list):
    item_all = {}
    for key, v in item_list[0].items():
        if isinstance(v, np.ndarray):
            item_all[key] = np.stack([item[key] for item in item_list])
        elif torch.is_tensor(v):
            item_all[key] = torch.stack([item[key] for item in item_list])
        elif isinstance(v, float):
            item_all[key] = np.array([item[key] for item in item_list])
        elif isinstance(v, int):
            item_all[key] = np.array([item[key] for item in item_list])
        elif isinstance(v, bool):
            item_all[key] = np.array([item[key] for item in item_list])
        elif isinstance(v, str):
            item_all[key] = [item[key] for item in item_list]
        elif isinstance(v, dict):
            item_all[key] = collate_all([item[key] for item in item_list])
        else:
            item_all[key] = [item[key] for item in item_list]

    return item_all
