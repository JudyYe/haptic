import torch
from haptic.datasets.utils import convert_cvimg_to_tensor, generate_image_patch_cv2
import cv2
import os.path as osp
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from haptic.datasets.utils import expand_to_aspect_ratio


def split_to_list_dl(
    model_cfg,
    seq,
    num_frames,
    overlap=1,
    pad=True,
    down=1,
    box_mode="box_size_vary",
    load_image=True,
    load_depth=False,
    rescale_factor=2,
):
    ds = Seq2Clip(
        model_cfg,
        seq,
        num_frames,
        overlap,
        pad,
        down,
        box_mode,
        load_image=load_image,
        load_depth=load_depth,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=10, prefetch_factor=2)
    return dl


class Seq2Clip(Dataset):
    def __init__(
        self,
        cfg,
        seq,
        num_frames,
        overlap=1,
        pad=True,
        down=1,
        box_mode="box_size_vary",
        rescale_factor=2,
        load_image=True,
        load_depth=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.seq = seq
        self.index_list = split2list_index(
            len(seq["imgname"]), num_frames, overlap, pad
        )
        self.num_frames = num_frames
        self.overlap = overlap
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]

        img_file = osp.join(self.seq["img_dir"], self.seq["imgname"][0])
        inp_img = Image.open(img_file).convert("RGB")
        W, H = inp_img.size
        self.W = W
        self.H = H

        self.mean = 255.0 * np.array(IMAGE_MEAN)
        self.std = 255.0 * np.array(IMAGE_STD)
        self.down = down
        self.box_mode = box_mode
        self.rescale_factor = rescale_factor
        self.load_image = load_image
        self.load_depth = load_depth

        self.use_dino = not cfg.MODEL.get('SHARE_BACKBONE', True)
        self.keep_ratio = not self.use_dino or cfg.DATASETS.BG_CONFIG.get("KEEP_RATIO", True)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index_list = self.index_list[idx]
        batch = []
        for i in index_list:
            meta = self.load_one_frame(i, self.load_image, )
            batch.append(meta)
        batch = collate_all(batch)
        if self.box_mode == "box_size_same":
            batch["box_size"] = batch["box_size"] * 0 + batch["box_size"].mean()
            batch["box_size"] = batch["box_size"].astype(np.float32)

        return batch

    def load_one_frame(self, idx, load_image=True):
        meta = {}
        img_file = osp.join(self.seq["img_dir"], self.seq["imgname"][idx])
        W, H = self.W, self.H
        meta["box_center"] = self.seq["center"][idx].astype(np.float32)
        scale = self.seq["scale"][idx].astype(np.float32) / 2
        meta["intr"] = self.seq["focal"][idx].squeeze(0).astype(np.float32)
        meta["right"] = self.seq["is_right"][idx]
        # flip. 

        if load_image:
            inp_img = Image.open(img_file).convert("RGB")
            inp_img = np.array(inp_img)
        else:
            inp_img = np.zeros((H, W, 3), dtype=np.uint8)

        c_x, c_y = meta["box_center"]
        if self.rescale_factor == -1:
            BBOX_SHAPE = self.cfg.MODEL.get("BBOX_SHAPE", None)
            bbox_size = expand_to_aspect_ratio(
                scale * 200, target_aspect_ratio=BBOX_SHAPE
            ).max()
            bbox_expand_factor = bbox_size / ((scale * 200).max())
        else:
            bbox_expand_factor = self.rescale_factor
            bbox_size = bbox_expand_factor * scale.max() * 200

        meta["box_size"] = np.array(bbox_size).astype(np.float32)
        if self.down > 1:
            W, H = W // self.down, H // self.down
            if load_image:
                inp_img = inp_img.resize((W, H), Image.BILINEAR)
            meta["box_center"] = meta["box_center"] / self.down
            meta["box_size"] = meta["box_size"] / self.down
            meta["intr"][:2] = meta["intr"][:2] / self.down

        if load_image:
            img_patch = image_preprocess(
                inp_img, c_x, c_y, bbox_size, self.mean, self.std, is_right=meta["right"]
            )
            bg_size = 224 if self.use_dino else 256
            if self.keep_ratio:
                # default
                width = max(W, H)
                height = max(W, H)
            else:
                width = W 
                height = H

            img_orig = image_preprocess(
                inp_img, W / 2, H / 2, width, self.mean, self.std, height=height, patch_height=bg_size, patch_width=bg_size, is_right=meta["right"]
            )
            meta["orig_img"] = img_orig
            meta["img"] = img_patch
            meta["inp_img"] = ToTensor()(inp_img)

        if self.load_depth:
            depth_file = osp.join(self.seq["depth_dir"], self.seq["imgname"][idx].split(".")[0] + ".png")
            depth = Image.open(depth_file)
            depth = np.array(depth).astype(np.float32) / 255
            depth = torch.from_numpy(depth).unsqueeze(0)
            meta["depth"] = depth

        meta["cTw"] = self.seq["cTw"][idx].astype(np.float32)

        meta["hand_pose"] = self.seq["hand_pose"][idx].astype(np.float32)
        meta["hand_tsl"] = self.seq["hand_tsl"][idx].astype(np.float32)

        meta["img_file"] = str(self.seq["imgname"][idx])
        meta["img_size"] = np.array([W, H])
        meta["name"] = self.seq["imgname"][idx].split(".")[-2]

        return meta


def split2list_index(total_len, num_frames, overlap=1, pad=True):
    index_list = []
    dt = num_frames - overlap
    for i in range(0, total_len, dt):
        i1 = min(i + num_frames, total_len)
        index_list.append(list(range(i, i1)))
    if pad:
        m_frame = i + num_frames - total_len
        if m_frame > 0:
            index_list[-1] = list(range(i, total_len)) + [total_len - 1] * m_frame
    return index_list


def image_preprocess(
    cvimg,
    center_x,
    center_y,
    width,
    mean,
    std,
    height=None,
    patch_width=256,
    patch_height=256,
    border_mode=cv2.BORDER_CONSTANT,
    is_bgr=False,
    is_right=True,
):
    if height is None:
        height = width
    scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = (
        1.0,
        0,
        False,
        False,
        0,
        [1.0, 1.0, 1.0],
        0.0,
        0.0,
    )
    do_flip = not is_right
    img_patch_cv, trans = generate_image_patch_cv2(
        cvimg,
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        do_flip,
        scale,
        rot,
        border_mode=border_mode,
    )
    # apply normalization
    image = img_patch_cv.copy()
    if is_bgr:
        image = image[:, :, ::-1]
    img_patch_cv = image.copy()
    img_height, img_width, img_channels = cvimg.shape
    img_patch = convert_cvimg_to_tensor(image)

    for n_c in range(min(img_channels, 3)):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    img_patch = img_patch.astype(np.float32)
    return img_patch




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
