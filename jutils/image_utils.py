# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from skimage import measure


def save_images(images, fname, text_list=[None], merge=1, col=8, scale=False, bg=None, mask=None, r=0.9,
                keypoint=None, color=(0, 1, 0), ext='.png'):
    """
    :param it:
    :param images: Tensor of (N, C, H, W)
    :param text_list: str * N
    :param name:
    :param scale: if RGB is in [-1, 1]
    :param keypoint: (N, K, 2) in scale of [-1, 1]
    :return:
    """
    if images.shape[1] == 4:
        if scale:
            images = images / 2 + 0.5  # [0,1]  # (N, 4, H, W)
        image = vutils.make_grid(images, col)
        image = image.cpu().detach().numpy().transpose([1, 2, 0])
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
        os.makedirs(osp.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + ext, image)
        return image
        
        image.numpy().transpose([1, 2, 0])
    if bg is not None:
        images = blend_images(images, bg, mask, r)
    if keypoint is not None:
        images = vis_j2d(images, keypoint, -1, color=color)
    
    if merge == 1:
        merge_image = tensor_text_to_canvas(images, text_list, col=col, scale=scale)

        if fname is not None:
            if not os.path.exists(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname), exist_ok=True)
            imageio.imwrite(fname + ext, merge_image)
        return merge_image
    elif merge == 0:
        if scale:
            images = images / 2 + 0.5
        images = images.cpu().detach()  # N, C, H, W
        os.makedirs(fname, exist_ok=True)
        for i, image in enumerate(images):
            image = image.numpy().transpose([1, 2, 0])
            image = np.clip(255 * image, 0, 255).astype(np.uint8)
            imageio.imwrite(osp.join(fname, '%02d%s' % (i, ext)), image)




def tensor_text_to_canvas(image, text=None, col=8, scale=False):
    """
    :param image: Tensor / numpy in shape of (N, C, H, W)
    :param text: [str, ] * N
    :param col:
    :return: uint8 numpy of (H, W, C), in scale [0, 255]
    """
    if scale:
        image = image / 2 + 0.5
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    image = image.cpu().detach()  # N, C, H, W

    image = write_text_on_image(image.numpy(), text)  # numpy (N, C, H, W) in scale [0, 1]
    image = vutils.make_grid(torch.from_numpy(image), nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)
    return image

def write_text_on_image(images, text):
    """
    :param images: (N, C, H, W) in scale [0, 1]
    :param text: (str, ) * N
    :return: (N, C, H, W) in scale [0, 1]
    """
    if text is None or text[0] is None:
        return images

    images = np.transpose(images, [0, 2, 3, 1])
    images = np.clip(255 * images, 0, 255).astype(np.uint8)

    image_list = []
    for i in range(images.shape[0]):
        img = images[i].copy()
        img = put_multi_line(img, text[i])
        image_list.append(img)
    image_list = np.array(image_list).astype(np.float32)
    image_list = image_list.transpose([0, 3, 1, 2])
    image_list = image_list / 255
    return image_list


def put_multi_line(img, multi_line, h=15):
    for i, line in enumerate(multi_line.split('\n')):
        img = cv2.putText(img, line, (h, h * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
    return img


def blend_images(fg, bg, mask=None, r=0.9):
    fg = fg.cpu()
    bg=bg.cpu()
    if mask is None:
        image = fg.cpu() * r + bg.cpu() * (1-r)
    else:
        mask = mask.cpu().float()
        image = bg * (1 - mask) + (fg * r + bg * (1 - r)) * mask
    return image




def vis_j2d(image_tensor, pts, j_list=[0, 8, 11, 14, 17, 20], color=(0, 1, 0), normed=True):
    """if normed, then 2D space is in [-1,1], else: [0, HorW]
    :param: image_tensor: tensor of (N, C, H, W)
    :param: pts: (N, V, 2) of (x, y) pairs
    :return: torch.cpu.tensor (N, C, H, W) RGB range(0, 1)
    """
    image_tensor = vis_pts(image_tensor, pts, color, normed, j_list)
    
    bones = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16], 
        [0, 17], [17, 18], [18, 19], [19, 20],
    ]

    # draw bone
    if torch.is_tensor(image_tensor):
        image_tensor = image_tensor.cpu().detach().numpy().transpose([0, 2, 3, 1])  # N, H, W, C
    if torch.is_tensor(pts):
        pts = pts.cpu().detach().numpy()
    N, H, W, _ = image_tensor.shape
    if normed:
        pts = (pts + 1) / 2 * np.array([[[W, H]]])
    image_list = []
    for n in range(N):
        image = image_tensor[n].copy()
        for bs, bt in bones:
            x1, y1 = pts[n, bs] 
            x2, y2 = pts[n, bt]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        image_list.append(image)
    image_list = np.array(image_list)
    image_list = torch.FloatTensor(image_list.transpose(0, 3, 1, 2))
    return image_list



def vis_pts(image_tensor, pts, color=(0, 1, 0), normed=True, subset=-1):
    if torch.is_tensor(image_tensor):
        image_tensor = image_tensor.cpu().detach().numpy().transpose([0, 2, 3, 1])  # N, H, W, C
    if torch.is_tensor(pts):
        pts = pts.cpu().detach().numpy()
    N, H, W, _ = image_tensor.shape
    if normed:
        pts = (pts + 1) / 2 
        pts[..., 0] *= W
        pts[..., 1] *= H
        # * np.array([[[W, H]]])
    image_list = []
    for n in range(N):
        image = image_tensor[n].copy()
        image = draw_hand(image, pts[n], subset, color)
        image_list.append(image)
    image_list = np.array(image_list)
    image_list = torch.FloatTensor(image_list.transpose(0, 3, 1, 2))
    return image_list


def draw_hand(image, pts, j_list, color):
    if j_list == -1:
        j_list = range(len(pts))
    for j in j_list:
        x, y = pts[j]
        cv2.circle(image, (int(x), int(y)), 2, color, -1)
    return image
