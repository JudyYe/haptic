import os
import os.path as osp
import pickle
from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from hydra import main
from tqdm import tqdm
from vitpose_model import ViTPoseModel

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

device = "cuda:0"
kpt2d_th = 0.5
det_th = 0.1 # 0.5
roi_th = 0.25 # 0.25

def load_detector():
    # Load detector
    import haptic
    from detectron2.config import LazyConfig
    from haptic.utils.utils_detectron2 import DefaultPredictor_Lazy
    cfg_path = Path(haptic.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = roi_th
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)
    return detector, cpm


def load_all_imgs(seq_dir, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    img_paths = sorted([img for end in ['*.jpg', '*.png'] for img in Path(seq_dir).glob(end)])
    if len(img_paths) == 0:
        # try decode mp4 first
        print('decoding mp4 to *.jpg')
        mp4_path = sorted(glob(os.path.join(seq_dir, '*.mp4')))

        if not len(mp4_path):
            print(f"video {mp4_path} not found")
            return [], []
        mp4_path = mp4_path[0]
        os.makedirs(seq_dir, exist_ok=True)
        # use opencv to decode video
        cap = cv2.VideoCapture(mp4_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_path = os.path.join(seq_dir, f'{frame_count:05d}.jpg')
            cv2.imwrite(img_path, frame)
            frame_count += 1            
        cap.release()
        img_paths = sorted([img for end in ['*.jpg', '*.png'] for img in Path(seq_dir).glob(end)])
        print(f"decoded {len(img_paths)} images from {mp4_path} to {seq_dir}")


    pkl_paths = []
    for img_path in img_paths:
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        pkl_file = os.path.join(out_folder, f'{img_fn}.pkl')
        pkl_paths.append(pkl_file)
    return img_paths, pkl_paths


def parse_det_seq(data_cfg, cfg, t0=0, skip=True):
    # TODO: lazy
    detector, cpm = None, None
    if data_cfg.video_list is not None:
        with open(data_cfg.video_list, 'r') as f:
            seq_list = yaml.load(f, Loader=yaml.FullLoader)
            seq_list = [osp.join(data_cfg.video_dir, seq) for seq in seq_list]
        
    else:
        seq_list = glob(data_cfg.video_dir)

    # list all seq
    parsed_data = []
    for seq_dir in tqdm(seq_list, desc='seq'):
        img_paths, pkl_paths = load_all_imgs(seq_dir, osp.join(seq_dir, 'det'))
        
        pkl_vid_file = osp.join(seq_dir, 'det.pkl')
        if skip and os.path.exists(pkl_vid_file):
            with open(pkl_vid_file, 'rb') as f:
                parsed_data.extend(pickle.load(f))
                continue
        # iterate all images and fire detection. 
        # Iterate over all images in folder
        if detector is None:
            # lazy load
            detector, cpm = load_detector()

        bboxes = defaultdict(list)
        is_right = defaultdict(list)
        all_k2d = defaultdict(list)
        valid_list = defaultdict(list)

        for img_path, pkl_file in tqdm(zip(img_paths, pkl_paths), total=len(img_paths), desc=f'img for {seq_dir}'):
            img_cv2 = cv2.imread(str(img_path))
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            
            # Detect humans in image
            det_out = detector(img_cv2)
            img = img_cv2.copy()[:, :, ::-1]

            det_instances = det_out['instances'] # this only detects people
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > det_th)
            pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores=det_instances.scores[valid_idx].cpu().numpy()

            # Detect human keypoints for each person
            # multiple hands event
            vitposes_out = cpm.predict_pose(
                img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )
            if len(vitposes_out) == 0:
                for side in ['left', 'right']:
                    bboxes[side].append([0,0,0,0])
                    is_right[side].append(0)
                    all_k2d[side].append(np.zeros((21,3)))
                    valid_list[side].append(0)

            # Use hands based on hand keypoint detections
            # TODO: jsut support one person
            for vitposes in vitposes_out[:1]:  
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:,2] > kpt2d_th
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes['left'].append(bbox)
                    is_right['left'].append(0)
                    all_k2d['left'].append(keyp)
                    valid_list['left'].append(1)
                else:
                    bboxes['left'].append([0,0,0,0])
                    is_right['left'].append(0)
                    all_k2d['left'].append(np.zeros((21,3)))
                    valid_list['left'].append(0)

                keyp = right_hand_keyp
                valid = keyp[:,2] > kpt2d_th
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes['right'].append(bbox)    
                    is_right['right'].append(1)
                    all_k2d['right'].append(keyp)
                    valid_list['right'].append(1)
                else:
                    bboxes['right'].append([0,0,0,0])
                    is_right['right'].append(1)
                    all_k2d['right'].append(np.zeros((21,3)))
                    valid_list['right'].append(0)
            
        # > 0.9 frame is valid
        all_hand_info = []
        for side in ['left', 'right']:
            valid_list[side] = np.array(valid_list[side])
            print(f"valid {valid_list[side].shape}, {valid_list[side].mean()}")
            if valid_list[side].mean() < 0.5:
                print(f"Skipping {side} {seq_dir} due to low valid frames")
                continue
        
            T = len(bboxes[side])
            # convert output to our data
            seq_info = defaultdict(list)
            seq_info['hand_pose'] = np.zeros([T, 45])  # null 
            seq_info['hand_tsl'] = np.zeros([T, 3])  # null 
            seq_info['cTw'] = np.tile(np.eye(4)[None], [T, 1, 1]) # null 
            seq_info['is_right'] = np.array(is_right[side]) # (T,)
            H, W = img_cv2.shape[:2]
            focal = (W**2 + H**2)**0.5
            intr = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])

            x1, y1, x2, y2 = np.split(np.array(bboxes[side]), [1, 2, 3], -1)
            c_x = (x1 + x2) / 2
            c_y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            scale = np.concatenate([w, h], axis=1)  # (T, 2)
            scale = scale / 100 * 1.5  # empirically the same size of other datasets

            seq_info['focal'] = np.tile(intr, [T, 1, 1])[:, None]
            seq_info['center'] = np.concatenate([c_x, c_y], axis=1)
            seq_info['scale'] = scale
                        
            seq_info['img_dir'] = osp.dirname(seq_dir)

            seq_info['imgname'] = [osp.join(osp.basename(seq_dir), osp.basename(p)) for p in img_paths]
            seq_info['valid'] = np.array(valid_list[side])
            seq_info['seq'] = osp.basename(seq_dir) + f"_{side}"
            # do infill
            infill_seq_info(seq_info)
            smooth_bbox(seq_info)
            all_hand_info.append(seq_info)

        # save to pkl
        with open(pkl_vid_file, 'wb') as f:
            pickle.dump(all_hand_info, f)
        parsed_data.extend(all_hand_info)
        
    return parsed_data


@torch.enable_grad()
def smooth_bbox(seq_info,device = 'cuda:0', T=1000, w=0.05, w_c=0.02, w_s=10):
    # SGD scale and center
    
    center = torch.FloatTensor(seq_info['center']).to(device)
    scale = torch.FloatTensor(seq_info['scale']).to(device)
    dcenter = nn.Parameter(torch.zeros_like(center))
    dscale = nn.Parameter(torch.zeros_like(scale))
    optimizer = torch.optim.AdamW([dcenter, dscale], lr=1e-3)

    for t in range(T):
        cur_center = center + dcenter
        cur_scale = scale + dscale

        # acceleration
        center_diff = cur_center[1:] - cur_center[:-1]
        scale_diff = cur_scale[1:] - cur_scale[:-1]
        center_acc = center_diff[1:] - center_diff[:-1]
        scale_acc = scale_diff[1:] - scale_diff[:-1]  # (T-2, 2)
        # print(scale_diff.shape, scale_acc, cur_scale)

        loss_acc = w_c * center_acc.norm() + w_s * scale_acc.norm()
        # loss_vel = w * (w_c * center_diff.norm() + w_s * scale_diff.norm())
        loss_reg = w * (w_c * dcenter.norm() + w_s * dscale.norm())
        loss = loss_acc + loss_reg 
        if t % 10 == 0:
            print(f"smoothing box t={t}, loss={loss.item()}, loss_acc={loss_acc.item()}, loss_reg={loss_reg.item()}, center_acc={center_acc.norm().item()}, scale_acc={scale_acc.norm().item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    seq_info['center'] = (center + dcenter).cpu().detach().numpy()
    seq_info['scale'] = (scale + dscale).cpu().detach().numpy()
        
    return seq_info



def infill_seq_info(seq_info):
    # interpolate frames
    T = len(seq_info['is_right'])
    # use NN nearest neighbor to fill 
    for key in ['center', 'scale']:
        for i in range(1, T):
            if seq_info['valid'][i]:
                continue
            seq_info[key][i] = seq_info[key][i-1]
        for i in range(T-2, -1, -1):
            if seq_info['valid'][i]:
                continue
            seq_info[key][i] = seq_info[key][i+1]
    
    # valid: non-continuous valid [0, 0, 1, 1, 0, 0, 1, 0, 1] 
    # if 0, find the nearerst left valid frame and nearest right valid frame, do linear interpolateion 
    # use two passes to find the left and value indices, then do linear interpolation
    valid = seq_info['valid']
    left = np.zeros_like(valid)
    right = np.zeros_like(valid) + T - 1
    for i in range(1, T):
        left[i] = left[i-1] if not valid[i] else i
    for i in range(T-2, -1, -1):
        right[i] = right[i+1] if not valid[i] else i    
    for key in ['center', 'scale']:
        # use two passes to find the left and value indices, then do linear interpolation
        for i in range(T):
            if valid[i]:
                continue
            l, r = left[i], right[i]
            print(f"filling {key} {i} with {l} {r}")
            seq_info[key][i] = (seq_info[key][l] * (r-i) + seq_info[key][r] * (i-l)) / (r-l)
    
    return seq_info



@main(config_path="../config", config_name="eval", version_base=None)
def test(cfg):
    vis_dir = '/is/cluster/fast/yye/vis/det'
    seq_list = parse_det_seq(cfg.data, cfg, skip=False)
    for seq in seq_list:
        vid = []
        for t in range(len(seq['is_right'])):
            img_path = osp.join(seq['img_dir'], seq['imgname'][t])
            img = imageio.imread(img_path)
            w, h = seq['scale'][t] * 200
            x, y = seq['center'][t]
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), LIGHT_BLUE, 2)
            img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
            vid.append(img)
        fname = osp.join(vis_dir, seq['seq'] + '.mp4')
        print(fname)
        os.makedirs(osp.dirname(fname), exist_ok=True)
        skvideo.io.vwrite(fname, np.asarray(vid), inputdict={'-r': '30'}, outputdict={"-pix_fmt": "yuv420p", '-r': '30'})    



if __name__ == "__main__":
    import os

    import imageio
    import skvideo.io
    test()



