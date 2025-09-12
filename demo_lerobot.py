#!/usr/bin/env python3
"""
LeRobot demo script for HAPTIC hand pose estimation.

Usage:
    python demo_lerobot.py --input_dir /path/to/lerobot/data --output_dir /path/to/output --intrinsics /path/to/intrinsics.txt

Input structure expected:
    input_dir/videos/chunk-000/observation.cam_azure_kinect.color/*.mp4
"""

import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile
from collections import defaultdict
from glob import glob

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import haptic.models.haptic as haptic_module
from haptic.utils.renderer import cam_crop_to_full_w_depth, cam_crop_to_full_w_pp
from nnutils import model_utils
from nnutils.det_utils import parse_det_seq
from nnutils.hand_utils import ManopthWrapper
from haptic.datasets.seq2clip import split_to_list_dl

from demo import load_haptic_model, get_depth_by_weak2full, integrate_depth, cvt2camera_space

device = "cuda:0"


def load_intrinsics(intrinsics_path):
    """Load camera intrinsics from file."""
    intrinsics = np.loadtxt(intrinsics_path)
    return intrinsics


def project_3d_to_2d(points_3d, intrinsics):
    """Project 3D points to 2D image coordinates."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    return np.stack([u, v], axis=1)


def create_video_with_overlay(video_path, hand_poses_list, haptic_intrinsics_list, rgb_intrinsics, output_path):
    """Create video with hand pose projections overlaid."""
    frames = iio.imread(video_path)
    fps = 15

    height, width = frames.shape[1], frames.shape[2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if frame_idx < len(hand_poses_list):
            hand_pose = hand_poses_list[frame_idx]
            haptic_intr = haptic_intrinsics_list[frame_idx]
            
            # Project using haptic intrinsics
            projected = project_3d_to_2d(hand_pose, haptic_intr)

            # Filter valid projections
            valid_mask = (
                (hand_pose[:, 2] > 0.1) &
                (projected[:, 0] >= 0) & (projected[:, 0] < width) &
                (projected[:, 1] >= 0) & (projected[:, 1] < height)
            )
            
            valid_projected = projected[valid_mask]
            
            # Draw projections
            for point in valid_projected:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue circles
        
        out.write(frame)
    out.release()


@torch.no_grad()
def process_video(video_path, model, hand_wrapper, rgb_intrinsics, output_dir):
    """Process a single video file."""
    print(f"Processing: {video_path}")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy video to temp directory
        temp_video_path = osp.join(temp_dir, osp.basename(video_path))
        shutil.copy2(video_path, temp_video_path)
        
        # Create temporary config for this video
        temp_cfg = OmegaConf.create({
            'data': {
                'name': 'custom',
                'video_dir': temp_dir,
                'video_list': None,
                'video_name': osp.basename(video_path).replace('.mp4', '')
            },
            'box_mode': 'det',
            'depth_mode': 'weak2full',
            'num': -1
        })

        # Parse video sequence
        seq_list = parse_det_seq(temp_cfg.data, temp_cfg)
        
        if not seq_list:
            print(f"No sequences found for {video_path}")
            return
        
        seq = seq_list[0]  # Should be one sequence per video
        
        overlap = 1 if model.cfg.MODEL.NUM_FRAMES > 1 else 0
        rescale_factor = 2
        
        seq_dl = split_to_list_dl(
            model.cfg,
            seq,
            model.cfg.MODEL.NUM_FRAMES,
            overlap,
            box_mode=temp_cfg.box_mode,
            load_depth=False,
            rescale_factor=rescale_factor,
        )
        
        all_wHands = []
        all_scaled_wHands = []
        all_haptic_intrinsics = []
        depth0 = 0

        num_frames = iio.imread(video_path).shape[0]
        
        # Process batches
        for b, bs in enumerate(tqdm(seq_dl, desc="Processing frames")):
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
                pred, bs, hand_wrapper, temp_cfg.depth_mode
            )

            # Handle overlapping frames like demo.py
            t0 = 0 if b == 0 else overlap

            # Get wHands (world space hand vertices) - shape: (batch_size, 778, 3)
            wHands_batch = cHands_pred.verts_padded().detach().cpu().numpy()
            haptic_intr_batch = bs["intr"][0].detach().cpu().numpy()
            
            # Process each frame in the batch starting from t0 to avoid duplicates
            batch_size = len(wHands_batch)
            for batch_idx in range(t0, batch_size):
                wHands_frame = wHands_batch[batch_idx]  # (778, 3)
                haptic_intr_frame = haptic_intr_batch[batch_idx]  # (3, 3)
                
                # Apply Z-scaling to match RGB camera coordinate system
                haptic_fx = haptic_intr_frame[0, 0]
                rgb_fx = rgb_intrinsics[0, 0]
                z_scale_factor = rgb_fx / haptic_fx
                
                # Scale only Z coordinate
                scaled_wHands = wHands_frame.copy()
                scaled_wHands[:, 2] *= z_scale_factor
                
                all_wHands.append(wHands_frame)  # Original unscaled
                all_scaled_wHands.append(scaled_wHands)  # Z-scaled
                all_haptic_intrinsics.append(haptic_intr_frame)

        # Convert to numpy arrays
        all_wHands = np.array(all_wHands)[:num_frames]
        all_scaled_wHands = np.array(all_scaled_wHands)[:num_frames]
        all_haptic_intrinsics = np.array(all_haptic_intrinsics)[:num_frames]
        
        # Save numpy file
        video_name = osp.basename(video_path)
        npy_path = osp.join(output_dir, f"{video_name}.npy")
        np.save(npy_path, all_scaled_wHands)
        print(f"Saved hand poses to: {npy_path}")
        
        # Create overlay video using original video path
        overlay_path = osp.join(output_dir, f"{video_name}_overlay.mp4")
        create_video_with_overlay(
            video_path,  # Use original video path, not temp
            all_wHands,
            all_haptic_intrinsics,
            rgb_intrinsics,
            overlay_path
        )
        print(f"Saved overlay video to: {overlay_path}")
        
    # Temporary directory and its contents are automatically cleaned up here


def main():
    parser = argparse.ArgumentParser(description="LeRobot HAPTIC demo")
    parser.add_argument("--input_dir", required=True, help="Input directory containing LeRobot data")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--intrinsics", required=True, help="Path to camera intrinsics file")
    parser.add_argument("--model_path", default="output/release/mix_all/checkpoints/last.ckpt", help="Path to HAPTIC model checkpoint")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load RGB camera intrinsics
    rgb_intrinsics = load_intrinsics(args.intrinsics)
    print(f"Loaded RGB intrinsics:\n{rgb_intrinsics}")
    
    # Load HAPTIC model
    print("Loading HAPTIC model...")
    model = load_haptic_model(args.model_path, device)
    hand_wrapper = ManopthWrapper("assets/mano/").to(device)
    
    # Find video files
    video_pattern = osp.join(args.input_dir, "videos/chunk-*/observation.images.cam_azure_kinect.color/episode_*.mp4")
    video_files = sorted(glob(video_pattern))
    
    if not video_files:
        print(f"No video files found matching pattern: {video_pattern}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for video_path in video_files:
        process_video(video_path, model, hand_wrapper, rgb_intrinsics, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
