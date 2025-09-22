#!/usr/bin/env python3
import numpy as np
import cv2
import rerun as rr
import av
from pathlib import Path
import argparse
import pickle
import glob
from tqdm import tqdm
import imageio.v3 as iio


def read_depth_video(video_path):
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    loaded_frames = []
    for frame in container.decode(video_stream):
        if frame.format.name in ["gray16le", "gray16be"]:
            frame_array = frame.to_ndarray(format="gray16le") / 1000.0
        else:
            raise NotImplementedError("Not supporting other formats right now.")
        loaded_frames.append(frame_array)
    container.close()

    frames = np.stack([frame for frame in loaded_frames])
    return frames

def load_intrinsics(intrinsics_path):
    intrinsics = np.loadtxt(intrinsics_path)
    return intrinsics


def project_3d_to_2d(points_3d, intrinsics):
    """
    Project 3D points to 2D image coordinates using camera intrinsics.
    
    Args:
        points_3d: (N, 3) array of 3D points
        intrinsics: (3, 3) camera intrinsic matrix
    
    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Extract x, y, z coordinates
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    # Project to 2D
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    return np.stack([u, v], axis=1)


def depth_to_pointcloud(depth, intrinsics):
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    valid_mask = (depth > 0.1) & (depth < 5.0)
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask]
    
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    points = np.stack([x, y, z], axis=1)
    return points, valid_mask


def load_lerobot_videos(dset_name, video_index):
    """Load RGB and depth videos from lerobot dataset."""
    cache_dir = Path.home() / ".cache/huggingface/lerobot" / dset_name
    
    # Find RGB video
    rgb_pattern = str(cache_dir / f"videos/chunk*/observation.images.cam_azure_kinect.color/episode_{video_index:06d}.mp4")
    rgb_files = glob.glob(rgb_pattern)
    if not rgb_files:
        raise FileNotFoundError(f"No RGB video found for pattern: {rgb_pattern}")
    rgb_path = rgb_files[0]
    
    # Find depth video  
    depth_pattern = str(cache_dir / f"videos/chunk*/observation.images.cam_azure_kinect.transformed_depth/episode_{video_index:06d}.mkv")
    depth_files = glob.glob(depth_pattern)
    if not depth_files:
        raise FileNotFoundError(f"No depth video found for pattern: {depth_pattern}")
    depth_path = depth_files[0]
    
    return rgb_path, depth_path


def load_haptic_hand_poses(haptic_dir, dset_name, video_index):
    """Load haptic hand poses for specific dataset and video index."""
    haptic_dir = Path(haptic_dir)
    npy_path = haptic_dir / dset_name / "haptic_output" / f"episode_{video_index:06d}.mp4.npy"
    
    if not npy_path.exists():
        raise FileNotFoundError(f"Haptic hand pose file not found: {npy_path}")
    
    # Load the scaled hand poses (scaling already done in demo_lerobot.py)
    hand_poses = np.load(npy_path)
    return hand_poses


def visualize_lerobot_sequence(dset_name, video_index, intrinsics_path, haptic_pose_dir):
    rr.init("LeRobot_Visualization", spawn=True)

    # Load camera intrinsics
    intrinsics = load_intrinsics(intrinsics_path)
    
    # Load RGB and depth videos from lerobot dataset
    rgb_video_path, depth_video_path = load_lerobot_videos(dset_name, video_index)
    rgb_frames = iio.imread(rgb_video_path)
    depth_frames = read_depth_video(depth_video_path)

    # Load haptic hand poses (scaling already done in demo_lerobot.py)
    haptic_poses = load_haptic_hand_poses(haptic_pose_dir, dset_name, video_index)

    # Ensure all data have the same number of frames
    assert len(rgb_frames) == len(depth_frames), f"RGB frames: {len(rgb_frames)}, Depth frames: {len(depth_frames)}"
    assert len(rgb_frames) == len(haptic_poses), f"RGB frames: {len(rgb_frames)}, Haptic poses: {len(haptic_poses)}"

    # Set up camera pinhole model for visualization
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = rgb_frames[0].shape[:2]
    
    # Process each frame
    for frame_idx in tqdm(range(len(rgb_frames))):
        rr.set_time_sequence("frame", frame_idx)
        
        rgb_frame = rgb_frames[frame_idx]
        depth_frame = depth_frames[frame_idx]
        
        # Create and log 3D point cloud
        points_3d, valid_mask = depth_to_pointcloud(depth_frame, intrinsics)
        # Get colors for all valid points
        v_indices, u_indices = np.where(valid_mask)
        colors = rgb_frame[v_indices, u_indices] / 255.0
        # Log 3D point cloud
        rr.log("world/pointcloud", rr.Points3D(points_3d, colors=colors))
        
        haptic_pose_frame = haptic_poses[frame_idx]

        # Log haptic hand poses in blue (no scaling needed - already done in demo_lerobot.py)
        rr.log("world/hand/haptic", rr.Points3D(haptic_pose_frame, colors=[0.0, 0.0, 1.0]))

        # Project haptic hand poses to 2D using RGB camera intrinsics
        haptic_projected = project_3d_to_2d(haptic_pose_frame, intrinsics)
        
        # Filter points that are within image bounds and have positive depth
        haptic_valid_mask = (
            (haptic_pose_frame[:, 2] > 0.1) &  # Positive depth
            (haptic_projected[:, 0] >= 0) & (haptic_projected[:, 0] < w) &  # Within width
            (haptic_projected[:, 1] >= 0) & (haptic_projected[:, 1] < h)    # Within height
        )
        haptic_valid_projected = haptic_projected[haptic_valid_mask]
            
        # Create RGB image with hand pose projections overlaid
        rgb_with_overlay = rgb_frame.copy()
        for point in haptic_valid_projected:
            x, y = int(point[0]), int(point[1])
            cv2.circle(rgb_with_overlay, (x, y), 2, (0, 0, 255), -1)  # Blue circles
        rr.log("world/camera/rgb_with_overlay", rr.Image(rgb_with_overlay))
        
    print("Visualization complete! Check the Rerun viewer.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lerobot dataset with haptic hand poses in Rerun",
        epilog="Example: python rerun_lerobot.py --dset_name sriramsk/mug_on_platform_20250910_human/ --video_index 0 --intrinsics assets/examples/0910_0/intrinsics.txt --haptic_pose_dir /data/sriram/lerobot_extradata"
    )
    parser.add_argument("--dset_name", default="sriramsk/mug_on_platform_20250910_human/", help="Dataset name")
    parser.add_argument("--video_index", type=int, default=0, help="Video index")
    parser.add_argument("--intrinsics", default="assets/examples/0910_0/intrinsics.txt", help="Path to camera intrinsics file")
    parser.add_argument("--haptic_pose_dir", default="/data/sriram/lerobot_extradata", help="Path to haptic hand pose directory")

    args = parser.parse_args()
    
    visualize_lerobot_sequence(args.dset_name, args.video_index, args.intrinsics, args.haptic_pose_dir)


if __name__ == "__main__":
    main()
