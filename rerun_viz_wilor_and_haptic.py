#!/usr/bin/env python3
import numpy as np
import cv2
import imageio.v3 as iio
import rerun as rr
import av
from pathlib import Path
import argparse
import pickle
import glob
from tqdm import tqdm


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


def load_hand_poses(hand_pose_path):
    hand_poses = np.load(hand_pose_path)
    return hand_poses


def load_haptic_hand_poses(haptic_dir):
    haptic_dir = Path(haptic_dir)
    pkl_files = sorted(glob.glob(str(haptic_dir / "*.pkl")))
    
    hand_poses = []
    intrinsics_list = []
    
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            
            wHands = data['wHands'].detach().cpu().numpy().squeeze()
            intr = data['intr'].detach().cpu().numpy().squeeze()
            
            hand_poses.append(wHands)
            intrinsics_list.append(intr)
    
    return np.array(hand_poses), np.array(intrinsics_list)


def visualize_rgbd_sequence(rgb_video_path, depth_video_path, intrinsics_path, hand_pose_path=None, haptic_pose_dir=None):
    rr.init("RGBD_Visualization", spawn=True)
    #rr.serve(open_browser=False, web_port=9090, ws_port=9877)
    
    # Load camera intrinsics
    intrinsics = load_intrinsics(intrinsics_path)
    # Read RGB and depth videos
    rgb_frames = iio.imread(rgb_video_path)
    depth_frames = read_depth_video(depth_video_path)

    hand_poses = load_hand_poses(hand_pose_path)
    haptic_poses, haptic_intrinsics = load_haptic_hand_poses(haptic_pose_dir)

    # Ensure all data have the same number of frames
    assert len(rgb_frames) == len(depth_frames)
    assert len(rgb_frames) == len(hand_poses)
    assert len(rgb_frames) == len(haptic_poses)

    # Set up camera pinhole model for visualization
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = rgb_frames[0].shape[:2]

    haptic_fx = haptic_intrinsics[0, 0, 0]
    scale_factor = fx / haptic_fx
    
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
        
        hand_pose_frame = hand_poses[frame_idx]
        # Log wilor hand in red
        rr.log("world/hand/wilor", rr.Points3D(hand_pose_frame, colors=[1.0, 0.0, 0.0]))  # Red for wilor

        # Project wilor hand poses to 2D and overlay on RGB image
        wilor_projected = project_3d_to_2d(hand_pose_frame, intrinsics)

        # Filter points that are within image bounds and have positive depth
        h, w = rgb_frame.shape[:2]
        wilor_valid_mask = (
            (hand_pose_frame[:, 2] > 0.1) &  # Positive depth
            (wilor_projected[:, 0] >= 0) & (wilor_projected[:, 0] < w) &  # Within width
            (wilor_projected[:, 1] >= 0) & (wilor_projected[:, 1] < h)    # Within height
        )

        wilor_valid_projected = wilor_projected[wilor_valid_mask]
        
        haptic_pose_frame = haptic_poses[frame_idx]
        haptic_intrinsics_frame = haptic_intrinsics[frame_idx]

        # Scale only Z-coordinate to match RGB camera coordinate system
        haptic_fx = haptic_intrinsics_frame[0, 0]
        rgb_fx = intrinsics[0, 0]
        z_scale_factor = rgb_fx / haptic_fx

        # Apply Z-only scaling to haptic hand poses for 3D visualization
        z_scaled_haptic_pose = haptic_pose_frame.copy()
        z_scaled_haptic_pose[:, 2] *= z_scale_factor

        # Log Z-scaled haptic hand in blue
        rr.log("world/hand/haptic", rr.Points3D(z_scaled_haptic_pose, colors=[0.0, 0.0, 1.0]))  # Blue for haptic

        # Project haptic hand poses to 2D using haptic intrinsics
        haptic_projected = project_3d_to_2d(haptic_pose_frame, haptic_intrinsics_frame)

        # Filter points that are within image bounds and have positive depth
        h, w = rgb_frame.shape[:2]
        haptic_valid_mask = (
            (haptic_pose_frame[:, 2] > 0.1) &  # Positive depth
            (haptic_projected[:, 0] >= 0) & (haptic_projected[:, 0] < w) &  # Within width
            (haptic_projected[:, 1] >= 0) & (haptic_projected[:, 1] < h)    # Within height
        )

        haptic_valid_projected = haptic_projected[haptic_valid_mask]
            
        # Create RGB image with both hand pose projections overlaid
        rgb_with_overlay = rgb_frame.copy()
            
        for point in wilor_valid_projected:
            x, y = int(point[0]), int(point[1])
            cv2.circle(rgb_with_overlay, (x, y), 2, (255, 0, 0), -1)  # Red circles
            
        for point in haptic_valid_projected:
            x, y = int(point[0]), int(point[1])
            cv2.circle(rgb_with_overlay, (x, y), 2, (0, 0, 255), -1)  # Blue circles
            
        # Log the RGB image with both hand pose projections
        rr.log("world/camera/rgb_with_overlay", rr.Image(rgb_with_overlay))
        
    print("Visualization complete! Check the Rerun viewer.")


def main():
    parser = argparse.ArgumentParser(description="Visualize RGB-D video sequence with Rerun")
    parser.add_argument("--vid_name", default="0910_0", help="Dataset name")

    args = parser.parse_args()
    
    # Validate paths
    rgb_path = Path(f"assets/examples/{args.vid_name}/video.mp4")
    depth_path = Path(f"assets/examples/{args.vid_name}/depth.mkv")
    intrinsics_path = Path(f"assets/examples/{args.vid_name}/intrinsics.txt")
    wilor_hand_pose_path = Path(f"assets/examples/{args.vid_name}/wilor_hand_pose.npy")
    haptic_hand_pose_path = Path(f"output/release/mix_all/demo/{args.vid_name}_right/")
    visualize_rgbd_sequence(rgb_path, depth_path, intrinsics_path, wilor_hand_pose_path, haptic_hand_pose_path)


if __name__ == "__main__":
    main()
