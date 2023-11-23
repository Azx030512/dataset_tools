
import numpy as np
import os
import argparse
from collections import defaultdict
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

def interpolate_pose(pose_begin, pose_end, interp_frame_num = 15):
    poses_test = pose_begin[None, ...].repeat(interp_frame_num, axis=0)

    # 将旋转矩阵转换为四元数
    pose_begin_rot = pose_begin[:, :3]
    pose_end_rot = pose_end[:, :3]
    Rot = Rotation.from_matrix([pose_begin_rot, pose_end_rot])
    key_times = [0, 1]
    slerp = Slerp(key_times, Rot)
    times = np.linspace(0, 1, interp_frame_num)
    pose_rot_interp = slerp(times).as_matrix()  # 旋转
    delta_trans = (pose_end[:, 3] - pose_begin[:, 3]) / interp_frame_num  # 平移

    for i in range(interp_frame_num):
        poses_test[i, :, 3] += delta_trans * i
        poses_test[i, :, :3] = pose_rot_interp[i]
    return poses_test


transform_info = json.load(open('/home/azx/dataset/pond/json/transform_info.json'))
start_img="IMG20231117103741.jpg"
end_img="IMG20231117103709.jpg"
midway_imgs = ["IMG20231117103731.jpg","IMG20231117103723.jpg"]

midway_poses = [i for i in range(len(midway_imgs))]

for index, info in transform_info.items():
    if start_img == info["image_name"]:
        pose_begin = np.array(info["transform_matrix"])
    elif end_img == info["image_name"]:
        pose_end = np.array(info["transform_matrix"])
    elif info["image_name"] in midway_imgs:
        midway_poses[midway_imgs.index(info["image_name"])] = np.array(info["transform_matrix"])

scale = transform_info['1']["scale"]
pose_begin[..., 3] *= scale
pose_end[..., 3] *= scale
for i in range(len(midway_poses)):
    midway_poses[i][..., 3] *= scale

if len(midway_imgs) == 0:
    trajectory = interpolate_pose(pose_begin, pose_end, interp_frame_num = 30)
else:
    trajectory = []
    begin_poses = interpolate_pose(pose_begin, midway_poses[0], interp_frame_num = 15)
    trajectory.append(begin_poses)
    for i in range(len(midway_poses)-1):
        mid_poses = interpolate_pose(midway_poses[i], midway_poses[i+1], interp_frame_num = 15)
        trajectory.append(mid_poses)
    end_poses = interpolate_pose(midway_poses[-1], pose_end, interp_frame_num = 15)
    trajectory.append(end_poses)
    trajectory = np.concatenate(trajectory,axis=0)
np.save("interpolate_trajectory.npy", trajectory)

