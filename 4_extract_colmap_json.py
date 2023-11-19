import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob  # 查找符合自己要求的文件，
from colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

import numpy as np
import os
import argparse
from collections import defaultdict
import json
from tqdm import tqdm


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/azx/dataset/seek_truth_lecture_hall_vertical',
                        help='root directory of dataset')

    return vars(parser.parse_args())


def find_cam_param(cam_idx, camdata):
    # 根据提供的图片对应的cam_idx，返回对应的width、height和params
    for cam in camdata:
        if cam_idx == camdata[cam].id:
            if camdata[cam].model == "SIMPLE_PINHOLE" or camdata[cam].model == "SIMPLE_RADIAL":
                params = np.array([camdata[cam].params[0], camdata[cam].params[0],
                                   camdata[cam].params[1], camdata[cam].params[2]])
                return [camdata[cam].width, camdata[cam].height, params]
            return [camdata[cam].width, camdata[cam].height, camdata[cam].params]


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
    else:
        center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg


def center_poses(poses_no_center, pts3d=None):
    # 将poses转换为np.array
    poses = np.array([pose for pose in poses_no_center.values()])
    pose_avg = average_poses(poses, None)  # (3, 4)，平均位姿&点云的中心
    # 只需要center
    translate = pose_avg[:, 3]
    poses[:, :, 3] -= translate
    poses_centered = poses

    scale = np.linalg.norm(poses_centered[..., 3], axis=-1).max()  # 让其在[-1,1]内
    poses_centered[..., 3] /= scale

    for pose, pose_centered in zip(poses_no_center, poses_centered):
        poses_no_center[pose] = pose_centered
    return poses_no_center, scale


def extract_json(root_dir):
    # 先从image.bin中提取图片的索引，图片对应的idx，图片
    # image.txt:IMAGE_ID, [QW, QX, QY, QZ] , [TX, TY, TZ], CAMERA_ID, NAME
    imdata = read_images_binary(os.path.join(root_dir, 'sparse/0/images.bin'))
    camdata = read_cameras_binary(os.path.join(root_dir, 'sparse/0/cameras.bin'))
    # camera.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    img_info = defaultdict(dict)
    Extrinsics = {}
    World2Cam = {}

    for im_idx in tqdm(imdata):
        img_info[im_idx] = {}
        img = imdata[im_idx]
        img_info[im_idx]["image_name"] = img.name
        img_info[im_idx]["rgb_path"] = os.path.join(root_dir, "images", img.name)
        img_info[im_idx]["depth_path"] = os.path.join(root_dir, "depth_maps", f"{img.name}.geometric.bin")

        img_info[im_idx]["idx"] = img.id
        img_info[im_idx]["cam_idx"] = img.camera_id
        # 根据图片拍摄的对应相机的idx返回其对应的width、height和params
        width, height, intrinsics = find_cam_param(img.camera_id, camdata)
        img_info[im_idx]["width"] = width
        img_info[im_idx]["height"] = height
        img_info[im_idx]["intrinsics"] = intrinsics.tolist()
        # world_2_cam
        R = img.qvec2rotmat()  # 四元数变换为旋转矩阵
        t = img.tvec.reshape(3, 1)
        # w2c = np.concatenate([R, t], 1)
        w2c = np.zeros([4, 4])
        w2c[:3, :3] = R
        w2c[:3, 3] = t.squeeze()
        w2c[3, 3] = 1
        World2Cam[im_idx] = w2c
        # 没有问题

        # cam_2_world
        # *****************************************很重要**********************************************
        transform_matrix = np.linalg.inv(w2c)[:3]

        Extrinsics[im_idx] = transform_matrix
        # img_info[im_idx]["transform_matrix"] = transform_matrix.tolist()

    pts3d = read_points3d_binary(os.path.join(root_dir, 'sparse/0/points3D.bin'))
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
    xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
    # 用点云引导位姿中心化
    Extrinsics, scale = center_poses(Extrinsics, xyz_world)  # 位姿中心化 或者只调整中心就行
    # 在json中添加外参和near和far
    for idx in img_info:
        img_info[idx]["transform_matrix"] = Extrinsics[idx].tolist()
        img_info[idx]["scale"] = scale

    return img_info


if __name__ == "__main__":
    hparams = get_opts()
    print(hparams)

    img_info = extract_json(hparams['root_dir'])
    print("Finish convert the colmap file into json!")

    os.makedirs(os.path.join(hparams['root_dir'], "json"), exist_ok=True)
    save_path = os.path.join(hparams['root_dir'], "json/transform_info.json")

    imgs = {}
    for idx in tqdm(img_info):
        imgs[idx] = img_info[idx]
        with open(save_path, "w") as fp:
            json.dump(imgs, fp)
            fp.close()

    print("The information has been saved in the path of {0}".format(save_path))
