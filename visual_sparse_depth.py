from datasets.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

import numpy as np
import os
import argparse
import cv2
from collections import defaultdict
import json
from tqdm import tqdm


## 该代码用于记录每一张图对应的稀疏深度图

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/yangzesong/Projects/NeRF/datasets/room2/dense4',
                        help='root directory of dataset')
    parser.add_argument('--visual_depth', type=bool,
                        default=True,
                        help='visualize the depth')
    parser.add_argument('--image_downscale', type=int,
                        default=1,
                        help='scale the image')
    return vars(parser.parse_args())


def get_c2w_poses(imdata):
    poses = []
    for i in imdata:
        R = imdata[i].qvec2rotmat()
        t = imdata[i].tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)


def extract_image_depth(root_dir):
    # 先从image.bin中提取图片的pose
    imdata = read_images_binary(os.path.join(root_dir, 'sparse/0/images.bin'))
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME, point3D_ids, xys[三维特征点及其对应的二维坐标]
    pts3d = read_points3d_binary(os.path.join(root_dir, 'sparse/0/points3D.bin'))
    # POINT3D_ID,[X,Y,Z],[R,G,B], ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    Errs = np.array([point3D.error for point3D in pts3d.values()])  # 投影误差
    Err_mean = np.mean(Errs)

    poses = get_c2w_poses(imdata)

    # 获取相机的pose
    data_list = {}
    # 找到每一张图片上特征点对应的深度，二维坐标以及投影误差
    for img_idx, image in enumerate(imdata.values()):
        depth_list = []
        coord_list = []
        weight_list = []

        R = image.qvec2rotmat()
        t = image.tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        # 可视化二维特征点
        for i in range(len(image.xys)):
            # 每一张图的特征点 POINTS2D[] as (X, Y, POINT3D_ID)
            # 这张图上的特征点不一定会出现在稀疏点云上，所以可能是-1
            point2D = image.xys[i]  # 该点对应的二维坐标
            id_3D = image.point3D_ids[i]
            if id_3D == -1:  # 该二维坐标并没有被用到3D稀疏点云上
                continue
            point3D = pts3d[id_3D].xyz
            # 获取该点对应的深度
            #           Z轴
            depth = (c2w[:3, 2].T @ (point3D - c2w[:3, 3]))  # * sc
            # 相当于(pose*vector).sum()

            err = pts3d[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)  # weight越大，error越小

            depth_list.append(depth)
            coord_list.append(point2D)  # factor之后的二维坐标
            weight_list.append(weight)

        data_list[image.name] = {"depth": depth_list, "coord": coord_list, "weight": weight_list}

    return data_list


def visual_depth(root_dir, depth_info, img_info, image_downscale=1):
    save_dir = os.path.join(root_dir, f"visual_feature_{image_downscale}")
    os.makedirs(save_dir, exist_ok=True)
    max_depth = 0
    for depth in depth_info:
        if depth_info[depth]["depth"] != []:
            max_depth = max(depth_info[depth]["depth"]) if max(depth_info[depth]["depth"]) > max_depth else max_depth
    print(max_depth)
    for img_idx in tqdm(depth_info):
        image_info = depth_info[img_idx]
        image = cv2.imread(os.path.join(root_dir, "images", img_idx))  # img_info[img_idx]["image_name"]))
        image = cv2.resize(image, [image.shape[1] // image_downscale, image.shape[0] // image_downscale])

        depths = np.array(image_info["depth"])
        coords = np.array(image_info["coord"])
        depth_colors = depths / max_depth * 255
        for idx, coord in enumerate(coords):
            depth_color = depth_colors[idx]
            cv2.circle(image, (int(coord[0] / image_downscale), int(coord[1] / image_downscale)), 4 // image_downscale,
                       (0, 0, depth_color), -1)
        cv2.imwrite(os.path.join(save_dir, img_idx), image)


if __name__ == "__main__":
    hparams = get_opts()
    print(hparams)

    depth_info = extract_image_depth(hparams["root_dir"])

    with open(os.path.join(hparams["root_dir"], f'json/transform_info.json')) as fp:
        img_info = json.load(fp)

    visual_depth(hparams["root_dir"], depth_info, img_info)
