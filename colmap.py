import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from PIL import Image
from einops import rearrange
import imageio
from torchvision import transforms
from kornia import create_meshgrid
import cv2

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from collections import defaultdict

# mask
mask_label = [0, 1, 2, 3, 7, 9]


def get_ray_directions(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24json
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx + 0.5) / fx, (j - cy + 0.5) / fy, torch.ones_like(i)], -1)  # (H, W, 3)
    return directions


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d


def read_depth(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def view_depth(depth_map):
    x = np.nan_to_num(depth_map)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)

    cmap = cv2.COLORMAP_JET
    x_ = cv2.applyColorMap(x, cmap)
    Image.fromarray(x_).show()


class ColmapDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 split='train', 
                 downsample=1.0,
                 use_segmentation=False, 
                 segmentation_net=None,
                 use_DS=False,
                 trajectory=[],
                 **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.transform = transforms.ToTensor()
        self.use_segmentation = use_segmentation
        self.use_DS = use_DS
        self.trajectory = trajectory
        if use_segmentation:
            self.predictor = segmentation_net

        with open(os.path.join(self.root_dir, f'json/transform_info.json')) as fp:
            self.meta = json.load(fp)

        if self.split == "train" or self.split == "val":  # val主要是为了提取mask
            self.read_meta(split, **kwargs)

        if self.split == "eval_single_block":
            img_trajectory = kwargs["begin_end"]
            # 在两张图之间做插值
            self.eval_begin = img_trajectory[0]
            trajectory_begin = img_trajectory[:-1]
            trajectory_end = img_trajectory[1:]

            self.interp_frame_num = kwargs["interp_frame_num"] // len(trajectory_begin)

            self.poses_traj = []
            self.cam_index = []

            for index, (image_begin, image_end) in enumerate(zip(trajectory_begin, trajectory_end)):
                print("Begin:{0}, End:{1}".format(self.meta[image_begin]["image_name"],
                                                  self.meta[image_end]["image_name"]))

                pose_begin = np.array(self.meta[image_begin]["transform_matrix"])
                pose_end = np.array(self.meta[image_end]["transform_matrix"])

                cam_index = np.array([self.meta[image_begin]["cam_idx"], self.meta[image_end]["cam_idx"]])[
                    None, ...].repeat(self.interp_frame_num, axis=0)

                poses_test = pose_begin[None, ...].repeat(self.interp_frame_num, axis=0)

                # 将旋转矩阵转换为四元数
                pose_begin_rot = pose_begin[:, :3]
                pose_end_rot = pose_end[:, :3]
                Rot = Rotation.from_matrix([pose_begin_rot, pose_end_rot])
                key_times = [0, 1]
                slerp = Slerp(key_times, Rot)
                times = np.linspace(0, 1, self.interp_frame_num)
                pose_rot_interp = slerp(times).as_matrix()  # 旋转
                delta_trans = (pose_end[:, 3] - pose_begin[:, 3]) / self.interp_frame_num  # 平移

                for i in range(self.interp_frame_num):
                    # z+0.03，绕x轴旋转90度
                    poses_test[i, :, 3] += delta_trans * i
                    poses_test[i, :, :3] = pose_rot_interp[i]

                self.poses_traj.append(poses_test)
                self.cam_index.append(cam_index)
            self.poses_test = np.concatenate(self.poses_traj, axis=0)
            self.cam_index = np.concatenate(self.cam_index, axis=0)





    def read_meta(self, split, **kwargs):
        self.val_index = 15
        if split == "train":
            self.rgbs = []  # rgb
            self.rays_dir = []
            self.rays_origin = []
            self.ts = []

            if self.use_DS:
                self.depths = []
                self.depth_min_prec = 5
                self.depth_max_prec = 95
                depth_max = 0
                depth_min = 1000

            for img_idx, index in enumerate(tqdm(self.meta)):
                img_info = self.meta[index]
                c2w = torch.FloatTensor(img_info['transform_matrix'])
                
                height, width = img_info["height"], img_info["width"]
                if not self.use_segmentation and self.downsample > 1.0:
                    height = height // int(self.downsample)
                    width = width // int(self.downsample)
                
                # K用于occupy grid进行初始化
                K = np.zeros((3, 3), dtype=np.float32)
                K[0, 0] = img_info['intrinsics'][0]  # fx
                K[1, 1] = img_info['intrinsics'][1]  # fy

                if not self.use_segmentation and self.downsample > 1.0:
                    K[0, 0] = K[0, 0] / self.downsample
                    K[1, 1] = K[1, 1] / self.downsample

                K[0, 2] = width / 2  # cx
                K[1, 2] = height / 2  # cy
                K[2, 2] = 1
                self.K = torch.FloatTensor(K)

                img = Image.open(os.path.join(self.root_dir,'images',img_info["image_name"])).convert('RGB')
                if not self.use_segmentation and self.downsample > 1.0:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                
                # 读取img的mask
                if self.use_segmentation:
                    # 筛选mask
                    seg_out = self.predictor(cv2.imread(os.path.join(self.root_dir,'images',img_info["image_name"])))["instances"]
                    mask = []
                    for pred_class, pred_mask in zip(seg_out.pred_classes, seg_out.pred_masks):
                        if pred_class in mask_label:
                            mask.append(pred_mask.cpu())
                    mask.append(torch.zeros([img.height, img.width]) == 1)  # 防止什么都没检测出来
                    img_mask = ~torch.any(torch.stack(mask, dim=0), dim=0).view(-1)

                img = self.transform(img)  # (3,h,w)
                img = img.view(3, -1).permute(1, 0)
                if self.use_segmentation:
                    img = img[img_mask]
                self.rgbs.append(img)

                # depth load
                if self.use_DS:
                    depth_img = read_depth(img_info["depth_path"])
                    min_depth, max_depth = np.percentile(depth_img, [self.depth_min_prec, self.depth_max_prec])
                    depth_img[depth_img < min_depth] = min_depth
                    depth_img[depth_img > max_depth] = max_depth

                    depth_scale_factor = img_info["scale"]
                    # view_depth(depth_img / depth_scale_factor)
                    if self.downsample != 1:
                        depth_img = np.array(
                            Image.fromarray(depth_img).resize((width, height), Image.Resampling.NEAREST))
                    depth = torch.Tensor((depth_img / depth_scale_factor).astype(np.float32))
                    valid_depth = torch.logical_and(depth > (min_depth / depth_scale_factor),
                                                    depth < (max_depth / depth_scale_factor))
                    depth_max = depth[valid_depth].max() if depth[valid_depth].max() > depth_max else depth_max
                    depth_min = depth[valid_depth].min() if depth[valid_depth].min() < depth_min else depth_min

                    depth = depth.view(-1, 1)
                    valid_depth = valid_depth.view(-1, 1)
                    self.depths.append(torch.cat([depth, valid_depth], -1))

                ts = int(img_idx)
                self.ts.append(ts * torch.ones_like(img[:, :1]))

                # load rays
                directions = get_ray_directions(height, width, K)
                rays_o, rays_d = get_rays(directions, c2w)

                rays_d = rays_d.view(-1, 3)
                if self.use_segmentation:
                    rays_d = rays_d[img_mask]
                self.rays_dir += [rays_d]
                rays_o = rays_o.view(-1, 3)
                if self.use_segmentation:
                    rays_o = rays_o[img_mask]
                self.rays_origin += [rays_o]

            # 读取rays
            self.N_image = len(self.rgbs)
            self.rgbs = torch.cat(self.rgbs, 0)
            self.rays_origin = torch.cat(self.rays_origin, 0)
            self.rays_dir = torch.cat(self.rays_dir, 0)
            self.ts = torch.cat(self.ts, 0)
            print(f"Total {self.N_image} images and {len(self.rgbs)} rays...")

            if self.use_DS:
                print(f"Max depth:{depth_max},  Min depth:{depth_min}")
                self.depth_max = depth_max
                self.depth_min = depth_min
                self.depths = torch.cat(self.depths, 0)

            if self.use_segmentation:
                del self.predictor
            torch.cuda.empty_cache()

        elif split == "val":
            # 这里只是为了先求val的seg_mask
            if self.use_segmentation:
                self.img_mask = None
                index = list(self.meta.keys())[self.val_index]
                img_info = self.meta[index]

                seg_out = self.predictor(cv2.imread(os.path.join(self.root_dir,'images',img_info["image_name"])))["instances"]
                mask = []
                for pred_class, pred_mask in zip(seg_out.pred_classes, seg_out.pred_masks):
                    if pred_class in mask_label:
                        mask.append(pred_mask.cpu())
                mask.append(torch.zeros([img_info["height"], img_info["width"]]) == 1)
                self.img_mask = ~torch.any(torch.stack(mask, dim=0), dim=0).cpu()
                del self.predictor
                torch.cuda.empty_cache()

    def __len__(self):
        if self.split.startswith('train'):
            return len(self.rgbs)  # 每个epoch跑1000次，每次选8192张图，每张图选8192条光线
        elif self.split == "val":
            return 1
        elif self.split == "eval_single_block" or "compose_blocks" in self.split:
            return len(self.poses_test)
        elif self.split == "eval_val":
            return len(self.val_images.keys())
        elif self.split == "trajectory":
            return len(self.trajectory)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            sample = {'rays_o': self.rays_origin[idx],
                      "rays_d": self.rays_dir[idx],
                      "rgb": self.rgbs[idx],
                      "ts": self.ts[idx].long()}
            if self.use_DS:
                sample["depth"] = self.depths[idx]

        elif self.split == "val":  # 可以选择用pose来生成rays
            index = list(self.meta.keys())[self.val_index]
            img_info = self.meta[index]
            print("The image to validate is {0}".format(img_info["image_name"]))

            c2w = torch.FloatTensor(img_info['transform_matrix'])

            height, width = img_info["height"], img_info["width"]

            # K用于occupy grid进行初始化
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0] = img_info['intrinsics'][0]  # fx
            K[1, 1] = img_info['intrinsics'][1]  # fy
            K[0, 2] = width / 2  # cx
            K[1, 2] = height / 2  # cy
            K[2, 2] = 1
            self.K = torch.FloatTensor(K)

            img = Image.open(os.path.join(self.root_dir,'images',img_info["image_name"])).convert('RGB')
            # 读取img的mask
            img_mask = torch.zeros([img.height, img.width]) == 0
            if self.use_segmentation:
                img_mask = self.img_mask

            img = self.transform(img)  # (3,h,w)
            img = img.view(3, -1).permute(1, 0)

            # load rays
            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)

            ts = int(self.val_index) * torch.ones(len(rays_o), 1)

            img_wh = [img_info["width"], img_info["height"]]

            sample = {'rays_o': rays_o,
                      "rays_d": rays_d,
                      "rgb": img,
                      "ts": ts.long(),
                      "img_wh": img_wh,
                      "img_name": img_info["image_name"]}
            sample["mask"] = img_mask

        elif self.split == "eval_single_block":
            img_info = self.meta[self.eval_begin]
            c2w = torch.FloatTensor(self.poses_test[idx])

            height, width = img_info["height"] // self.downsample, img_info["width"] // self.downsample

            # K用于occupy grid进行初始化
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0] = img_info['intrinsics'][0] / self.downsample  # fx
            K[1, 1] = img_info['intrinsics'][1] / self.downsample  # fy
            K[0, 2] = width / 2  # cx
            K[1, 2] = height / 2  # cy
            K[2, 2] = 1
            self.K = torch.FloatTensor(K)

            # load rays
            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)
            directions = (directions / torch.norm(directions, dim=-1, keepdim=True)).view(-1, 3)

            ts = (int(self.eval_begin) - 1) * torch.ones(len(rays_o), 1)

            img_wh = [width, height]

            sample = {'rays_o': rays_o,
                      "rays_d": rays_d,
                      "direction": directions,
                      "ts": ts.long(),
                      "img_wh": img_wh,
                      "img_name": img_info["image_name"]}
        elif self.split == "trajectory":
            img_info = self.meta["1"]
            pose=self.trajectory[idx]
            c2w = torch.FloatTensor(pose[ :3])
            c2w[..., 3] /= img_info['scale']
            height = img_info["height"] // self.downsample
            width = img_info["width"] // self.downsample
            # K用于occupy grid进行初始化
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0] = img_info['intrinsics'][0] / self.downsample  # fx
            K[1, 1] = img_info['intrinsics'][1] / self.downsample  # fy
            K[0, 2] = width / 2  # cx
            K[1, 2] = height / 2  # cy
            K[2, 2] = 1
            self.K = torch.FloatTensor(K)

            # load rays
            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)
            directions = (directions / torch.norm(directions, dim=-1, keepdim=True)).view(-1, 3)

            ts = torch.ones(len(rays_o), 1)

            img_wh = [width, height]

            sample = {'rays_o': rays_o,
                      "rays_d": rays_d,
                      "direction": directions,
                      "ts": ts.long(),
                      "img_wh": img_wh,
                      "img_name": img_info["image_name"]}
        return sample
