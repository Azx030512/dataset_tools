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


def get_nearest_info(c2w, meta, block_index, cam_idx):
    # 首先是选取距离最近的5个相机位姿
    name_list = []
    poses_distance = []
    origin = c2w[:, 3]
    for element in block_index:
        img_info = meta[element[0]]
        if img_info["cam_idx"] in cam_idx:
            current_origin = torch.tensor(img_info['transform_matrix'])[:, 3]
            distance = (origin - current_origin).norm()
            poses_distance.append(distance)
            name_list.append(element[0])
    poses_distance = torch.stack(poses_distance, dim=0)
    _, sort_index = torch.sort(poses_distance)

    nearest_elements = np.array(name_list)[sort_index.numpy()].tolist()[:3]
    # 然后选择旋转矩阵最为接近的相机位姿
    rotate = c2w[:, :3].numpy()
    rotvec = Rotation.from_matrix([rotate]).as_rotvec()
    min_distance = 1000
    nearest_info = None
    for element in nearest_elements:
        img_info = meta[element]
        current_rotate = np.array(img_info['transform_matrix'])[:, :3]
        current_rotvec = Rotation.from_matrix([current_rotate]).as_rotvec()
        cos = np.clip((np.trace(np.dot(rotvec.T, current_rotvec)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))
        if e_R < min_distance:
            nearest_info = img_info
            min_distance = e_R
    return nearest_info


def DistanceWeight(point, centroid, p=1):
    weight = (point - centroid).norm() ** (-p)
    weight = min(weight, 10000)  # 有可能刚好是中点
    return weight


def find_nearest_cam_pose(pose_epoch, cam_index, meta, split_json, block_range):
    block_start, block_end = block_range  # 36，40 -> 36,37,38,39,40
    block_indexes = np.arange(block_start, block_end + 1)
    min_distance = 1000
    nearest_poses = None
    for block_index in block_indexes:
        block_elements = split_json[f"block_{block_index}"]["elements"]
        for element in block_elements:
            img_info = meta[element[0]]
            if img_info["cam_idx"] == cam_index:
                current_origin = np.array(img_info['transform_matrix'])[:, 3]
                distance = ((pose_epoch[:, 3] - current_origin) ** 2).sum()
                if distance < min_distance:
                    min_distance = distance
                    nearest_poses = np.array(img_info['transform_matrix'])  # 只要rotation
    return nearest_poses


class WaymoDataset(Dataset):
    def __init__(self, root_dir, split='train', block_index="block_0", downsample=1.0,
                 use_segmentation=False, segmentation_net=None, scale_pose=False, pose_scale_ratio=1,
                 use_rays=True,
                 **kwargs):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.transform = transforms.ToTensor()
        self.block_index = block_index
        self.use_segmentation = use_segmentation
        self.scale_pose = scale_pose
        self.use_rays = use_rays
        if use_segmentation:
            self.predictor = segmentation_net

        if split == "val":
            if self.scale_pose:
                self.pose_scale_ratio = pose_scale_ratio

        if split == "test" or split == "train" or "compose_blocks" in split or (
                "eval" in split and split != "eval_val"):
            with open(os.path.join(self.root_dir, f'json/train.json'), 'r') as fp:
                self.meta = json.load(fp)  # 所有image的信息
            with open(os.path.join(self.root_dir, f'json/split_block_train.json'), 'r') as fp:  # 每一块的情况
                self.block_split_info = json.load(fp)  # 每一个block包含的对象
        elif split == "val" or split == "eval_val":
            with open(os.path.join(self.root_dir, f'json/val.json'), 'r') as fp:
                self.meta = json.load(fp)  # 所有image的信息
            with open(os.path.join(self.root_dir, f'json/split_block_val.json'), 'r') as fp:  # 每一块的情况
                self.block_split_info = json.load(fp)  # 每一个block包含的对象
            if split == "eval_val":
                with open(os.path.join(self.root_dir, f'json/split_block_train.json'), 'r') as fp:  # 每一块的情况
                    self.block_split_train_info = json.load(fp)  # 每一个block包含的对象

        with open(os.path.join(root_dir, f'json/block_center_translate.json'), 'r') as fp:
            self.block_center_translate = json.load(fp)  # 所有image的信息

        # 加载translate和center
        if "compose_blocks" not in split:  # compose_blocks不需要减去translate
            center_translate = self.block_center_translate[block_index]  # center & translate
            self.translate = torch.tensor(center_translate["translate"])
            self.center = torch.FloatTensor(center_translate["center"])  # self.rays_origin.mean(0)  # model_center

        if self.split == "train" or self.split == "val":  # val主要是为了提取mask
            self.read_meta(split, **kwargs)

        if self.split == "eval_single_block":
            self.filter_closest_element = kwargs["filter_closest_element"]
            img_trajectory = kwargs["begin_end"]
            # 在两张图之间做插值
            self.eval_begin = img_trajectory[0]
            self.interp_frame_num = kwargs["interp_frame_num"]
            trajectory_begin = img_trajectory[:-1]
            trajectory_end = img_trajectory[1:]

            self.poses_traj = []
            self.cam_index = []

            for index, (image_begin, image_end) in enumerate(zip(trajectory_begin, trajectory_end)):
                print(f"Begin:{image_begin}, End:{image_end}")

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
                    '''
                    
                    '''
                    poses_test[i, :, 3] += delta_trans * i
                    poses_test[i, :, :3] = pose_rot_interp[i]

                self.poses_traj.append(poses_test)
                self.cam_index.append(cam_index)
            self.poses_test = np.concatenate(self.poses_traj, axis=0)
            self.cam_index = np.concatenate(self.cam_index, axis=0)

        if self.split == "compose_blocks":
            self.radius = kwargs["radius"]
            # self.filter_closest_element = kwargs["filter_closest_element"]
            self.img_trajectory = kwargs["block_begin_end"]
            # 在两张图之间做插值
            self.eval_block_begin = self.img_trajectory[0]
            self.eval_block_end = self.img_trajectory[-1]
            self.cam_index = kwargs["cam_idx"]
            self.interp_frame_num = kwargs["interp_frame_num"]

            # 找到当前两个block对应cam_index的始终点
            # 首先找最前面的block
            for element in self.block_split_info[self.eval_block_begin]["elements"]:
                image_name = element[0]
                img_info = self.meta[image_name]
                if img_info["cam_idx"] == self.cam_index:
                    first_element = img_info
                    break
            # 倒序
            self.block_split_info[self.eval_block_end]["elements"].reverse()
            for element in self.block_split_info[self.eval_block_end]["elements"]:
                image_name = element[0]
                img_info = self.meta[image_name]
                if img_info["cam_idx"] == self.cam_index:
                    last_element = img_info
                    break
            print(
                f"The first and last element of {self.cam_index} between {self.eval_block_begin} and {self.eval_block_end} is ",
                first_element["image_name"], " and ", last_element["image_name"])  # 467429854.png 2017858206.png

            self.first_element = first_element
            self.poses_traj = []

            pose_begin = np.array(first_element["transform_matrix"])
            pose_end = np.array(last_element["transform_matrix"])
            poses_test = pose_begin[None, ...].repeat(self.interp_frame_num, axis=0)

            pose_begin_rot = pose_begin[:, :3]
            pose_end_rot = pose_end[:, :3]
            Rot = Rotation.from_matrix([pose_begin_rot, pose_end_rot])
            key_times = [0, 1]
            slerp = Slerp(key_times, Rot)
            times = np.linspace(0, 1, self.interp_frame_num)
            pose_rot_interp = slerp(times).as_matrix()  # 旋转
            delta_trans = (pose_end[:, 3] - pose_begin[:, 3]) / self.interp_frame_num  # 平移

            for i in range(self.interp_frame_num):
                poses_test[i, :, 3] += delta_trans * i
                poses_test[i, :, :3] = pose_rot_interp[i]

                rot = np.pi / 2
                camera_rot = np.array([[1, 0.0, 0],
                                       [0.0, np.cos(rot), -np.sin(rot)],
                                       [0, np.sin(rot), np.cos(rot)]], dtype=np.float32)
                poses_test[i, :, :3] = camera_rot @ poses_test[i, :, :3]

            self.poses_traj.append(poses_test)
            # 一般这里会有多个轨迹
            self.poses_test = np.concatenate(self.poses_traj, axis=0)

        if self.split == "eval_val":
            # 筛选出总共有哪些图需要val
            self.val_images = defaultdict(list)
            block_start, block_end = kwargs["block_begin_end"]
            block_indexes = np.arange(block_start, block_end + 1)
            for block_index in block_indexes:
                # 遍历每一个block查找其含有哪些element
                for element in self.block_split_info[f"block_{block_index}"]["elements"]:
                    self.val_images[element[0]].append(block_index)
            # 还要筛选mask
            if self.use_segmentation:
                self.img_mask = {}
                '''
                val_images=self.val_images
                self.val_images={}
                self.val_images["16759920"]=val_images["16759920"]
                '''
                for val_image in self.val_images:
                    img_info = self.meta[val_image]
                    print()

                    seg_out = self.predictor(
                        cv2.imread(os.path.join(self.root_dir, f'images', img_info['image_name'])))["instances"]
                    mask = []
                    for pred_class, pred_mask in zip(seg_out.pred_classes, seg_out.pred_masks):
                        if pred_class in mask_label:
                            mask.append(pred_mask.cpu())
                    mask.append(torch.zeros([img_info["height"], img_info["width"]]) == 1)
                    img_mask = ~torch.any(torch.stack(mask, dim=0), dim=0).cpu()
                    '''
                    img_mask = ~torch.any(self.predictor(
                        cv2.imread(os.path.join(self.root_dir, f'images', img_info['image_name']))
                    )["instances"].pred_masks, dim=0).cpu()
                    '''
                    self.img_mask[val_image] = img_mask

        if self.split == "compose_blocks_diy_trajectory":
            self.radius = kwargs["radius"]  # 用于筛选block
            # self.filter_closest_element = kwargs["filter_closest_element"]
            # 首先找起始点
            self.img_trajectory = kwargs["block_begin_end"]
            self.block_range = [int(kwargs["block_begin_end"][0].split("_")[-1]),
                                int(kwargs["block_begin_end"][1].split("_")[-1])]
            # 在两张图之间做插值
            self.eval_block_begin = self.img_trajectory[0]
            self.eval_block_end = self.img_trajectory[-1]
            self.cam_main_idx = kwargs["cam_main_idx"]
            self.interp_frame_num = kwargs["interp_frame_num"]

            cam_index_rotate = kwargs["cam_index_rotate"]
            trajectory_cam_begin = cam_index_rotate[:-1]
            trajectory_cam_end = cam_index_rotate[1:]

            epoch_frame_delta = self.interp_frame_num // len(trajectory_cam_begin)
            self.interp_frame_num = epoch_frame_delta * len(trajectory_cam_begin) + 1  # 0~100 总共101

            # 找到当前两个block对应cam_index的始终点
            # 首先找最前面的block
            for element in self.block_split_info[self.eval_block_begin]["elements"]:
                image_name = element[0]
                img_info = self.meta[image_name]
                if img_info["cam_idx"] == self.cam_main_idx:
                    first_element = img_info
                    break
            # 倒序
            self.block_split_info[self.eval_block_end]["elements"].reverse()
            for element in self.block_split_info[self.eval_block_end]["elements"]:
                image_name = element[0]
                img_info = self.meta[image_name]
                if img_info["cam_idx"] == self.cam_main_idx:
                    last_element = img_info
                    break
            print(
                f"The first and last element of {self.cam_main_idx} between {self.eval_block_begin} and {self.eval_block_end} is ",
                first_element["image_name"], " and ", last_element["image_name"])  # 467429854.png 2017858206.png

            self.first_element = first_element
            self.poses_traj = []
            # 先求解出主路径的位姿
            pose_begin = np.array(first_element["transform_matrix"])
            pose_end = np.array(last_element["transform_matrix"])
            poses_test = pose_begin[None, ...].repeat(self.interp_frame_num, axis=0)
            delta_trans = (pose_end[:, 3] - pose_begin[:, 3]) / self.interp_frame_num  # 平移
            for i in range(self.interp_frame_num):
                poses_test[i, :, 3] += delta_trans * i

            # 找到每一段的起始cam
            self.cam_index = []
            for epoch, (cam_begin, cam_end) in enumerate(zip(trajectory_cam_begin, trajectory_cam_end)):
                # 找到当前pose_origin最接近的位姿
                if epoch is not len(trajectory_cam_begin) - 1:
                    cam_index = np.array([cam_begin, cam_end])[None, ...].repeat(epoch_frame_delta, axis=0)
                else:
                    cam_index = np.array([cam_begin, cam_end])[None, ...].repeat(epoch_frame_delta + 1, axis=0)
                self.cam_index.append(cam_index)
                begin_frame = epoch * epoch_frame_delta  # 每个epoch起始和终止
                if epoch == 0:  # 只需要求第一个epoch
                    pose_epoch_begin = poses_test[begin_frame]
                    pose_epoch_begin = find_nearest_cam_pose(pose_epoch_begin, cam_begin, self.meta,
                                                             self.block_split_info,
                                                             self.block_range)  # 寻找在一段block范围内某cam_idx中离该点最近的pose
                end_frame = (epoch + 1) * epoch_frame_delta
                pose_epoch_end = poses_test[end_frame]
                pose_epoch_end = find_nearest_cam_pose(pose_epoch_end, cam_end, self.meta, self.block_split_info,
                                                       self.block_range)

                pose_begin_rot = pose_epoch_begin[:, :3]
                pose_end_rot = pose_epoch_end[:, :3]
                Rot = Rotation.from_matrix([pose_begin_rot, pose_end_rot])
                key_times = [0, 1]
                slerp = Slerp(key_times, Rot)
                times = np.linspace(0, 1, epoch_frame_delta)
                pose_rot_interp = slerp(times).as_matrix()  # 旋转
                # 对位姿进行插值
                for frame in range(epoch_frame_delta):
                    global_frame = epoch * epoch_frame_delta + frame
                    poses_test[global_frame, :, :3] = pose_rot_interp[frame]
                pose_epoch_begin = pose_epoch_end

            self.cam_index = np.concatenate(self.cam_index, axis=0)
            self.poses_test = poses_test

            self.load_pose = True if kwargs["visual_ply"] else False

    def read_meta(self, split, **kwargs):
        if split == "train":
            self.rgbs = []  # rgb
            self.rays_dir = []
            self.rays_origin = []
            self.exposures = []
            self.ts = []
            self.poses = []
            for img_element in tqdm(self.block_split_info[self.block_index]['elements']):
                img_name = img_element[0]
                img_info = self.meta[img_name]

                c2w = torch.FloatTensor(img_info['transform_matrix'])
                self.poses.append(c2w)  # pose用于occupy grid初始化

                height, width = img_info["height"], img_info["width"]

                # K用于occupy grid进行初始化
                K = np.zeros((3, 3), dtype=np.float32)
                K[0, 0] = img_info['intrinsics'][0]  # fx
                K[1, 1] = img_info['intrinsics'][1]  # fy
                K[0, 2] = width / 2  # cx
                K[1, 2] = height / 2  # cy
                K[2, 2] = 1
                self.K = torch.FloatTensor(K)

                self.img_wh = (img_info['width'], img_info['height'])

                img = Image.open(os.path.join(self.root_dir, f'images', img_info['image_name'])).convert('RGB')
                # 读取img的mask
                if self.use_segmentation:
                    # 筛选mask
                    seg_out = \
                        self.predictor(cv2.imread(os.path.join(self.root_dir, f'images', img_info['image_name'])))[
                            "instances"]
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

                exposure = torch.tensor(img_info["equivalent_exposure"])
                self.exposures.append(exposure * torch.ones_like(img[:, :1]))

                ts = img_element[1]
                self.ts.append(ts * torch.ones_like(img[:, :1]))

                # 选择使用pose还是rays
                if self.use_rays:
                    rays_d = torch.tensor(
                        np.load(os.path.join(self.root_dir, f"images", f"{img_name}_ray_dirs.npy")))
                    rays_o = torch.tensor(
                        np.load(os.path.join(self.root_dir, f"images", f"{img_name}_ray_origins.npy")))
                else:
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
            self.rays_origin = self.rays_origin - self.translate

            self.rays_dir = torch.cat(self.rays_dir, 0)

            # 优化poses，poses主要是用于初始化occupy grid
            for idx in range(len(self.poses)):
                self.poses[idx][:, 3] -= self.translate
            self.poses = torch.stack(self.poses, 0)

            if self.scale_pose:  # 将位姿归一化到一个单位球内
                # 找到距离原点最远处
                self.pose_scale = self.rays_origin.norm(dim=-1).max() * 1.5
                self.rays_origin /= self.pose_scale

            self.exposures = torch.cat(self.exposures, 0)
            self.ts = torch.cat(self.ts, 0)
            print(f"Total {self.N_image} images and {len(self.rgbs)} rays...")

            del self.predictor
            torch.cuda.empty_cache()

        elif split == "val":
            # 这里只是为了先求val的seg_mask
            if self.use_segmentation:
                self.img_mask = {}
                for img_element in tqdm(self.block_split_info[self.block_index]['elements']):
                    img_name = img_element[0]
                    img_info = self.meta[img_name]

                    seg_out = self.predictor(
                        cv2.imread(os.path.join(self.root_dir, f'images', img_info['image_name'])))["instances"]
                    mask = []
                    for pred_class, pred_mask in zip(seg_out.pred_classes, seg_out.pred_masks):
                        if pred_class in mask_label:
                            mask.append(pred_mask.cpu())
                    mask.append(torch.zeros([img_info["height"], img_info["width"]]) == 1)
                    img_mask = ~torch.any(torch.stack(mask, dim=0), dim=0).cpu()
                    '''
                    img_mask = ~torch.any(self.predictor(
                        cv2.imread(os.path.join(self.root_dir, f'images', img_info['image_name']))
                    )["instances"].pred_masks, dim=0).cpu()
                    '''
                    self.img_mask[img_name] = img_mask

                del self.predictor
                torch.cuda.empty_cache()

    def __len__(self):
        if self.split.startswith('train'):
            return len(self.rgbs)  # 每个epoch跑1000次，每次选8192张图，每张图选8192条光线
        elif self.split == "eval_single_block" or "compose_blocks" in self.split:
            return len(self.poses_test)
        elif self.split == "eval_val":
            return len(self.val_images.keys())
        return len(self.block_split_info[self.block_index]["elements"])

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            sample = {'rays_o': self.rays_origin[idx],
                      "rays_d": self.rays_dir[idx],
                      "rgb": self.rgbs[idx],
                      "ts": self.ts[idx].long(),
                      "exposure": self.exposures[idx]}

        elif self.split == "eval_single_block":  # 用来生成轨迹
            # 给image_begin
            img_name = self.eval_begin  # self.block_split_info[self.block_index]['elements'][idx][0]
            img_info = self.meta[img_name]
            print("The image to validate is {0}".format(img_info["image_name"]))

            c2w = torch.FloatTensor(self.poses_test[idx])
            cam_idx = self.cam_index[idx]
            height, width = img_info["height"] // self.downsample, img_info["width"] // self.downsample
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0] = img_info['intrinsics'][0] // self.downsample  # fx
            K[1, 1] = img_info['intrinsics'][1] // self.downsample  # fy
            K[0, 2] = width / 2  # cx
            K[1, 2] = height / 2  # cy
            K[2, 2] = 1

            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)

            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3) - self.translate
            '''
            rays_d = torch.tensor(
                np.load(os.path.join(self.root_dir, f"images", f"{img_name}_ray_dirs.npy")))
            rays_o = torch.tensor(
                np.load(os.path.join(self.root_dir, f"images", f"{img_name}_ray_origins.npy")))
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3) - self.val_translate
            '''

            # 还是要找离得最近的作为exposure
            # 选择最近的element作为参考exposure
            if self.filter_closest_element:
                nearest_info = get_nearest_info(c2w, self.meta, self.block_split_info[self.block_index]["elements"],
                                                cam_idx)
                exposure = torch.tensor(nearest_info["equivalent_exposure"]) * torch.ones_like(rays_o[:, :1])
                # index和block_index有关
                for element in self.block_split_info[self.block_index]["elements"]:
                    if element[0] == nearest_info["image_name"].split(".")[0]:
                        ts = torch.tensor(element[1]) * torch.ones_like(rays_o[:, :1])
                        break
            else:
                exposure = torch.tensor(img_info["equivalent_exposure"]) * torch.ones_like(rays_o[:, :1])
                # index和block_index有关
                for element in self.block_split_info[self.block_index]["elements"]:
                    if element[0] == img_name:
                        ts = torch.tensor(element[1]) * torch.ones_like(rays_o[:, :1])
                        break

            if self.scale_pose:
                rays_o /= self.pose_scale_ratio

            img_wh = [width, height]

            sample = {'rays_o': rays_o,
                      "rays_d": rays_d,
                      "exposure": exposure,
                      "ts": ts.long(),
                      "img_wh": img_wh,
                      "img_name": img_name}

        elif self.split == "compose_blocks":  # 用来生成轨迹
            # 给image_begin
            c2w = torch.FloatTensor(self.poses_test[idx])
            img_info = self.first_element
            height, width = img_info["height"] // self.downsample, img_info["width"] // self.downsample
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0] = img_info['intrinsics'][0] // self.downsample  # fx
            K[1, 1] = img_info['intrinsics'][1] // self.downsample  # fy
            K[0, 2] = width / 2  # cx
            K[1, 2] = height / 2  # cy
            K[2, 2] = 1

            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)

            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)  # - self.translate

            # 判断当前pose属于哪些block
            block_start, block_end = self.img_trajectory  # 36，40 -> 36,37,38,39,40
            block_indexes = np.arange(int(block_start.split("_")[-1]), int(block_end.split("_")[-1]) + 1)
            block_involve = []
            for block in block_indexes:
                block_info = self.block_split_info[f"block_{block}"]
                block_center = torch.tensor(block_info["centroid"][1])
                if (c2w[:, 3] - block_center).norm() < self.radius:
                    # 计算权重，距离的倒数
                    weight = DistanceWeight(c2w[:, 3], block_center, p=1)
                    block_involve.append([block, weight])

            # 包含在block_involve

            rays_o = rays_o + torch.tensor([0, 0, 0.1])
            sample = {}
            for block in block_involve:
                # 首先找出translate和center
                weight = block[1]
                block = f"block_{block[0]}"
                center_translate = self.block_center_translate[block]  # center & translate
                translate = torch.tensor(center_translate["translate"])
                rays_o_temp = rays_o - translate
                # 在当前block中找到最接近的expsoure和ts
                nearest_info = get_nearest_info(c2w, self.meta, self.block_split_info[block]["elements"],
                                                [self.cam_index])
                exposure = torch.tensor(nearest_info["equivalent_exposure"]) * torch.ones_like(rays_o[:, :1])
                # index和block_index有关
                for element in self.block_split_info[block]["elements"]:
                    if element[0] == nearest_info["image_name"].split(".")[0]:
                        ts = torch.tensor(element[1]) * torch.ones_like(rays_o[:, :1])
                        break

                img_wh = [width, height]
                sample[block] = {'rays_o': rays_o_temp,
                                 "rays_d": rays_d,
                                 "exposure": exposure,
                                 "ts": ts.long(),
                                 "weight": weight,
                                 "img_wh": img_wh}

        elif self.split == "compose_blocks_diy_trajectory":  # 用来生成轨迹
            # 给image_begin
            c2w = torch.FloatTensor(self.poses_test[idx])
            img_info = self.first_element
            height, width = img_info["height"] // self.downsample, img_info["width"] // self.downsample
            K = np.zeros((3, 3), dtype=np.float32)
            K[0, 0] = img_info['intrinsics'][0] // self.downsample  # fx
            K[1, 1] = img_info['intrinsics'][1] // self.downsample  # fy
            K[0, 2] = width / 2  # cx
            K[1, 2] = height / 2  # cy
            K[2, 2] = 1

            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)

            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)  # - self.translate

            # 判断当前pose属于哪些block
            block_start, block_end = self.img_trajectory  # 36，40 -> 36,37,38,39,40
            block_indexes = np.arange(int(block_start.split("_")[-1]), int(block_end.split("_")[-1]) + 1)
            block_involve = []
            for block in block_indexes:
                block_info = self.block_split_info[f"block_{block}"]
                block_center = torch.tensor(block_info["centroid"][1])
                if (c2w[:, 3] - block_center).norm() < self.radius:
                    # 计算权重，距离的倒数
                    weight = DistanceWeight(c2w[:, 3], block_center, p=1)
                    block_involve.append([block, weight])

            # 包含在block_involve
            sample = {}
            for block in block_involve:
                # 首先找出translate和center
                weight = block[1]
                block = f"block_{block[0]}"
                center_translate = self.block_center_translate[block]  # center & translate
                translate = torch.tensor(center_translate["translate"])
                rays_o_temp = rays_o - translate
                # 在当前block中找到最接近的expsoure和ts
                nearest_info = get_nearest_info(c2w, self.meta, self.block_split_info[block]["elements"],
                                                self.cam_index[idx])
                exposure = torch.tensor(nearest_info["equivalent_exposure"]) * torch.ones_like(rays_o[:, :1])
                # index和block_index有关
                for element in self.block_split_info[block]["elements"]:
                    if element[0] == nearest_info["image_name"].split(".")[0]:
                        ts = torch.tensor(element[1]) * torch.ones_like(rays_o[:, :1])
                        break

                img_wh = [width, height]
                sample[block] = {'rays_o': rays_o_temp,
                                 "rays_d": rays_d,
                                 "exposure": exposure,
                                 "ts": ts.long(),
                                 "weight": weight,
                                 "img_wh": img_wh}
                if self.load_pose:
                    sample[block]["c2w"] = c2w.numpy()
                    sample[block]["intrinsics"] = [height, width, K[0, 0], K[1, 1]]

        elif self.split == "eval_val":
            val_image = list(self.val_images.keys())[idx]
            val_image_blocks = self.val_images[val_image]
            # 首先提取rays_d,rays_o,exposure还有image
            val_info = self.meta[val_image]

            c2w = torch.FloatTensor(val_info['transform_matrix'])

            img = Image.open(os.path.join(self.root_dir, f'images', val_info['image_name'])).convert('RGB')
            img = self.transform(img)  # (3,h,w)
            img = img.view(3, -1).permute(1, 0)

            if self.use_segmentation:
                img_mask = self.img_mask[val_image]

            width, height = val_info["width"], val_info["height"]
            if self.use_rays:
                rays_d = torch.tensor(
                    np.load(os.path.join(self.root_dir, f"images", f"{val_image}_ray_dirs.npy")))
                rays_o = torch.tensor(
                    np.load(os.path.join(self.root_dir, f"images", f"{val_image}_ray_origins.npy")))
            else:
                K = np.zeros((3, 3), dtype=np.float32)
                K[0, 0] = val_info['intrinsics'][0]  # fx
                K[1, 1] = val_info['intrinsics'][1]  # fy
                K[0, 2] = width / 2  # cx
                K[1, 2] = height / 2  # cy
                K[2, 2] = 1
                directions = get_ray_directions(height, width, K)
                rays_o, rays_d = get_rays(directions, c2w)
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)

            exposure = torch.tensor(val_info["equivalent_exposure"]) * torch.ones_like(rays_o[:, :1])

            # 根据block找ts
            sample = {}
            for block in val_image_blocks:
                block = f"block_{block}"
                center_translate = self.block_center_translate[block]  # center & translate
                translate = torch.tensor(center_translate["translate"])
                rays_o_temp = rays_o - translate

                for element in self.block_split_info[block]["elements"]:
                    if element[0] == val_image:
                        ts = torch.tensor(element[1]) * torch.ones_like(rays_o[:, :1])
                        break
                img_wh = [width, height]

                # 计算weight
                block_info = self.block_split_train_info[block]
                # 计算weight要用train的split_block
                block_center = torch.tensor(block_info["centroid"][1])
                weight = DistanceWeight(c2w[:, 3], block_center, p=1)
                sample[block] = {'rays_o': rays_o_temp,
                                 "rays_d": rays_d,
                                 "rgb": img,
                                 "exposure": exposure,
                                 "ts": ts.long(),
                                 "img_wh": img_wh,
                                 "img_name": val_image,
                                 "weight": weight}
                if self.use_segmentation:
                    sample[block]["mask"] = img_mask



        elif self.split == "val":  # 可以选择用pose来生成rays
            img_name = self.block_split_info[self.block_index]['elements'][idx][0]
            img_info = self.meta[img_name]
            print("The image to validate is {0}".format(img_info["image_name"]))
            img = Image.open(os.path.join(self.root_dir, f'images', img_info['image_name'])).convert('RGB')

            if self.use_segmentation:
                img_mask = self.img_mask[img_name]

            img = self.transform(img)  # (3,h,w)
            img = img.view(3, -1).permute(1, 0)

            exposure = torch.tensor(img_info["equivalent_exposure"]) * torch.ones_like(img[:, :1])

            ts = torch.tensor(self.block_split_info[self.block_index]['elements'][idx][1]) * torch.ones_like(img[:, :1])

            rays_d = torch.tensor(
                np.load(os.path.join(self.root_dir, f"images", f"{img_name}_ray_dirs.npy")))
            rays_o = torch.tensor(
                np.load(os.path.join(self.root_dir, f"images", f"{img_name}_ray_origins.npy")))
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3) - self.translate

            if self.scale_pose:
                rays_o /= self.pose_scale_ratio

            img_wh = [img_info["width"], img_info["height"]]

            sample = {'rays_o': rays_o,
                      "rays_d": rays_d,
                      "rgb": img,
                      "exposure": exposure,
                      "ts": ts.long(),
                      "img_wh": img_wh,
                      "img_name": img_name}
            if self.use_segmentation:
                sample["mask"] = img_mask


        elif self.split == "test":
            # 返回img_wh,K
            img_name = self.block_split_info[self.block_index]['elements'][idx][0]
            img_info = self.meta[img_name]

            height = img_info["height"]
            width = img_info["width"]

            height_scale = int(height * self.downsample)
            width_scale = int(width * self.downsample)

            K = np.zeros((3, 3), dtype=np.float32)
            img_w, img_h = width, height
            K[0, 0] = img_info['intrinsics'][0] * self.downsample  # fx
            K[1, 1] = img_info['intrinsics'][1] * self.downsample  # fy
            K[0, 2] = width_scale / 2  # cx
            K[1, 2] = height_scale / 2  # cy
            K[2, 2] = 1

            sample = {'img_wh': [width_scale, height_scale],
                      "K": K}

        elif self.split == "eval":
            img_name = self.block_split_info[self.block_index]['elements'][idx][0]
            img_info = self.meta[img_name]
            print("The image to eval is {0}".format(img_info["image_name"]))

            exposure = torch.tensor(img_info["equivalent_exposure"])

            ts = torch.tensor(self.block_split_info[self.block_index]['elements'][idx][1])

            # rays_d使用pose转换

            width = img_info['width'] // self.downsample
            height = img_info['height'] // self.downsample

            rays_d = torch.tensor(np.load(os.path.join(self.root_dir, "images", f"{img_name}_ray_dirs.npy")))
            rays_o = torch.tensor(np.load(os.path.join(self.root_dir, "images", f"{img_name}_ray_origins.npy")))
            if self.downsample != 1:
                rays_d = transforms.functional.resize(rays_d.permute(2, 0, 1), [height, width]).permute(1, 2, 0)
                rays_o = transforms.functional.resize(rays_o.permute(2, 0, 1), [height, width]).permute(1, 2, 0)
            '''
            #   利用相机内参求得directions
            K = np.zeros((3, 3), dtype=np.float32)
            img_w, img_h = int(img_info['width']), int(img_info['height'])
            img_w_, img_h_ = img_w // self.downsample, img_h // self.downsample
            K[0, 0] = img_info['intrinsics'][0] * img_w_ / img_w  # fx
            K[1, 1] = img_info['intrinsics'][1] * img_h_ / img_h  # fy
            K[0, 2] = img_info['width'] / 2 * img_w_ / img_w  # cx
            K[1, 2] = img_info['height'] / 2 * img_h_ / img_h  # cy
            K[2, 2] = 1

            directions = get_ray_directions(height, width, K)
            c2w = torch.FloatTensor(img_info['transform_matrix'])
            rays_o, rays_d = get_rays(directions, c2w)
            '''
            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)

            img_wh = [width, height]

            sample = {'rays_o': rays_o,
                      "rays_d": rays_d,
                      "exposure": exposure.unsqueeze(0),
                      "ts": ts.unsqueeze(0),
                      "img_wh": img_wh}
        return sample
