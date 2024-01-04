import json
import open3d as o3d
import numpy as np
import imageio
import os
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from collections import defaultdict

cams = []
imgs = []
num_frames = 15


def count_cam_idx(cam_dict):
    indexs = []
    for cam in cam_dict:
        index = cam_dict[cam]["cam_idx"]
        if index not in indexs:
            indexs.append(index)
    return indexs


    # pose_scale用于放大位姿
def get_camera_frustum(img_size, K, C2W, frustum_length, color, scale_pose=False):
    # [w,h]  [4,4]  [3,4]
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)  # 光心到图像左右两边的角度
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)  # 光心到图像上下两边的角度
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))  # 归一化平面
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))
    # 不就是W,H的一半

    pose = np.eye(4)
    pose[:3] = C2W
    C2W = pose

    # 调整尺度
    '''
    half_w *= 5e-3
    half_h *= 5e-3
    frustum_length *= 5e-3
    '''

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               [-half_w, -half_h, frustum_length],  # 左上角，注意x朝右，y朝上
                               [half_w, -half_h, frustum_length],  # 右上角
                               [half_w, half_h, frustum_length],  # bottom-right image corner
                               [-half_w, half_h, frustum_length],
                               # 坐标轴
                               [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]  # x轴,y轴,z轴
                               ])  # bottom-left image corner
    if scale_pose:
        frustum_points *= 1e-1

    frustum_lines = np.array([[0, i] for i in range(1, 5)] +
                             [[i, (i + 1)] for i in range(1, 4)] +
                             [[4, 1], [0, 5], [0, 6], [0, 7]])
    # 平铺
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    frustum_colors[-1] = np.array([0, 0, 1])  # Z 蓝色
    frustum_colors[-2] = np.array([0, 1, 0])  # y 绿色
    frustum_colors[-3] = np.array([1, 0, 0])  # x 红色

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    frustum_points = np.dot(
        np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))),  # 齐次坐标
        C2W.T)  # 8，4
    # 归一化矩阵乘以C2W.T
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 8, 3))  # 5 vertices per frustum 总共有5个点
    merged_lines = np.zeros((N * 11, 2))  # 8 lines per frustum # 总共有8条线
    merged_colors = np.zeros((N * 11, 3))  # each line gets a color # 每条线一个颜色

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 8:(i + 1) * 8, :] = frustum_points
        merged_lines[i * 11:(i + 1) * 11, :] = frustum_lines + i * 8
        merged_colors[i * 11:(i + 1) * 11, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visual_cameras(cam_dicts, data_type="colmap", scale_pose=True, ply=None,dataset_cameras=None, bbox=None, mesh=None):
    # 设置起点在路径中心
    things_to_draw = [ply] if ply is not None else []

    if bbox is not None: things_to_draw.append(bbox)
    if mesh is not None: things_to_draw.append(mesh)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))
    things_to_draw.append(sphere)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw.append(coord_frame)

    frustums = []

    cam_idxs = count_cam_idx(cam_dicts)
    colors = {}  # 每个index显示一种颜色
    '''
    color_begin = [1, 0, 0]
    color_end = [0, 1, 1]
    color_delta = (np.array(color_end) - np.array(color_begin)) / len(cam_idxs)
    for i,cam in enumerate(cam_idxs):
        colors[cam] = list(
            np.array(color_begin)+i*color_delta
        )
    '''
    for i, cam in enumerate(cam_idxs):
        colors[cam] = list(
            np.random.random(3)
        )

    for cam_dict in cam_dicts:
        cam_info = cam_dicts[cam_dict]
        cam_show_idxs = cam_idxs
        if cam_info["cam_idx"] in cam_show_idxs:  # 筛选index进行展示
            index = cam_info["cam_idx"]

            K = np.array([
                [cam_info["intrinsics"][2], 0, cam_info["intrinsics"][1] / 2, 0],  # 宽
                [0, cam_info["intrinsics"][2], cam_info["intrinsics"][0] / 2, 0],  # 高
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            C2W = cam_info["c2w"]

            # NeRF坐标系和open3D不一样，需要转换
            if data_type != "colmap":
                C2W[:, 1:3] *= -1

            # C2W[:, 1:3] *= -1

            img_size = [cam_info["intrinsics"][1], cam_info["intrinsics"][0]]  # 宽、高
            frustum_length = cam_info["intrinsics"][2] / cam_info["intrinsics"][1]

            frustums.append(get_camera_frustum(img_size, K, C2W, frustum_length, colors[index], scale_pose=scale_pose))

    cameras = frustums2lineset(frustums)
    # o3d.visualization.draw_geometries([cameras])
    things_to_draw.append(cameras)
    if dataset_cameras != None:
        things_to_draw.append(dataset_cameras)
    o3d.visualization.draw_geometries(things_to_draw)


def capture_poses(vis):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    print(f"current pose:{param.extrinsic}")
    cams.append(param.extrinsic)

    img = vis.capture_screen_float_buffer()
    img_np = np.asarray(img)
    imgs.append(img_np)


def get_render_trajectory(ply, transform_json=None, screen_size=None):
    # app = o3d.visualization.gui.Application.instance
    # app.initialize()

    # vis = o3d.visualization.O3DVisualizer("trajectory recorder", 1024, 768)
    # vis.show_settings = True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if screen_size != None:
        vis.create_window(window_name='pcd', width=int(screen_size[0]), height=int(screen_size[1]))
    else:
        vis.create_window(window_name='pcd', width=800, height=600)
    
    # vis.add_geometry("Points",pcd)
    vis.add_geometry(ply)
    vis.register_key_callback(ord('C'), capture_poses)

    colors = {}  # 每个index显示一种颜色
    for i, cam in enumerate(range(40)):
        colors[cam] = list(
            np.random.random(3)
        )

    frustums = []

    if transform_json != None:
        for cam_dict in transform_json:
            cam_info = transform_json[cam_dict]
            index = cam_info["cam_idx"]
            K = np.array([
                [cam_info["intrinsics"][0], 0, cam_info["width"]/2 , 0],  # 宽
                [0, cam_info["intrinsics"][1], cam_info["height"]/2 , 0],  # 高
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            C2W = np.array(cam_info["transform_matrix"])
            C2W [..., 3] *= cam_info["scale"]
            # NeRF坐标系和open3D不一样，需要转换
            # C2W[:, 1:3] *= -1

            img_size = [cam_info["width"], cam_info["height"]]  # 宽、高
            frustum_length = cam_info["intrinsics"][0] / cam_info["width"]
            frustum = get_camera_frustum(img_size, K, C2W, frustum_length, colors[index], scale_pose=True)
            frustums.append(frustum)
            # widget3d = vis.SceneWidget()
            # text=torch.FloatTensor(frustum[0][0].reshape(3,1))

            # vis.add_3d_label(text.numpy(), index)

        cameras = frustums2lineset(frustums)
    if screen_size != None:
        ctr = vis.get_view_control()
        ctr.set_constant_z_near(0.1)
        ctr.set_constant_z_far(1000)
        param = ctr.convert_to_pinhole_camera_parameters()
        window_scale = 2
        param.intrinsic.width = int(screen_size[0]) // window_scale
        param.intrinsic.height = int(screen_size[1]) // window_scale
        ctr.convert_from_pinhole_camera_parameters(param)

    vis.add_geometry(cameras)
    vis.run()
    # app.add_window(vis)
    # app.run()
    return cameras


def poses_trajectory_interp(num_frames = 15):
    trajectorys = []
    for idx in range(len(cams) - 1):
        c2w_0 = np.linalg.inv(cams[idx])
        c2w_1 = np.linalg.inv(cams[idx + 1])

        c2w_rot_0 = c2w_0[:3, :3]
        c2w_rot_1 = c2w_1[:3, :3]
        c2w_trans_0 = c2w_0[:3, 3]
        c2w_trans_1 = c2w_1[:3, 3]

        Rot = Rotation.from_matrix([c2w_rot_0, c2w_rot_1])
        key_times = [0, 1]
        slerp = Slerp(key_times, Rot)
        times = np.linspace(0, 1, num_frames + 1)
        pose_interp = slerp(times).as_matrix()

        for i in range(num_frames + 1):
            ratio = np.sin(((i / num_frames) - 0.5) * np.pi) * 0.5 + 0.5
            val_c2w = np.diag([1.0, 1.0, 1.0, 1.0])
            val_c2w[:3, :3] = pose_interp[i]
            val_c2w[:3, 3] = (1.0 - ratio) * c2w_trans_0 + ratio * c2w_trans_1
            trajectorys.append(val_c2w)

    return trajectorys


def visual_trajectory(trajectory, pcd, dataset_cameras = None, phone_screen = None):
    vis_meta = defaultdict(dict)

    for frame in range(len(trajectory)):
        vis_meta[frame]["c2w"] = trajectory[frame][:3]
        if phone_screen != None:
            vis_meta[frame]["intrinsics"] = [phone_screen[1]*2,
                                         phone_screen[0]*2, 
                                         3847.55356130808]
        else:
            vis_meta[frame]["intrinsics"] = [4624.0,
                                            2080.0, 
                                            3847.55356130808]
        """[1080,
        1920,
        1165.144
        ]"""
        vis_meta[frame]["cam_idx"] = frame % num_frames

    visual_cameras(vis_meta, data_type="colmap", scale_pose=True, ply=pcd, dataset_cameras=dataset_cameras, bbox=None, mesh=None)


if __name__ == "__main__":
    record =False
    pcd_path = "/home/azx/dataset/zju_gate_front_crosswise/zju_gate_front_crosswise.ply"
    pcd = o3d.io.read_point_cloud(pcd_path)
    # 获取pose
    # 按大写C记录位姿，按Q退出
    transform_info = json.load(open('/home/azx/dataset/zju_gate_front_crosswise/json/transform_info.json'))
    phone_screen = [transform_info["1"]["intrinsics"][2],transform_info["1"]["intrinsics"][3]]
    original_cameras = get_render_trajectory(pcd, transform_json=transform_info,screen_size=phone_screen)
    # imageio.mimwrite("tmp.mp4", imgs, fps=25, quality=8)
    if record :
        # 插值
        trajectory = poses_trajectory_interp()
        np.save(os.path.join(os.path.dirname(pcd_path),"trajectory.npy"), trajectory)
    else:
        trajectory = np.load('interpolate_trajectory3.npy','r')
    
    # 可视化位姿
    visual_trajectory(trajectory, pcd, dataset_cameras = original_cameras, phone_screen = phone_screen)
