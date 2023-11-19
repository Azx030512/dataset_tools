import os
import argparse
import shutil
# 用于提取kitti数据集用于colmap


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/yangzesong/Projects/NeRF/datasets/virtual_Kitti',
                        help='root directory of dataset')
    parser.add_argument('--scene', type=str,
                        default='Scene01',
                        help='The scene for choosing')

    return vars(parser.parse_args())


if __name__ == "__main__":
    hparams = get_opts()
    root_dir = hparams["root_dir"]
    os.makedirs(os.path.join(root_dir, "json"), exist_ok=True)
    save_path = os.path.join(root_dir, "json/transform_info.json")

    RGB_root_dir = os.path.join(root_dir, "vkitti_2.0.3_rgb")
    Depth_root_dir = os.path.join(root_dir, "vkitti_2.0.3_depth")
    Text_root_dir = os.path.join(root_dir, "vkitti_2.0.3_textgt")
    Seg_root_dir = os.path.join(root_dir, "vkitti_2.0.3_classSegmentation")

    View_chosed_dir = ["15-deg-left", "30-deg-left", "15-deg-right", "30-deg-right"]

    # image_name
    # idx
    # cam_idx
    # width
    # height
    # intrins
    # transform
    img_info = []
    for idx, view in enumerate(View_chosed_dir):
        rgb_folder_path = os.path.join(RGB_root_dir, hparams["scene"], view, "frames/rgb/Camera_0")  # 默认camera_0

        count = 0
        for rgb_file in sorted(os.listdir(rgb_folder_path)):
            if count == 35:
                break
            # img_info.append(os.path.join(rgb_folder_path, rgb_file))
            old_file_path = os.path.join(rgb_folder_path, rgb_file)
            new_file_path = os.path.join(root_dir, "temp_image", "{0}_{1}".format(view, rgb_file))
            shutil.copy(old_file_path, new_file_path)
            count += 1

