import numpy as np
import matplotlib.pyplot as plt
import cv2
# 用于标记深度信息

img_path = '/home/yangzesong/Projects/NeRF/datasets/virtual_Kitti/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg'
img = cv2.imread(img_path)
depth_img_path = '/home/yangzesong/Projects/NeRF/datasets/virtual_Kitti/vkitti_2.0.3_depth/Scene01/15-deg-left/frames/depth/Camera_0/depth_00000.png'
depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(depth_img[y][x])
        xy = f"{depth_img[y][x]}"
        # xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while (1):
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
