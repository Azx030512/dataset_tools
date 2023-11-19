import numpy as np
import matplotlib.pyplot as plt
import cv2


# 用于标记深度信息

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


img_path = '/home/yangzesong/Projects/NeRF/datasets/4B-sky/images/DJI_0859.jpg'
img = cv2.imread(img_path)
depth_img_path = '/home/yangzesong/Projects/NeRF/datasets/4B-sky/depth_maps/DJI_0859.jpg.geometric.bin'
depth_img = read_depth(depth_img_path)

min_depth, max_depth = np.percentile(depth_img, [5, 95])
depth_img[depth_img < min_depth] = min_depth
depth_img[depth_img > max_depth] = max_depth


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(depth_img[y][x])
        xy = f"{depth_img[y][x]}"
        # xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=1)
        cv2.imshow("image", img)


import pylab as plt

plt.figure()
plt.imshow(depth_img)
plt.title("depth map")
plt.show()

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while (1):
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
