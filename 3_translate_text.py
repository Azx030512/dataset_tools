import numpy as np

coordinate_dict = {
    "blender_to_opengl":{
        "4":np.array([[1,0,0,0],
                    [0,0,-1,0],
                    [0,1,0,0],
                    [0,0,0,1]]),
        "3":np.array([[1,0,0],
                    [0,0,-1],
                    [0,1,0]])
    } ,
    "opengl_to_blender": {
        "4":np.array([[1,0,0,0],
                    [0,0,1,0],
                    [0,-1,0,0],
                    [0,0,0,1]]),
        "3":np.array([[1,0,0],
                    [0,0,1],
                    [0,-1,0]])
    },
    "blender_to_opencv": {
        "4":np.array([[1,0,0,0],
                    [0,0,-1,0],
                    [0,1,0,0],
                    [0,0,0,1]]),
        "3":np.array([[1,0,0],
                    [0,-1,0],
                    [0,0,-1]])
    },
    "opencv_to_blender": {
        "4":np.array([[1,0,0,0],
                    [0,0,1,0],
                    [0,-1,0,0],
                    [0,0,0,1]]),
        "3":np.array([[1,0,0],
                    [0,0,1],
                    [0,-1,0]])
    }           
}

colmap = True
mode = "w2c" 
# mode = "c2w"


poses = []
export_file = "/home/azx/Desktop/dataset_tools/library2.txt"
with open(export_file, 'r+') as txtfile:
    row = txtfile.readline()
    while row :
        data = np.array([float(item) for item in row.split()])
        poses.append(data)
        row = txtfile.readline()
    pose_number = len(poses)//4
    poses = np.concatenate(poses,0).reshape(-1,4,4)

if mode == "c2w":
    poses = np.linalg.inv(poses[:])

if colmap:
    # poses[:,:3,:3] = np.matmul(coordinate_dict["opengl_to_colmap"]["3"],np.matmul(coordinate_dict["blender_to_opengl"]["3"],poses[:,:3,:3]))
    # poses[:,:3,:3] = np.matmul(poses[:,:3,:3],coordinate_dict["blender_to_opencv"]["3"])
    poses[...,:3,:3] = np.matmul(coordinate_dict["blender_to_opencv"]["3"],poses[...,:3,:3])
np.save("library2.npy", poses)
