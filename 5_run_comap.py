import os

'''
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export CUDA_HOME=/usr/local/cuda-11.3/

'''
def run_colmap(data_dir, mode = "all"):
    if not os.path.exists(os.path.join(data_dir, "database.db")):
        command = "colmap feature_extractor  " \
                  f"--database_path {data_dir}/database.db     " \
                  f"--image_path {data_dir}/images     " \
                  "--ImageReader.camera_model PINHOLE     " \
                  "--SiftExtraction.gpu_index 0 " 
                #   f"--ImageReader.camera_params {camera_params[0]},{camera_params[1]},{camera_params[2]},{camera_params[3]}"
        # "--ImageReader.single_camera 1" \
        os.system(command)

        command = f"colmap exhaustive_matcher \
        --database_path {data_dir}/database.db \
        --SiftMatching.gpu_index 0"
        os.system(command)
    if mode == 'all' or mode == 'construction':
        command = "colmap mapper " \
                f"--database_path {data_dir}/database.db " \
                f"--image_path {data_dir}/images " \
                f"--output_path {data_dir}/sparse "
        os.system(command)

    '''triangulator for dense point cloud reconstruction
       may use up to 20 threads, take caution!!!
    '''
    # command = "colmap point_triangulator " \
    #           f"--database_path {data_dir}/database.db " \
    #           f"--image_path {data_dir}/images " \
    #           f"--input_path {data_dir}/sparse/0 " \
    #           f"--output_path {data_dir}/sparse/0 " \
    #           f"--Mapper.tri_min_angle 10 --Mapper.tri_merge_max_reproj_error 1"
    # os.system(command)

    if mode == 'all' or mode == 'ply':

        command = f"colmap model_converter \
        --input_path {data_dir}/sparse/0/ \
        --output_path {data_dir}/{os.path.basename(data_dir)}.ply \
        --output_type PLY"
        os.system(command)

def make_director(path, exists_ok = True):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    dataset_path = "/mnt/nas_9/group/srtp2/datasets/pond"
    make_director(os.path.join(dataset_path,"sparse"))
    run_colmap(dataset_path, mode = 'all')
