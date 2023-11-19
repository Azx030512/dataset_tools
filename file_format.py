import glob
import os

path = '/home/azx/dataset/seek_truth_lecture_hall_vertical/images/*'
for file_abs, i in zip(glob.glob(path),range(1,200)):
    os.rename(file_abs,os.path.join(os.path.dirname(file_abs),str(i)+'.jpg'))
