import os
import numpy as np
import shutil

path = r"dataset/images"
new_path = r"dataset/images_tmp"
file_list = os.listdir(path)

files_tmp = np.random.choice(file_list, size=500)

for file in files_tmp:
    shutil.copy(f"{os.getcwd()}/{path}/{file}",
                f"{os.getcwd()}/{new_path}/{file}")
