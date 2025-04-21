import os
import shutil
from tqdm import tqdm

data_path = "../data_preprocess/dataset_origin/"
output_path = "./dataset_origin/"
os.makedirs(output_path)

file_count = sum(len(files) for _, _, files in os.walk(data_path))
with tqdm(total=file_count) as pbar:
    for root, dirs, files in os.walk(data_path):
        for file in files:
            old_path = "/".join([root, file])
            new_path = "/".join([output_path, file])
            shutil.copy(old_path, new_path)
            pbar.update(1)
