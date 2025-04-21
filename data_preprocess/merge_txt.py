import os
from tqdm import tqdm

path_1 = "./labels_pallet/"
path_0 = "../annotation/res_ground/labels/"
path = './labels/'

os.makedirs(path)
labels_set = set(os.listdir(path_1))
labels_1 = os.listdir(path_1)
labels_0 = os.listdir(path_0)

for i in tqdm(range(len(labels_0))):
    labels = []
    with open(path_0 + labels_0[i], "r") as file:
        for line in file:
            cls, x, y, w, h = line.split(" ")
            x, y, w, h = float(x), float(y), float(w), float(h)
            line_new = " ".join([cls, f"{x:8.6f}", f"{y:8.6f}", f"{w:8.6f}", f"{h:8.6f}"]) + "\n"
            labels.append(line_new)
    
    if labels_0[i] in labels_set:
        with open(path_1 + labels_0[i], "r") as file:
            for line in file:
                labels.append(line)
    
    with open(path + labels_0[i], "w") as file:
        for line in labels:
            file.write(line)
