import os
import json

file = "./annotations.json"
savepath = "./labels_pallet/"
os.makedirs(savepath)

with open(file) as json_file:
    data = json.load(json_file)

images = data['images']
annotations = data['annotations']

image_dict = dict()
for image in images:
    image_dict[image['id']] = image

i, j = 0, 0
while i < len(annotations):
    if j < len(annotations) and annotations[i]['image_id'] == annotations[j]['image_id']:
        j += 1
    else:
        with open(savepath + image_dict[annotations[i]['image_id']]['path'][10:-4] + ".txt", "w") as file:
            for k in range(i,j):
                width = image_dict[annotations[k]['image_id']]['width']
                height = image_dict[annotations[k]['image_id']]['height']
                [x_min, y_min, w, h] = annotations[k]['bbox']
                x_min = (x_min + w / 2) / width
                y_min = (y_min + h / 2) / height
                w = w / width
                h = h / height
                file.write(" ".join([str(1.0), f"{x_min:8.6f}", f"{y_min:8.6f}", f"{w:8.6f}", f"{h:8.6f}"]) + "\n")
        i = j
