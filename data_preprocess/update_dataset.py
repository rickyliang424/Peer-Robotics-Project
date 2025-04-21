import os
import json
import shutil
from tqdm import tqdm

area_threshold = 0.005
file = "./loco/rgb/loco-all-v1.json"
file_new = "./annotations.json"
name_txt_0 = "./name_list_0.txt"
name_txt_1 = "./name_list_1.txt"
data_path = "./dataset_origin/"
output_path = "./dataset/"
os.makedirs(output_path)

with open(file) as json_file:
    data = json.load(json_file)

images = data['images']
categories = data['categories']
annotations = data['annotations']
images_new = []
categories_new = []
annotations_new = []
name_list_0 = []
name_list_1 = []

#%%
name_set = set()
for img in images:
    img_name = img['file_name']
    if img_name in name_set:
        print(f"Repeated name: {img_name}")
    else:
        name_set.add(img_name)

file_count = sum(len(files) for _, _, files in os.walk(data_path))
with tqdm(total=file_count) as pbar:
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file not in name_set:
                name_list_0.append(file[:-4])
            old_path = "/".join([root, file])
            new_path = "/".join([output_path, file])
            shutil.copy(old_path, new_path)
            pbar.update(1)

for i in range(len(images)):
    new_path = output_path + images[i]['path'].split('/')[-1]
    images[i]['path'] = new_path

for item in os.listdir(output_path):
    if os.path.isdir(output_path + item):
        shutil.rmtree(output_path + item)

#%%
for category in categories:
    if category['name'] == "pallet":
        categories_new.append(category)

image_dict = dict()
for image in images:
    image_dict[image['id']] = image

i, j = 0, 0
image_set = set()
while i < len(annotations):
    if j < len(annotations) and annotations[i]['image_id'] == annotations[j]['image_id']:
        j += 1
    else:
        if annotations[i]['image_id'] not in image_set:
            image_set.add(annotations[i]['image_id'])
        else:
            print(f"image_id re-appear at annotations[{j}]")
        cnt = 0
        for k in range(i,j):
            w = image_dict[annotations[i]['image_id']]['width']
            h = image_dict[annotations[i]['image_id']]['height']
            if annotations[k]['category_id'] == 7 and annotations[k]['area'] > area_threshold * w * h:
                annotations_new.append(annotations[k])
                cnt += 1
        images_new.append(image_dict[annotations[i]['image_id']])
        if cnt > 0:
            name_list_1.append(image_dict[annotations[i]['image_id']]['file_name'][:-4])
        else:
            name_list_0.append(image_dict[annotations[i]['image_id']]['file_name'][:-4])
        i = j

#%%
data_new = {'images': images_new, 'categories': categories_new, 'annotations': annotations_new}
with open(file_new, 'w') as json_file:
    json.dump(data_new, json_file, indent=4)

with open(name_txt_0, "w") as file:
    for name in name_list_0:
        file.write(name + "\n")

with open(name_txt_1, "w") as file:
    for name in name_list_1:
        file.write(name + "\n")
