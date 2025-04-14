import os
import json
import shutil
from tqdm import tqdm

file = "./loco/rgb/loco-all-v1.json"
file_new = "./annotations.json"
data_path = "./dataset/"

with open(file) as json_file:
    data = json.load(json_file)

images = data['images']
categories = data['categories']
annotations = data['annotations']

#%%
# name_set = set()
# for img in images:
#     img_name = img['path'].split('/')[-1]
#     if img_name in name_set:
#         print(img_name)
#     else:
#         name_set.add(img_name)

for i in tqdm(range(len(images))):
    old_path = "." + images[i]['path']
    new_path = data_path + images[i]['path'].split('/')[-1]
    images[i]['path'] = new_path
    shutil.move(old_path, new_path)

for item in os.listdir(data_path):
    if os.path.isdir(data_path + item):
        shutil.rmtree(data_path + item)

#%%
# images = images[:100]
# annotations = annotations[:100]

categories_new = []
images_new = []
annotations_new = []

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
            if annotations[k]['category_id'] == 7:
                annotations_new.append(annotations[k])
                cnt += 1
        if cnt > 0:
            images_new.append(image_dict[annotations[i]['image_id']])
        else:
            os.remove(image_dict[annotations[i]['image_id']]['path'])
        i = j

#%%
data_new = {'images': images_new, 'categories': categories_new, 'annotations': annotations_new}
with open(file_new, 'w') as json_file:
    json.dump(data_new, json_file, indent=4)
