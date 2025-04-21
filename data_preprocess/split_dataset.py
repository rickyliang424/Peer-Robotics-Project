import os
import random
import shutil
from tqdm import tqdm

train_percentage = 0.8
images_dir = "./dataset"
labels_dir = "./labels"
output_dir = "../dataset"

# Create output folders
for split in ['train', 'val']:
    os.makedirs(output_dir + "/" + "images" + "/" + split, exist_ok=True)
    os.makedirs(output_dir + "/" + "labels" + "/" + split, exist_ok=True)

# List all images (assume .jpg or change to .png if needed)
images = os.listdir(images_dir)
labels = os.listdir(labels_dir)
images.sort()
labels.sort()

# check image names and name_list_x.txt are the same
name_list_all = []
name_list_0 = []
name_list_1 = []

with open("./name_list_0.txt", "r") as file:
    for line in file:
        name_list_all.append(line.strip())
        name_list_0.append(line.strip())

with open("./name_list_1.txt", "r") as file:
    for line in file:
        name_list_all.append(line.strip())
        name_list_1.append(line.strip())

name_list_all.sort()

try:
    assert len(images) == len(name_list_all)
except AssertionError as e:
    print(f"len(images) = {len(images)}; len(name_list_all) = {len(name_list_all)}")
for (image, name) in zip(images, name_list_all):
    try:
        assert image[:-4] == name
    except AssertionError as e:
        print(f"image[:-4] = {image[:-4]}; name_list_all = {name}")

# check names of images and labels are the same
try:
    assert len(images) == len(labels)
except AssertionError as e:
    print(f"len(images) = {len(images)}; len(labels) = {len(labels)}")
for (image, label) in zip(images, labels):
    try:
        assert image[:-4] == label[:-4]
    except AssertionError as e:
        print(f"image[:-4] = {image[:-4]}; label[:-4] = {label[:-4]}")

# Shuffle and split
random.seed(42)
random.shuffle(name_list_0)
random.shuffle(name_list_1)
n_train_0 = int(train_percentage * len(name_list_0))
n_train_1 = int(train_percentage * len(name_list_1))
n_val_0 = int((1 - train_percentage) * len(name_list_0))
n_val_1 = int((1 - train_percentage) * len(name_list_1))

train_images = name_list_0[:n_train_0] + name_list_1[:n_train_1]
val_images = name_list_0[n_train_0:] + name_list_1[n_train_1:]
splits = {"train": train_images, "val": val_images}

# Copy images and labels
for split_name, images in splits.items():
    print(f"\nCopying {split_name} data and labels...\n")
    for i in tqdm(range(len(images))):
        image_name = images[i] + ".jpg"
        label_name = images[i] + ".txt"
        shutil.copy(images_dir + "/" + image_name, output_dir + "/" + "images" + "/" + split_name + "/" + image_name)
        shutil.copy(labels_dir + "/" + label_name, output_dir + "/" + "labels" + "/" + split_name + "/" + label_name)

print("\nDone splitting dataset.")
