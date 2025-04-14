import os
import random
import shutil
from tqdm import tqdm

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
random.shuffle(images)
n_train = int(0.8 * len(images))
n_val = int(0.2 * len(images))

train_images = images[:n_train]
val_images = images[n_train:]
splits = {"train": train_images, "val": val_images}

# Copy images and labels
for split_name, images in splits.items():
    print(f"\nCopying {split_name} data and labels...\n")
    for i in tqdm(range(len(images))):
        image_name = images[i]
        label_name = images[i][:-4] + ".txt"
        shutil.copy(images_dir + "/" + image_name, output_dir + "/" + "images" + "/" + split_name + "/" + image_name)
        shutil.copy(labels_dir + "/" + label_name, output_dir + "/" + "labels" + "/" + split_name + "/" + label_name)

print("\nDone splitting dataset.")
