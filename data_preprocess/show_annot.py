import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

images = "../dataset/images"
labels = "../dataset/labels"
# folder = "train"
folder = "val"
savepath = f"./{folder}_annot"

def plot_boxes_to_image(img, tgt):
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    H, W = img.size[1], img.size[0]
    assert len(boxes) == len(labels), "boxes and labels must have same length"
    
    draw = ImageDraw.Draw(img)
    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * np.array([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
    return img

os.makedirs(savepath)
images_name = os.listdir(images + "/" + folder)
print(f"saving annotated {folder} images...")
for i in tqdm(range(len(images_name))):
    img_name = images_name[i]
    path = images + "/" + folder + "/" + img_name
    img = Image.open(path).convert("RGB")
    pred_phrases = []
    boxes_filt = []
    with open(labels + "/" + folder + "/" + img_name[:-4] + ".txt") as file:
        for line in file:
            annot = line.split(" ")
            if float(annot[0]) == 0.0:
                pred_phrases.append("Ground")
            else:
                pred_phrases.append("Pallet")
            boxes_filt.append([float(annot[1]), float(annot[2]), float(annot[3]), float(annot[4])])
    pred_dict = {"boxes": np.array(boxes_filt), "labels": np.array(pred_phrases)}
    image_with_box = plot_boxes_to_image(img, pred_dict)
    image_with_box.save(savepath + "/" + img_name)
