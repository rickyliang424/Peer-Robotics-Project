import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM, FastSAM

import torch
from copy import deepcopy
from ultralytics.data.augment import LetterBox
from ultralytics.utils.plotting import Annotator, colors

def plot_annotates(img, res, savepath, show):
    annotator = Annotator(deepcopy(img), None, None, "Arial.ttf", False, example='')
    img = LetterBox(res.masks.shape[1:])(image=annotator.result())
    im_gpu = (torch.as_tensor(img, dtype=torch.float16, device=res.masks.data.device).permute(2,0,1).flip(0).contiguous()/255)
    annotator.masks(res.masks.data, colors=[colors(1, True)], im_gpu=im_gpu)
    
    for i, d in enumerate(reversed(res.boxes)):
        d_conf, id = float(d.conf), None if d.id is None else int(d.id.item())
        name = ("" if id is None else f"id:{id} ") + res.names[1]
        label = (f"{name} {d_conf:.2f}")
        annotator.box_label(d.xyxy.squeeze(), label, color=colors(0, True), rotated=None)
    
    if show:
        annotator.show()
    annotator.save(savepath)
    return annotator.result()

def inference(model_det, model_seg, files, savepath, show):
    files.sort()
    for file in files:
        img = cv2.resize(cv2.imread(file), (640, 640))
        
        ## Run detection inference on a image
        print("\nRunning detection...")
        results_det = model_det(img)  # return a list of Results objects
        
        res = results_det[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        box = xyxy[cls == 0, :]
        if box.shape[0] > 1:
            conf = res.boxes.conf.cpu().numpy()
            box = box[np.argmax(conf)]
        
        ## Run segmentation inference with bboxes prompt
        print("\nRunning segmentation...")
        results_seg = model_seg(img, imgsz=640, bboxes=box)
        
        ## Process results list
        res.boxes = res.boxes[cls == 1]
        res.masks = results_seg[0].masks
        
        os.makedirs(savepath, exist_ok=True)
        output_img = plot_annotates(img, res, savepath + file[12:], show)
        print(f'\nFinish {file}\n\n')

def main():
    # files = ["../bag_imgs/frame_000000.jpg", "../bag_imgs/frame_000016.jpg"]
    files = ["../bag_imgs/" + file for file in os.listdir('../bag_imgs')]
    show = False
    savepath = "./res/"
    
    model_det = YOLO("./models/best4.pt")
    # model_det = YOLO("./models/best5.pt")
    # model_det = YOLO("./models/best6.pt")
    # model_seg = SAM("./models/sam2.1_t.pt")
    # model_seg = SAM("./models/mobile_sam.pt")
    model_seg = FastSAM("./models/FastSAM-s.pt")

    inference(model_det, model_seg, files, savepath, show)

if __name__ == '__main__':
    main()
