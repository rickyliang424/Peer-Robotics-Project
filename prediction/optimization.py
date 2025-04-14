import os
from ultralytics import YOLO, SAM, FastSAM
from predict import inference

#%%
# model_det = YOLO("./models/best4.pt")
# model_det.export(format="engine", half=True, device=0)

#%%
# model_seg = FastSAM("./models/FastSAM-s.pt")
# model_seg.export(format="engine", half=True, device=0)

#%%
tensorrt_model_det = YOLO("./models/best4.engine", task='detect')
tensorrt_model_seg = FastSAM("./models/FastSAM-s.engine")
model_det = YOLO("./models/best4.pt")
model_seg = FastSAM("./models/FastSAM-s.pt")
files = ["../bag_imgs/" + file for file in os.listdir('../bag_imgs')]

detection = tensorrt_model_det
# detection = model_det
segmentation = tensorrt_model_seg
# segmentation = model_seg

inference(detection, segmentation, files, savepath="./res/", show=False)
