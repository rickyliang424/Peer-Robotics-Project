import os
from ultralytics import YOLO, SAM, FastSAM
from predict import inference

export = True
# export = False

# model = "pt"
model = "onnx"
# model = "engine"

if export == True:
    model_det = YOLO("./models/best4.pt")
    model_seg = FastSAM("./models/FastSAM-s.pt")
    if model == "onnx":
        model_det.export(format="onnx", half=True, device=0)
        model_seg.export(format="onnx", half=True, device=0)
    if model == "engine":
        model_det.export(format="engine", half=True, device=0)
        model_seg.export(format="engine", half=True, device=0)
    
else:
    if model == "pt":
        model_det = YOLO("./models/best4.pt")
        model_seg = FastSAM("./models/FastSAM-s.pt")
    if model == "onnx":
        model_det = YOLO("./models/best4.onnx", task='detect')
        model_seg = FastSAM("./models/FastSAM-s.onnx")
    if model == "engine":
        model_det = YOLO("./models/best4.engine", task='detect')
        model_seg = FastSAM("./models/FastSAM-s.engine")
    
    files = ["../bag_imgs/" + file for file in os.listdir('../bag_imgs')]
    inference(model_det, model_seg, files, savepath="./res/", show=False)
