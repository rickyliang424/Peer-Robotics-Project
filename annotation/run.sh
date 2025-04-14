#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./inference_on_images.py \
-c ./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p ./GroundingDINO/weights/groundingdino_swint_ogc.pth \
-i ../dataset \
-o "./res_ground" \
-t "ground"
