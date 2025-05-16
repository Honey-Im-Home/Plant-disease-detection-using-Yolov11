# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 23:49:14 2024

@author: user
"""
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO("D:/Documents/plant_disease/yolov11/best.pt")
# Run batched inference on a list of images
results = model(["D:/Documents/plant_disease/plant_disease/Train/images/Train_801.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Orie
    