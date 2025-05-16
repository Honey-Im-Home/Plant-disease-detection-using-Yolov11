# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:35:01 2024

@author: user
"""

from ultralytics import YOLO

model = YOLO(weight="D:/Documents/plant_disease/yolov11/best.pt", augment=False,imgsz=608)
import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture("http://192.168.0.103:4747/video")
#cap = cv.VideoCapture(0)
if not cap.isOpened():
     print("Cannot open camera")
     exit()
while True:
    
 # Capture frame-by-frame
     ret, frame = cap.read()
     
     # if frame is read correctly ret is True
     if not ret:
         print("Can't receive frame (stream end?). Exiting ...")
         break
     # Our operations on the frame come here
     # Display the resulting frame
     #results = model.predict(source=frame, save=False)
     # Process results generator
     results = model(frame)
        # Visualize the results on the frame
     annotated_frame = results[0].plot()
       # display to screen
     cv.imshow('frame', annotated_frame)
     if cv.waitKey(1) == ord('q'):
         break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()