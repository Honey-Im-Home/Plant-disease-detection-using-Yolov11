# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:29:05 2024

@author: user
"""

import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture("http://192.168.0.103:4747/video")
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
     cv.imshow('frame', frame)
     if cv.waitKey(1) == ord('q'):
         break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()