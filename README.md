# Plant Disease Detection using YOLOv11

This project focuses on training and deploying a YOLOv11 model to detect plant diseases (e.g., Scab, Rust) and assess plant health. The code includes training in Google Colab and a script for inference with health scoring.

## Files Overview

### 1. `main_plant_disease_project_yolo11.py`
- **Purpose**: Train and validate a YOLOv11 model on a plant disease dataset in Google Colab.
- **Key Steps**:
  - Installs dependencies (Ultralytics, supervision, Roboflow).
  - Mounts Google Drive for dataset access.
  - Trains the model with varying hyperparameters (epochs, batch size).
  - Validates the model and computes metrics (mAP).
  - Runs predictions on test images and visualizes results.

### 2. `yolov11_detection.py`
- **Purpose**: Perform inference using a trained YOLOv11 model to detect diseases and calculate health scores.
- **Features**:
  - Loads a trained model from a specified path.
  - Processes images to extract detection results (bounding boxes, labels).
  - Maps detected labels to health scores (e.g., "Healthy" = 4, "Multiple Disease" = 1).

## Setup

### Dependencies
- Python 3.8+
- Install required packages:
  ```bash
  pip install ultralytics supervision roboflow opencv-python Pillow

##Dataset Preparation
Ensure the dataset is structured with data.yaml specifying paths and class labels.

Example data.yaml content:

train: ../train/images
val: ../val/images
test: ../test/images
names:
  0: Healthy
  1: Scab
  2: Rust
  3: Multiple Disease

## Usage
### Training (Colab Notebook)
Upload the dataset to Google Drive.

Update paths in data.yaml to match your Drive directory.

Run training commands:
!yolo task=detect mode=train model=yolo11n.pt data=data.yaml epochs=50 imgsz=640

####Validate and predict:
!yolo task=detect mode=val model=runs/detect/train4/weights/best.pt data=data.yaml
!yolo task=detect mode=predict model=best.pt source=Test/images
### Inference (yolov11_detection.py)
Update the model path and image path in the script.

Run the script:
python yolov11_detection.py
Output includes detected labels and health scores

## Health Score Mapping
### Detected Label	##Health Score
Healthy	                4
Scab	                  3
Rust	                  2
Multiple Disease	      1
Unknown/No Detection	  0

## Notes
### Model Path Fix: In yolov11_detection.py, ensure YOLO(weights="path/to/model.pt") uses weights (not weight).

### Class Order: The calculate_health_score function assumes class indices in data.yaml match the labels listed. Verify this to avoid misclassification.

#Results
Training metrics (mAP, confusion matrices) are saved in runs/detect/train*/.

Prediction visualizations are stored in runs/detect/predict*/.

This README provides a concise overview of the project, setup steps, usage instructions, and critical considerations for reproducibility.
