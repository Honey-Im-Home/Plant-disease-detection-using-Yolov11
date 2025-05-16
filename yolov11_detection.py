import cv2
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(weight="D:/Documents/plant_disease/yolov11/best.pt", augment=False, imgsz=608)

# Define a function to calculate the health score
def calculate_health_score(detection_label):
    if detection_label == "Healthy":
        return 4
    elif detection_label == "Scab":
        return 3
    elif detection_label == "Rust":
        return 2
    elif detection_label == "Multiple Disease":
        return 1
    else:
        return 0  # Default score for undetected or unknown labels

# Run batched inference on a list of images
image_path = "D:/Documents/plant_disease/plant_disease/Train/images/Train_801.jpg"
results = model([image_path], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    obb = result.obb  # Oriented bounding boxes (if applicable)
    probs = result.probs  # Probs object for classification outputs

    # Extracting the most confident label
    if probs is not None:
        detected_label = probs.top1  # Get the top label with the highest probability
        health_score = calculate_health_score(detected_label)
        print(f"Detected Label: {detected_label}, Health Score: {health_score}")
    else:
        print("No detections found.")
