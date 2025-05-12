#!/usr/bin/env python3
"""
Object Detection in an Image using a Pre-trained YOLOv5 Model from PyTorch Hub

This script:
1. Loads a pre-trained YOLOv5s model via torch.hub.
2. Reads an input image from disk.
3. Runs object detection on the image.
4. Draws bounding boxes and labels on detected objects.
5. Displays the full annotated image in a resizable window.
6. Saves the result to disk.

Requirements:
- Python 3.x
- PyTorch
- OpenCV (cv2)
- PIL (optional, for alternative image loading)

Install dependencies:
    pip install torch torchvision opencv-python pillow
"""

import torch
import cv2

# 1. Load the YOLOv5s model from PyTorch Hub (pre-trained on COCO dataset)
#    This will automatically download weights if not already cached.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # set model to evaluation mode

# 2. Read the input image
IMAGE_PATH = 'Test.jpg'  # replace with your image filename
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image file '{IMAGE_PATH}' not found.")

# Convert BGR (OpenCV default) to RGB for model inference
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Perform inference
#    The model accepts a list of images or NumPy arrays
results = model([img_rgb])

# 4. Parse results
#    results.xyxy[0] is a tensor of detections with each row: [xmin, ymin, xmax, ymax, confidence, class]
detections = results.xyxy[0].cpu().numpy()
class_names = results.names  # e.g., {0: 'person', 1: 'bicycle', ...}

# 5. Draw bounding boxes and labels on the original image
for *box, conf, cls_idx in detections:
    xmin, ymin, xmax, ymax = map(int, box)
    label = f"{class_names[int(cls_idx)]} {conf:.2f}"
    # Draw rectangle
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
    # Draw label background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (xmin, ymin - text_h - 4), (xmin + text_w, ymin), (0, 255, 0), -1)
    # Put label text
    cv2.putText(image, label, (xmin, ymin - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=1)

# 6. Display the result in a resizable window and force a window size that fits the screen
window_name = 'YOLOv5 Object Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)      # allow window to be resized
cv2.resizeWindow(window_name, 1280, 720)             # resize to a resolution that fits your screen
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Save the annotated image
OUTPUT_PATH = 'output.jpg'
cv2.imwrite(OUTPUT_PATH, image)
print(f"Result saved to '{OUTPUT_PATH}'")
