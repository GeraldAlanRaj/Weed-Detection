import pathlib
import torch
import cv2
import sys

# Patch for WindowsPath issue on macOS/Linux
pathlib.WindowsPath = pathlib.PosixPath

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/geraldalanraja/Desktop/Weed_Detection/best.pt', force_reload=True)
model.to('cpu')  # Or 'cuda' if using GPU

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Render results
    annotated_frame = results.render()[0]

    # Show frame
    cv2.imshow('YOLOv5 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
