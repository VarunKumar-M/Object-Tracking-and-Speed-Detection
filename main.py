from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Make sure this model is available

# Input and output paths
input_video_path = "input_videos/input.mp4"
output_video_path = "output_videos/output.mp4"

# Open video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)[0]
    annotated_frame = results.plot()

    # Save annotated frame
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Output saved to:", output_video_path)
