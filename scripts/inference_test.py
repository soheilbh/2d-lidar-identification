# YOLOv8 LiDAR Object Detection Inference Script
# This script performs inference using a trained YOLOv8 model on LiDAR-generated RGB images
# Supports both pure inference timing and visualization modes for performance analysis
# Designed for real-time object detection in 2D LiDAR applications

import torch
from ultralytics import YOLO
import cv2
import time
import os

# --- Global Configuration ---
# Control whether to enable visualization or focus on pure inference timing
ENABLE_VISUALIZATION = False # Set to False to disable visualization and see pure inference timea

# --- Paths ---
# Model path: Points to the trained YOLOv8 model weights
model_path = "training_outputs/120_DO_simple_Fused/webots_model_assets_simpel_fused_fulltrain_120/yolov8n_lidar.pt"
# Input image path: LiDAR-generated RGB image for inference
image_path = "output/frame_0_0_0_0.png"
# Output path: Where to save the visualization result (if enabled)
output_path = "output/multi_frame/detected_images/frame_0_0_0_0.png"

# --- Load YOLO model ---
# Initialize YOLOv8 model with detection task
model = YOLO(model_path, task='detect')
# Extract model type from file extension for reporting
model_type = os.path.splitext(model_path)[-1].lstrip('.').upper()
print(f"Running model type: {model_type}")

# --- Warmup run (first inference is always slower) ---
# Perform initial inference to warm up the model and avoid timing the first slow run
print("Warming up model...")
_ = model(image_path, imgsz=[64, 384])  # Use same image size as training (64x384)

# --- Pure Inference (just the model) ---
# Measure only the model inference time without any post-processing
print("Running inference test...")
start_time = time.time()
results = model(image_path, imgsz=[64, 384])  # Run inference on the LiDAR image
pure_inference_time = time.time() - start_time
print(f"Pure inference time: {pure_inference_time*1000:.1f} ms")

# --- Full Processing (including visualization) ---
# Measure time for complete processing including detection parsing and visualization
start_time = time.time()

# Load image and process detections
if ENABLE_VISUALIZATION:
    im_bgr = cv2.imread(image_path)  # Load image for visualization (BGR format for OpenCV)

# Draw detections
# Process each detection result from the model
for r in results:
    # Extract detection information in different formats
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format
    boxes_xywh = r.boxes.xywh.cpu().numpy()  # Bounding boxes in (center_x, center_y, width, height) format
    confs = r.boxes.conf.cpu().numpy()  # Confidence scores for each detection
    class_ids = r.boxes.cls.cpu().numpy().astype(int)  # Class IDs for each detection
    names = r.names  # Class name mapping

    # Process each detected object
    for (x1, y1, x2, y2), (cx, cy, w, h), conf, class_id in zip(boxes_xyxy, boxes_xywh, confs, class_ids):
        # Create label with class name and confidence score
        label = f"{names[class_id]} {conf:.2f}"
        # Print detection details to console
        print(f"[{label}] -> center: ({cx:.1f}, {cy:.1f}), size: ({w:.1f}x{h:.1f})")

        if ENABLE_VISUALIZATION:
            color = (0, 255, 0)  # Green color for bounding boxes
            # Draw bounding box rectangle on image
            cv2.rectangle(im_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            # Add label text above the bounding box
            cv2.putText(im_bgr, label, (int(x1), max(int(y1)-2, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)

# Show and save (only if visualization is enabled)
if ENABLE_VISUALIZATION:
    cv2.imshow("YOLOv8 Detection", im_bgr)  # Display the image with detections
    cv2.imwrite(output_path, im_bgr)  # Save the annotated image
    # cv2.waitKey(0)  # Removed - this waits for user input and affects timing
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("Visualization enabled - image displayed and saved")
else:
    print("Visualization disabled - pure inference only")

# Calculate and display timing information
full_processing_time = time.time() - start_time
print(f"Full processing time: {full_processing_time:.4f} seconds")
print(f"Visualization overhead: {full_processing_time:.4f} seconds")
print(f"Total time (inference + processing): {pure_inference_time + full_processing_time:.4f} seconds")