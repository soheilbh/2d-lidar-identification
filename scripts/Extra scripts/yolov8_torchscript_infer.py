import torch
from ultralytics import YOLO
import cv2
import time

# Paths
model_path = "training_outputs/120_DO_simple_Fused/webots_model_assets_simpel_fused_fulltrain_120/yolov8n_lidar.pt"
image_path = "output/frame_0_0_0_0.png"
output_path = "output/multi_frame/detected_images/frame_0_0_0_0.png"

# Load TorchScript YOLOv8 model
model = YOLO(model_path, task='detect')

# Run inference with timing
start_time = time.time()
results = model(image_path, imgsz=[64, 384])
detect_time = time.time() - start_time
print(f"Detection time: {detect_time:.4f} seconds")

# Load image
im_bgr = cv2.imread(image_path)

# Draw boxes and labels manually with small font
for r in results:
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()         # (x1, y1, x2, y2)
    boxes_xywh = r.boxes.xywh.cpu().numpy()         # (cx, cy, w, h)
    confs = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names

    for (x1, y1, x2, y2), (cx, cy, w, h), conf, class_id in zip(boxes_xyxy, boxes_xywh, confs, class_ids):
        label = f"{names[class_id]} {conf:.2f}"
        print(f"[{label}] → center: ({cx:.1f}, {cy:.1f}), size: ({w:.1f}×{h:.1f})")

        color = (0, 255, 0)
        cv2.rectangle(im_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.putText(im_bgr, label, (int(x1), max(int(y1)-2, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)

cv2.imshow("YOLOv8 Detection", im_bgr)
cv2.imwrite(output_path, im_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows() 