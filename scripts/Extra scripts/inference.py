import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import nms

# --- Paths ---
model_path = "training_outputs/webots_model_assets_not_aligned_simple/yolov8n_lidar.onnx"
image_path = "output/multi_frame/simple_rgb/X_yolo1d_scan_2025-05-19_15-12-56/frame_0.png"
output_path = "output/multi_frame/detected_images/X_yolo1d_scan_2025-05-19_15-12-56/frame_0_detected.png"

# --- Class names ---
class_names = ['chair', 'box', 'desk', 'door_frame']

# --- Preprocessing ---
def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (384, 64))
    img_norm = img.astype(np.float32) / 255.0
    img_trans = img_norm.transpose(2, 0, 1)[None, :]
    return img, img_trans

# --- Postprocessing (TorchScript-style) ---
def postprocess_onnx_like_torchscript(onnx_output, conf_thres=0.25):
    pred = torch.tensor(onnx_output[0]).squeeze(0)  # (8, 504)

    boxes = pred[0:4, :].permute(1, 0)  # (504, 4)
    objectness = torch.sigmoid(pred[4, :])  # Apply sigmoid
    class_logits = pred[5:, :].permute(1, 0)  # (504, 4)
    class_scores = torch.sigmoid(class_logits)  # Apply sigmoid

    max_class_probs, class_ids = torch.max(class_scores, dim=1)
    final_scores = objectness * max_class_probs

    print(f"🔬 Max objectness: {objectness.max().item():.4f}")
    print(f"🔬 Max class prob: {class_scores.max().item():.4f}")
    print(f"🔬 Max final conf: {final_scores.max().item():.4f}")
    print(f"🔬 Num predictions above 0.01: {(final_scores > 0.01).sum().item()}")

    mask = final_scores > conf_thres
    boxes = boxes[mask]
    scores = final_scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        print("⚠️ No boxes above confidence threshold.")
        return []

    results = []
    for cls in torch.unique(class_ids):
        cls_mask = class_ids == cls
        cls_scores = scores[cls_mask]
        cls_boxes = boxes[cls_mask]
        top_idx = torch.argmax(cls_scores)

        cx, cy, w, h = cls_boxes[top_idx]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        results.append({
            'class': class_names[cls.item()],
            'conf': cls_scores[top_idx].item(),
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'center': [cx.item(), cy.item()],
            'size': [w.item(), h.item()]
        })

    return results

# --- Draw boxes ---
def draw_detections(image, results):
    for det in results:
        label = f"{det['class']} {det['conf']:.2f}"
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, label, (x1, max(y1 - 2, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
    return image

# --- Main Inference ---
def run_onnx_inference():
    print(f"📦 Running model type: ONNX")
    img_bgr, img_input = preprocess(image_path)

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    start_time = time.time()
    onnx_output = session.run(None, {input_name: img_input})
    infer_time = time.time() - start_time
    print(f"⏱️ ONNX inference: {infer_time:.4f} seconds")
    print(f"📦 ONNX outputs: {[x.shape for x in onnx_output]}")

    results = postprocess_onnx_like_torchscript(onnx_output)
    print(f"🧮 Detections: {len(results)}")
    for r in results:
        cx, cy = r['center']
        w, h = r['size']
        print(f"[{r['class']} {r['conf']:.2f}] → center: ({cx:.1f}, {cy:.1f}), size: ({w:.1f}×{h:.1f})")

    image_with_boxes = draw_detections(img_bgr, results)
    cv2.imwrite(output_path, image_with_boxes)
    print(f"✅ Saved to: {output_path}")

# --- Run ---
if __name__ == "__main__":
    run_onnx_inference()