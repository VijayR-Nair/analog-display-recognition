"""
Analog Display Recognition System (ADRS)

This script:
1) Captures frames from an ESP32-CAM stream URL
2) Runs TensorFlow Object Detection (Faster R-CNN model) to detect:
   - Needle
   - Dial center (reference point)
   - Zeros (scale marks)
3) Computes the analog meter value using a geometry-based algorithm
4) Exposes the latest value via the global variable `measurement`
   (so your GUI script can read it as: main_adrs.measurement)

NOTE:
- This repo intentionally does NOT include the trained model or dataset.
- You must set correct paths for LABEL_MAP_PATH and SAVED_MODEL_DIR.
"""

from __future__ import annotations

import math
import time
import threading
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


# CONFIGURATION (EDIT THESE FOR YOUR MACHINE)

@dataclass
class Config:
    # ESP32-CAM endpoint for a single image capture
    stream_url: str = "http://192.168.7.174/capture"

    # Paths to TensorFlow label map and saved model directory
    # (Replace with your local paths)
    label_map_path: str = r"D:\Learn\THD\Intelligent Systems\Labelled Data\Data_v02\Needle-Zero-Referencepoint_label_map.pbtxt"
    saved_model_dir: str = r"D:\Learn\THD\Intelligent Systems\Model_R01\inference_graph2\saved_model"

    # Inference settings
    input_size: Tuple[int, int] = (640, 640)
    min_score_thresh: float = 0.90

    # Meter settings (you can also read these from GUI later if you want)
    max_value_read: float = 120.0

    # Save annotated frames (optional)
    save_frames: bool = True
    save_folder: str = r"D:\Photos\images"

    # Timing
    capture_interval_sec: float = 2.0

    # Display window
    show_window: bool = True
    window_name: str = "ADRS Live View"


CFG = Config()


# GLOBAL VALUE (GUI reads this)

# Initialize measurement so GUI doesn't crash before the first detection.
measurement: float = 0.0

# Optional: lock if GUI and main thread access `measurement` simultaneously.
measurement_lock = threading.Lock()



# UTILITIES


def load_image_into_numpy_array(path: str) -> np.ndarray:
    """Load image from disk into a NumPy array."""
    img_data = tf.io.gfile.GFile(path, "rb").read()
    image = Image.open(BytesIO(img_data))
    return np.array(image)


def fetch_esp32_frame(url: str) -> np.ndarray:
    """
    Fetch a single frame from ESP32-CAM capture URL and decode it using OpenCV.
    Returns: BGR image (OpenCV format)
    """
    with urllib.request.urlopen(url) as response:
        img_bytes = response.read()

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError("Failed to decode image from ESP32-CAM stream.")

    return frame


def run_inference_for_single_image(
    model: Any,
    image_np: np.ndarray
) -> Dict[str, Any]:
    """
    Run TensorFlow SavedModel inference on a single image.
    Returns a dict with boxes, scores, classes, etc.
    """
    image_np = np.asarray(image_np)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]  # add batch dimension

    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)

    # Number of detections
    num_detections = int(output_dict.pop("num_detections"))

    # Convert tensors to numpy arrays and slice valid detections
    output_dict = {k: v[0, :num_detections].numpy() for k, v in output_dict.items()}
    output_dict["num_detections"] = num_detections
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

    # If masks exist, reframe them
    if "detection_masks" in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"],
            output_dict["detection_boxes"],
            image_np.shape[0],
            image_np.shape[1],
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict


def filter_detections(
    output_dict: Dict[str, Any],
    min_score: float,
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    """
    Convert TF detections into a cleaner list of dicts and filter by score threshold.
    Bounding boxes are converted from normalized coordinates to pixel coordinates.
    """
    dets: List[Dict[str, Any]] = []

    scores = output_dict.get("detection_scores", [])
    classes = output_dict.get("detection_classes", [])
    boxes = output_dict.get("detection_boxes", [])

    for i in range(len(scores)):
        if float(scores[i]) < min_score:
            continue

        ymin, xmin, ymax, xmax = boxes[i]  # normalized
        dets.append(
            {
                "class": int(classes[i]),
                "score": float(scores[i]),
                "bbox": {
                    "xmin": float(xmin * image_w),
                    "ymin": float(ymin * image_h),
                    "xmax": float(xmax * image_w),
                    "ymax": float(ymax * image_h),
                },
            }
        )

    return dets


def bbox_center(b: Dict[str, float]) -> Tuple[float, float]:
    """Return center (x,y) of bbox dict."""
    cx = (b["xmin"] + b["xmax"]) / 2.0
    cy = (b["ymin"] + b["ymax"]) / 2.0
    return cx, cy


def pick_bottom_left_zero_bbox(detections: List[Dict[str, Any]], zero_class_id: int = 3) -> Optional[Dict[str, float]]:
    """
    Your project logic:
    - bottom-left and bottom-right zeros have the largest ymax
    - among them, bottom-left has minimum xmin
    This extracts the bottom-left zero bbox.
    """
    zeros = [d for d in detections if d["class"] == zero_class_id]
    if len(zeros) < 2:
        return None

    # Take two zeros with largest ymax (bottom-most)
    zeros_sorted = sorted(zeros, key=lambda z: z["bbox"]["ymax"], reverse=True)
    bottom_two = zeros_sorted[:2]

    # From those, choose the left-most (min xmin)
    bottom_left = min(bottom_two, key=lambda z: z["bbox"]["xmin"])
    return bottom_left["bbox"]


def compute_meter_reading(
    detections: List[Dict[str, Any]],
    max_value_read: float,
    needle_class_id: int = 1,
    reference_class_id: int = 2,
    zero_class_id: int = 3,
) -> Optional[float]:
    """
    Compute analog meter reading using:
    - needle center point
    - reference (dial center) center point
    - bottom-left zero center point

    Returns: measurement value (float) or None if required detections not found.
    """

    # Extract required bboxes
    needle = next((d["bbox"] for d in detections if d["class"] == needle_class_id), None)
    reference = next((d["bbox"] for d in detections if d["class"] == reference_class_id), None)
    zero_bbox = pick_bottom_left_zero_bbox(detections, zero_class_id=zero_class_id)

    if not needle or not reference or not zero_bbox:
        return None

    needle_cx, needle_cy = bbox_center(needle)
    ref_cx, ref_cy = bbox_center(reference)
    zero_cx, zero_cy = bbox_center(zero_bbox)

    # -algorithm-

    # Compute inner sweep angle of the meter.
    # used a triangle between reference center and zero point to find outer angle,
    # then inner_angle = 360 - outer_angle.
    adj_side = zero_cy - ref_cy
    hyp_side = math.sqrt((zero_cx - ref_cx) ** 2 + (zero_cy - ref_cy) ** 2)
    if hyp_side == 0:
        return None

    angle = math.acos(adj_side / hyp_side)
    outer_angle = 2.0 * math.degrees(angle)
    inner_angle = 360.0 - outer_angle
    if inner_angle <= 0:
        return None

    # Least count: value per degree of needle sweep
    least_count = max_value_read / inner_angle

    # Vectors from reference center:
    vec_1 = [zero_cx - ref_cx, zero_cy - ref_cy]      # reference -> zero
    vec_2 = [needle_cx - ref_cx, needle_cy - ref_cy]  # reference -> needle

    dot = (vec_1[0] * vec_2[0]) + (vec_1[1] * vec_2[1])
    mag1 = math.sqrt(vec_1[0] ** 2 + vec_1[1] ** 2)
    mag2 = math.sqrt(vec_2[0] ** 2 + vec_2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return None

    # Angle swept by needle relative to zero
    angle_swept = math.acos(dot / (mag1 * mag2))
    angle_swept_deg = math.degrees(angle_swept)

    # Final measurement
    return angle_swept_deg * least_count



# MAIN LOOP


def main() -> None:
    global measurement

    # Load label map & model once (expensive operations)
    category_index = label_map_util.create_category_index_from_labelmap(
        CFG.label_map_path, use_display_name=True
    )
    model = tf.saved_model.load(CFG.saved_model_dir)

    print("✅ Model loaded.")
    print("✅ Starting live capture from:", CFG.stream_url)

    while True:
        try:
            # 1) Capture a frame from ESP32-CAM
            frame = fetch_esp32_frame(CFG.stream_url)

            # 2) Resize for model
            w, h = CFG.input_size
            image_np = cv2.resize(frame, (w, h))

            # 3) Inference
            output_dict = run_inference_for_single_image(model, image_np)

            # 4) Filter detections (IMPORTANT: local list, not global growing list)
            dets = filter_detections(output_dict, CFG.min_score_thresh, w, h)

            # 5) Compute measurement
            value = compute_meter_reading(dets, max_value_read=CFG.max_value_read)

            if value is not None:
                with measurement_lock:
                    measurement = float(value)
                print(f"📌 Measurement: {measurement:.2f}")
            else:
                print("⚠️ Measurement not computed (missing detections).")

            # 6) Visualize boxes (optional but nice)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict["detection_boxes"],
                output_dict["detection_classes"],
                output_dict["detection_scores"],
                category_index,
                instance_masks=output_dict.get("detection_masks_reframed", None),
                use_normalized_coordinates=True,
                line_thickness=3,
                min_score_thresh=CFG.min_score_thresh,
            )

            # 7) Show window (optional)
            if CFG.show_window:
                cv2.imshow(CFG.window_name, image_np)

            # 8) Save annotated image (optional)
            if CFG.save_frames:
                ts = int(time.time())
                filename = f"{CFG.save_folder}/image_{ts}.jpg"
                cv2.imwrite(filename, image_np)

            # 9) Exit on 'q'
            if CFG.show_window and (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            time.sleep(CFG.capture_interval_sec)

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
