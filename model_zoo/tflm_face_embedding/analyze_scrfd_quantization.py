#!/usr/bin/env python3
"""
Analyze SCRFD quantization: compare float32 vs int8 model outputs.

This script:
1. Loads both float32 and int8 TFLite models
2. Runs inference on calibration images
3. Compares outputs (scores, boxes, keypoints) between models
4. Visualizes detection results with bbox and landmarks
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Please install tensorflow: pip install tensorflow")
    sys.exit(1)

# Config
INPUT_SIZE = 160
FLOAT32_MODEL = "scrfd_tf_output/scrfd_500m_kps_160_fixed_float32.tflite"
INT8_MODEL = "scrfd_500m_kps_int8.tflite"
CALIB_DIR = "calibration_data/fd_160"
OUTPUT_DIR = "output/scrfd_analysis"

# Detection thresholds
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def load_model(model_path: str):
    """Load TFLite model and return interpreter."""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_model_info(interpreter):
    """Get input/output details from interpreter."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    print(f"\n  Input: {input_details['shape']} {input_details['dtype']}")
    print(f"  Input quantization: {input_details.get('quantization', 'N/A')}")
    print(f"  Outputs: {len(output_details)}")

    for i, out in enumerate(output_details):
        shape = out['shape']
        dtype = out['dtype']
        quant = out.get('quantization_parameters', {})
        scale = quant.get('scales', [])
        zp = quant.get('zero_points', [])

        # Determine output type
        if len(shape) == 2:
            if shape[1] == 1:
                desc = "scores"
            elif shape[1] == 4:
                desc = "boxes"
            elif shape[1] == 10:
                desc = "keypoints"
            else:
                desc = "unknown"
        else:
            desc = "unknown"

        scale_str = f"scale={scale[0]:.6f}" if len(scale) > 0 else ""
        zp_str = f"zp={zp[0]}" if len(zp) > 0 else ""
        print(f"    [{i}] {out['name']}: {shape} {dtype} ({desc}) {scale_str} {zp_str}")

    return input_details, output_details


def preprocess_image(img_path: str, input_size: int = 160, dtype=np.float32, quant_params=None):
    """
    Load and preprocess image for SCRFD.

    Args:
        img_path: Path to image file
        input_size: Target size
        dtype: Output dtype (float32 or int8)
        quant_params: Quantization parameters (scale, zero_point) for int8

    Returns:
        preprocessed image, original image
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    orig_img = img.copy()

    # Resize
    img = cv2.resize(img, (input_size, input_size))

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert to int8 if needed
    if dtype == np.int8 and quant_params:
        scale, zero_point = quant_params
        img = img / scale + zero_point
        img = np.clip(img, -128, 127).astype(np.int8)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img, orig_img


def run_inference(interpreter, input_data):
    """Run inference and return outputs."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    outputs = []
    for out in output_details:
        data = interpreter.get_tensor(out['index'])

        # Dequantize if int8
        if out['dtype'] == np.int8:
            quant_params = out.get('quantization_parameters', {})
            scale = quant_params.get('scales', [1.0])
            zp = quant_params.get('zero_points', [0])
            if len(scale) > 0:
                data = (data.astype(np.float32) - zp[0]) * scale[0]

        outputs.append(data)

    return outputs


def parse_scrfd_outputs(outputs, input_size=160):
    """
    Parse SCRFD outputs into detections.

    SCRFD outputs 9 tensors: 3 scales x (scores, boxes, keypoints)
    Stride 8:  anchors at (input_size/8)^2 * 2 locations
    Stride 16: anchors at (input_size/16)^2 * 2 locations
    Stride 32: anchors at (input_size/32)^2 * 2 locations
    """
    strides = [8, 16, 32]

    # Group outputs by stride
    grouped = {}

    # Expected anchor counts for 160x160 input
    expected_counts = {
        800: 8,   # (160/8)^2 * 2 = 20*20*2 = 800
        200: 16,  # (160/16)^2 * 2 = 10*10*2 = 200
        50: 32,   # (160/32)^2 * 2 = 5*5*2 = 50
    }

    for out in outputs:
        # Handle batch dimension if present
        if len(out.shape) == 3:
            out = out[0]  # Remove batch: [1, anchors, features] -> [anchors, features]

        count = out.shape[0]  # Number of anchors

        # Find matching stride
        stride = None
        for exp_count, exp_stride in expected_counts.items():
            if count == exp_count:
                stride = exp_stride
                break

        if stride is None:
            continue

        if stride not in grouped:
            grouped[stride] = {}

        # Classify output type by feature dimension
        feat_dim = out.shape[-1]
        if feat_dim == 1:
            grouped[stride]['scores'] = out
        elif feat_dim == 4:
            grouped[stride]['boxes'] = out
        elif feat_dim == 10:
            grouped[stride]['keypoints'] = out

    # Decode detections
    detections = []

    for stride in strides:
        if stride not in grouped:
            continue

        data = grouped[stride]
        if 'scores' not in data or 'boxes' not in data:
            continue

        scores = data['scores'].flatten()
        boxes = data['boxes']
        kps = data.get('keypoints', None)

        # Generate anchor centers
        feat_size = input_size // stride

        anchors = []
        for y in range(feat_size):
            for x in range(feat_size):
                # 2 anchors per location
                anchors.append((x, y))
                anchors.append((x, y))

        for i, score in enumerate(scores):
            if score < SCORE_THRESHOLD:
                continue

            ax, ay = anchors[i]
            cx = (ax + 0.5) * stride
            cy = (ay + 0.5) * stride

            # Decode box (center + offset format)
            box = boxes[i]
            x1 = cx - box[0] * stride
            y1 = cy - box[1] * stride
            x2 = cx + box[2] * stride
            y2 = cy + box[3] * stride

            det = {
                'score': float(score),
                'box': [x1, y1, x2, y2],
                'keypoints': None
            }

            # Decode keypoints (use anchor corner, not center)
            if kps is not None:
                kp = kps[i]
                landmarks = []
                for k in range(5):
                    # Keypoints use anchor corner (ax, ay) * stride, not center
                    lx = ax * stride + kp[k*2] * stride
                    ly = ay * stride + kp[k*2+1] * stride
                    landmarks.append([lx, ly])
                det['keypoints'] = landmarks

            detections.append(det)

    # NMS
    if len(detections) > 0:
        detections = nms(detections, NMS_THRESHOLD)

    return detections


def nms(detections, threshold=0.4):
    """Apply non-maximum suppression."""
    if len(detections) == 0:
        return []

    boxes = np.array([d['box'] for d in detections])
    scores = np.array([d['score'] for d in detections])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


def match_outputs(outputs_f32, outputs_int8):
    """Match float32 and int8 outputs by shape."""
    matched = []

    for f32 in outputs_f32:
        for i8 in outputs_int8:
            if f32.shape == i8.shape:
                # Check if already matched
                already_matched = any(np.array_equal(i8, m[1]) for m in matched)
                if not already_matched:
                    matched.append((f32, i8))
                    break

    return matched


def compute_output_diff(outputs_f32, outputs_int8):
    """Compute difference metrics between float32 and int8 outputs."""
    print("\n" + "=" * 60)
    print("OUTPUT COMPARISON: float32 vs int8")
    print("=" * 60)

    # Match outputs by shape
    matched_pairs = match_outputs(outputs_f32, outputs_int8)
    print(f"\nMatched {len(matched_pairs)} output pairs")

    metrics = []

    for i, (f32, i8) in enumerate(matched_pairs):
        # Determine output type
        if f32.shape[-1] == 1:
            name = "scores"
        elif f32.shape[-1] == 4:
            name = "boxes"
        elif f32.shape[-1] == 10:
            name = "keypoints"
        else:
            name = f"output_{i}"

        # Compute metrics
        abs_diff = np.abs(f32 - i8)
        mse = np.mean((f32 - i8) ** 2)
        mae = np.mean(abs_diff)
        max_diff = np.max(abs_diff)

        # Relative error (avoid division by zero)
        mask = np.abs(f32) > 1e-6
        if np.any(mask):
            rel_error = np.mean(np.abs((f32[mask] - i8[mask]) / f32[mask])) * 100
        else:
            rel_error = 0

        # Correlation
        if f32.size > 1:
            corr = np.corrcoef(f32.flatten(), i8.flatten())[0, 1]
        else:
            corr = 1.0

        print(f"\n[{i}] {name} {f32.shape}:")
        print(f"    MSE:       {mse:.6f}")
        print(f"    MAE:       {mae:.6f}")
        print(f"    Max Diff:  {max_diff:.6f}")
        print(f"    Rel Error: {rel_error:.2f}%")
        print(f"    Corr:      {corr:.4f}")
        print(f"    F32 range: [{f32.min():.4f}, {f32.max():.4f}]")
        print(f"    I8  range: [{i8.min():.4f}, {i8.max():.4f}]")

        metrics.append({
            'name': name,
            'shape': f32.shape,
            'mse': mse,
            'mae': mae,
            'max_diff': max_diff,
            'rel_error': rel_error,
            'correlation': corr
        })

    return metrics


def draw_detections(img, detections, color=(0, 255, 0), scale_x=1.0, scale_y=1.0):
    """Draw bounding boxes and landmarks on image."""
    for det in detections:
        box = det['box']
        score = det['score']
        kps = det['keypoints']

        # Scale coordinates
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw score
        label = f"{score:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

        # Draw keypoints (5 facial landmarks)
        if kps is not None:
            kp_colors = [
                (255, 0, 0),    # Left eye - Red
                (0, 255, 0),    # Right eye - Green
                (0, 0, 255),    # Nose - Blue
                (255, 255, 0),  # Left mouth - Cyan
                (255, 0, 255),  # Right mouth - Magenta
            ]
            for j, (lx, ly) in enumerate(kps):
                px = int(lx * scale_x)
                py = int(ly * scale_y)
                cv2.circle(img, (px, py), 3, kp_colors[j], -1)

    return img


def main():
    print("=" * 60)
    print("SCRFD Quantization Analysis")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    print(f"\nLoading float32 model: {FLOAT32_MODEL}")
    interp_f32 = load_model(FLOAT32_MODEL)
    if interp_f32 is None:
        return 1
    get_model_info(interp_f32)

    print(f"\nLoading int8 model: {INT8_MODEL}")
    interp_i8 = load_model(INT8_MODEL)
    if interp_i8 is None:
        return 1
    get_model_info(interp_i8)

    # Get input quantization params for int8 model
    i8_input_details = interp_i8.get_input_details()[0]
    i8_quant = i8_input_details.get('quantization_parameters', {})
    i8_scale = i8_quant.get('scales', [1.0])
    i8_zp = i8_quant.get('zero_points', [0])
    quant_params = (i8_scale[0], i8_zp[0]) if len(i8_scale) > 0 else None

    print(f"\nInt8 input quantization: scale={i8_scale}, zero_point={i8_zp}")

    # Load calibration images
    calib_path = Path(CALIB_DIR)
    image_files = sorted(calib_path.glob("*.jpg"))[:10]  # Use first 10 images
    print(f"\nAnalyzing {len(image_files)} calibration images...")

    all_metrics = []

    for idx, img_path in enumerate(image_files):
        print(f"\n{'=' * 60}")
        print(f"Image {idx + 1}: {img_path.name}")
        print("=" * 60)

        # Preprocess for float32
        input_f32, orig_img = preprocess_image(str(img_path), INPUT_SIZE, dtype=np.float32)
        if input_f32 is None:
            continue

        # Preprocess for int8
        input_i8, _ = preprocess_image(str(img_path), INPUT_SIZE, dtype=np.int8, quant_params=quant_params)

        # Run inference
        outputs_f32 = run_inference(interp_f32, input_f32)
        outputs_i8 = run_inference(interp_i8, input_i8)

        # Compare outputs
        metrics = compute_output_diff(outputs_f32, outputs_i8)
        all_metrics.append(metrics)

        # Parse detections
        dets_f32 = parse_scrfd_outputs(outputs_f32, INPUT_SIZE)
        dets_i8 = parse_scrfd_outputs(outputs_i8, INPUT_SIZE)

        print(f"\nDetections: float32={len(dets_f32)}, int8={len(dets_i8)}")

        # Visualize
        h, w = orig_img.shape[:2]
        scale_x = w / INPUT_SIZE
        scale_y = h / INPUT_SIZE

        # Create side-by-side visualization
        vis_f32 = orig_img.copy()
        vis_i8 = orig_img.copy()

        draw_detections(vis_f32, dets_f32, color=(0, 255, 0), scale_x=scale_x, scale_y=scale_y)
        draw_detections(vis_i8, dets_i8, color=(0, 0, 255), scale_x=scale_x, scale_y=scale_y)

        # Add labels
        cv2.putText(vis_f32, "Float32", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_i8, "Int8", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Concatenate
        vis = np.hstack([vis_f32, vis_i8])

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"compare_{idx:04d}.jpg")
        cv2.imwrite(out_path, vis)
        print(f"Saved: {out_path}")

        # Print detection details
        if len(dets_f32) > 0:
            print("\nFloat32 detections:")
            for d in dets_f32:
                box = d['box']
                print(f"  score={d['score']:.3f}, box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]")

        if len(dets_i8) > 0:
            print("\nInt8 detections:")
            for d in dets_i8:
                box = d['box']
                print(f"  score={d['score']:.3f}, box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_metrics:
        # Average metrics across all images
        output_names = [m['name'] for m in all_metrics[0]]

        for i, name in enumerate(output_names):
            mse_avg = np.mean([m[i]['mse'] for m in all_metrics])
            mae_avg = np.mean([m[i]['mae'] for m in all_metrics])
            max_diff_avg = np.mean([m[i]['max_diff'] for m in all_metrics])
            corr_avg = np.mean([m[i]['correlation'] for m in all_metrics])

            print(f"\n{name}:")
            print(f"  Avg MSE:      {mse_avg:.6f}")
            print(f"  Avg MAE:      {mae_avg:.6f}")
            print(f"  Avg Max Diff: {max_diff_avg:.6f}")
            print(f"  Avg Corr:     {corr_avg:.4f}")

    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
