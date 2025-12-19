#!/usr/bin/env python3
"""
Enhanced SCRFD Quantization with more calibration data.

Based on convert_scrfd.py but uses 5000+ LFW images for better INT8 precision.
"""

import os
import sys
import argparse
import shutil
import numpy as np
from pathlib import Path

print("=" * 60)
print("SCRFD Enhanced Quantization (More Calibration Data)")
print("=" * 60)

# Check dependencies
try:
    import onnx
    print(f"ONNX version: {onnx.__version__}")
except ImportError:
    print("Please install onnx: pip install onnx")
    sys.exit(1)

try:
    import onnxsim
    print(f"onnxsim version: {onnxsim.__version__}")
except ImportError:
    print("Please install onnxsim: pip install onnxsim")
    sys.exit(1)

try:
    import onnx2tf
    print(f"onnx2tf version: {onnx2tf.__version__}")
except ImportError:
    print("Please install onnx2tf: pip install onnx2tf==1.22.3")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Please install tensorflow")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Please install opencv-python")
    sys.exit(1)


def load_lfw_images(lfw_dir: str, input_size: int, max_images: int = 5000) -> np.ndarray:
    """
    Load images from LFW directory (nested structure: person/image.jpg).

    Args:
        lfw_dir: Path to lfw-deepfunneled directory
        input_size: Target size (160)
        max_images: Max images to load

    Returns:
        Array [N, H, W, 3] NHWC float32 [0,1]
    """
    images = []
    lfw_path = Path(lfw_dir)

    if not lfw_path.exists():
        print(f"LFW directory not found: {lfw_dir}")
        return None

    # Collect all jpg files
    all_images = sorted(lfw_path.rglob("*.jpg"))
    print(f"Found {len(all_images)} images in LFW")

    # Sample evenly from all images
    if len(all_images) > max_images:
        step = len(all_images) // max_images
        all_images = all_images[::step][:max_images]

    print(f"Loading {len(all_images)} images...")

    for i, img_path in enumerate(all_images):
        if i % 1000 == 0:
            print(f"  Loaded {i}/{len(all_images)}...")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Resize to target size
        img = cv2.resize(img, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)

    print(f"Loaded {len(images)} calibration images")
    return np.array(images) if images else None


def fix_onnx_for_target_size(onnx_path: str, output_path: str, input_size: int = 160) -> str:
    """
    Fix ONNX model for target input size.

    Copied from convert_scrfd.py - this function correctly handles the dynamic shapes.
    """
    print(f"\nFixing ONNX model for {input_size}x{input_size} input...")

    model = onnx.load(onnx_path)

    # Calculate output anchor counts for target size
    scale = input_size / 640.0
    shape_map = {
        12800: int(12800 * scale * scale),  # stride 8
        3200: int(3200 * scale * scale),    # stride 16
        800: int(800 * scale * scale),      # stride 32
    }

    print(f"  Output shape mapping: {shape_map}")

    # Fix output value info shapes
    for out in model.graph.output:
        dims = out.type.tensor_type.shape.dim
        for dim in dims:
            if dim.dim_value in shape_map:
                old_val = dim.dim_value
                dim.dim_value = shape_map[old_val]
                print(f"  Fixed output {out.name}: {old_val} -> {dim.dim_value}")

    # Fix value_info shapes (intermediate tensors)
    for vi in model.graph.value_info:
        dims = vi.type.tensor_type.shape.dim
        for dim in dims:
            if dim.dim_value in shape_map:
                old_val = dim.dim_value
                dim.dim_value = shape_map[old_val]

    # Fix input shape to static [1, 3, H, W]
    for inp in model.graph.input:
        if inp.name == "input.1":
            dims = inp.type.tensor_type.shape.dim
            dims[0].dim_value = 1
            dims[1].dim_value = 3
            dims[2].ClearField('dim_param')
            dims[2].dim_value = input_size
            dims[3].ClearField('dim_param')
            dims[3].dim_value = input_size
            print(f"  Fixed input shape to [1, 3, {input_size}, {input_size}]")

    # Rename input
    old_name = "input.1"
    new_name = "input_1"
    for inp in model.graph.input:
        if inp.name == old_name:
            inp.name = new_name
            print(f"  Renamed input: {old_name} -> {new_name}")

    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old_name:
                node.input[i] = new_name

    # Clear value_info to let onnxsim recalculate
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()

    # Save intermediate model
    temp_path = output_path.replace(".onnx", "_temp.onnx")
    onnx.save(model, temp_path)

    # Run onnxsim to simplify and propagate shapes
    print("  Running onnxsim to propagate shapes...")
    try:
        model = onnx.load(temp_path)
        model_simp, check = onnxsim.simplify(
            model,
            overwrite_input_shapes={new_name: [1, 3, input_size, input_size]},
        )

        if check:
            print("  onnxsim succeeded")
            onnx.save(model_simp, output_path)
        else:
            print("  onnxsim check failed, using original")
            shutil.copy(temp_path, output_path)
    except Exception as e:
        print(f"  onnxsim failed: {e}, using original")
        shutil.copy(temp_path, output_path)

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f"  Saved to: {output_path}")

    # Verify output shapes
    model = onnx.load(output_path)
    print("\n  Output shapes after fix:")
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {dims}")

    return output_path


def convert_to_tflite(onnx_path: str, output_dir: str, calib_data: np.ndarray) -> bool:
    """Convert ONNX model to TFLite INT8."""
    print(f"\nConverting to TFLite with onnx2tf...")
    print(f"  Input: {onnx_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Calibration data: {calib_data.shape}")

    # Save calibration data as NPY file
    calib_npy_path = "scrfd_calib_enhanced.npy"
    np.save(calib_npy_path, calib_data.astype(np.float32))

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            copy_onnx_input_output_names_to_tflite=True,
            output_integer_quantized_tflite=True,
            custom_input_op_name_np_data_path=[
                ["input_1", calib_npy_path, [[[[0.0, 0.0, 0.0]]]], [[[[1.0, 1.0, 1.0]]]]]
            ],
            non_verbose=True,
        )
        print("  Conversion successful!")
        return True

    except Exception as e:
        print(f"  Conversion failed: {e}")
        return False
    finally:
        if os.path.exists(calib_npy_path):
            os.remove(calib_npy_path)


def validate_model(model_path: str, calib_data: np.ndarray, onnx_path: str) -> dict:
    """Validate INT8 model against float32 outputs."""
    print(f"\nValidating: {model_path}")

    # Load INT8 model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"  Outputs: {len(output_details)}")

    for i, out in enumerate(output_details):
        shape = out['shape']
        desc = ""
        if len(shape) == 2:
            if shape[1] == 1:
                desc = "(scores)"
            elif shape[1] == 4:
                desc = "(boxes)"
            elif shape[1] == 10:
                desc = "(keypoints)"
        print(f"    [{i}] {shape} {desc}")

    # Compare with ONNX outputs on a few samples
    import onnxruntime as ort

    onnx_sess = ort.InferenceSession(onnx_path)
    onnx_input_name = onnx_sess.get_inputs()[0].name

    print("\n  Comparing INT8 vs Float32 on sample images...")
    n_samples = min(10, len(calib_data))

    # Categorize outputs by shape
    int8_outputs_by_shape = {}
    for out in output_details:
        shape_key = tuple(out['shape'])
        if shape_key not in int8_outputs_by_shape:
            int8_outputs_by_shape[shape_key] = []
        int8_outputs_by_shape[shape_key].append(out)

    correlations = {
        'scores': [],
        'boxes': [],
        'keypoints': []
    }

    for i in range(n_samples):
        # ONNX inference (NCHW)
        img_nhwc = calib_data[i:i+1]
        img_nchw = np.transpose(img_nhwc, (0, 3, 1, 2)).astype(np.float32)
        onnx_outputs = onnx_sess.run(None, {onnx_input_name: img_nchw})

        # INT8 inference
        input_scale = input_details[0]['quantization'][0]
        input_zp = input_details[0]['quantization'][1]
        int8_input = np.clip(
            np.round(img_nhwc / input_scale + input_zp),
            -128, 127
        ).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], int8_input)
        interpreter.invoke()

        # Dequantize outputs and match by shape
        for onnx_out in onnx_outputs:
            onnx_shape = onnx_out.shape
            shape_key = tuple(onnx_shape)

            if shape_key in int8_outputs_by_shape:
                for int8_out_detail in int8_outputs_by_shape[shape_key]:
                    int8_raw = interpreter.get_tensor(int8_out_detail['index'])
                    scale = int8_out_detail['quantization'][0]
                    zp = int8_out_detail['quantization'][1]
                    int8_dequant = (int8_raw.astype(np.float32) - zp) * scale

                    # Determine output type by shape
                    if len(onnx_shape) == 2:
                        if onnx_shape[1] == 1:
                            output_type = 'scores'
                        elif onnx_shape[1] == 4:
                            output_type = 'boxes'
                        elif onnx_shape[1] == 10:
                            output_type = 'keypoints'
                        else:
                            continue

                        corr = np.corrcoef(onnx_out.flatten(), int8_dequant.flatten())[0, 1]
                        if not np.isnan(corr):
                            correlations[output_type].append(corr)
                    break  # Only match first

    print("\n  Correlation (higher is better):")
    result = {}
    for name, corrs in correlations.items():
        if corrs:
            avg_corr = np.mean(corrs)
            print(f"    {name}: {avg_corr:.4f}")
            result[name] = avg_corr

    return result


def main():
    parser = argparse.ArgumentParser(description='Enhanced SCRFD Quantization')
    parser.add_argument('--onnx', type=str, default='scrfd_500m_kps.onnx',
                        help='Input ONNX model')
    parser.add_argument('--output', type=str, default='scrfd_enhanced_int8.tflite',
                        help='Output TFLite model')
    parser.add_argument('--max-calib', type=int, default=5000,
                        help='Max calibration images (default: 5000)')
    parser.add_argument('--input-size', type=int, default=160,
                        help='Input size (default: 160)')
    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  Input ONNX: {args.onnx}")
    print(f"  Output: {args.output}")
    print(f"  Input size: {args.input_size}")
    print(f"  Max calibration: {args.max_calib}")

    # Check ONNX model
    if not os.path.exists(args.onnx):
        print(f"\nError: ONNX model not found: {args.onnx}")
        return 1

    # Load calibration data from LFW
    lfw_dir = "calibration_data/lfw/lfw-deepfunneled"
    calib_images = load_lfw_images(lfw_dir, args.input_size, args.max_calib)

    if calib_images is None or len(calib_images) < 100:
        print("\nNot enough calibration images!")
        return 1

    print(f"\nCalibration data: {calib_images.shape}")
    print(f"Range: [{calib_images.min():.3f}, {calib_images.max():.3f}]")

    # Fix ONNX model for target size
    fixed_onnx = args.onnx.replace(".onnx", f"_{args.input_size}_enhanced.onnx")
    fix_onnx_for_target_size(args.onnx, fixed_onnx, args.input_size)

    # Convert to TFLite
    output_dir = "scrfd_enhanced_output"
    success = convert_to_tflite(fixed_onnx, output_dir, calib_images)

    if not success:
        print("\nConversion failed!")
        return 1

    # Find INT8 model
    tflite_path = None
    for pattern in ["*full_integer_quant.tflite", "*integer_quant.tflite", "*.tflite"]:
        matches = list(Path(output_dir).glob(pattern))
        if matches:
            tflite_path = matches[0]
            break

    if tflite_path:
        shutil.copy(tflite_path, args.output)
        print(f"\nCopied INT8 model to: {args.output}")
    else:
        print("\nNo TFLite model found!")
        return 1

    # Validate
    correlations = validate_model(args.output, calib_images[:100], fixed_onnx)

    # Cleanup
    if os.path.exists(fixed_onnx):
        os.remove(fixed_onnx)

    print("\n" + "=" * 60)
    print("Conversion successful!")
    print("=" * 60)
    print(f"\nOutput: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1024:.1f} KB")

    if correlations:
        avg_corr = np.mean(list(correlations.values()))
        print(f"Average correlation: {avg_corr:.4f}")
        if avg_corr < 0.95:
            print("Warning: Correlation is low, quantization might affect accuracy")

    print("\nNext step: Compile with Vela and test on device")

    return 0


if __name__ == "__main__":
    sys.exit(main())
