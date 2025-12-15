#!/usr/bin/env python3
"""
Convert SCRFD_500M_KPS ONNX to TFLite INT8.

SCRFD (Sample and Computation Redistribution for Face Detection) is an efficient
anchor-free face detector. The 500M variant has ~500K parameters and outputs
bounding boxes + 5 facial keypoints.

Model source: InsightFace (buffalo_sc model pack)
    https://github.com/deepinsight/insightface

Conversion pipeline:
    1. Download SCRFD ONNX model via insightface
    2. Fix ONNX model (rename inputs, fix shapes for target resolution)
    3. Simplify with onnxsim
    4. Convert to TFLite using onnx2tf
    5. Quantize to INT8 with calibration data

Output specifications (for 160x160 input):
    - Input: [1, 160, 160, 3] INT8 (NHWC, normalized to [0,1])
    - Outputs (3 scales):
        - Stride 8:  scores [800,1], boxes [800,4], keypoints [800,10]
        - Stride 16: scores [200,1], boxes [200,4], keypoints [200,10]
        - Stride 32: scores [50,1],  boxes [50,4],  keypoints [50,10]

Usage:
    # First, prepare calibration data:
    python prepare_calibration_data.py

    # Then convert:
    python convert_scrfd.py

    # Or with custom options:
    python convert_scrfd.py --input-size 320 --output scrfd_320_int8.tflite
"""

import os
import sys
import argparse
import shutil
import numpy as np
from pathlib import Path

print("=" * 60)
print("SCRFD to TFLite INT8 Converter")
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
    print("Please install tensorflow: pip install tensorflow")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)


def download_scrfd_model(output_path: str = "scrfd_500m_kps.onnx") -> str:
    """
    Download SCRFD model using insightface library.

    Args:
        output_path: Path to save the ONNX model

    Returns:
        Path to downloaded model, or None if failed
    """
    if os.path.exists(output_path):
        print(f"SCRFD model found: {output_path}")
        return output_path

    print("\nDownloading SCRFD model via InsightFace...")

    try:
        from insightface.model_zoo import model_zoo

        # Download buffalo_sc model pack (contains SCRFD detector)
        print("  Downloading buffalo_sc model pack...")
        model = model_zoo.get_model('buffalo_sc')

        # Find the cached ONNX file
        home = Path.home()
        cache_dirs = [
            home / ".insightface" / "models" / "buffalo_sc",
            home / ".insightface" / "models",
        ]

        onnx_file = None
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                for f in cache_dir.rglob("*.onnx"):
                    if "det" in f.name.lower() or "scrfd" in f.name.lower():
                        onnx_file = f
                        break
            if onnx_file:
                break

        if onnx_file:
            shutil.copy(onnx_file, output_path)
            print(f"  Model copied to: {output_path}")
            print(f"  Original location: {onnx_file}")
            return output_path
        else:
            print("  Model downloaded but ONNX file not found in cache.")
            print(f"  Please check: {home / '.insightface' / 'models'}")
            return None

    except ImportError:
        print("  InsightFace not installed. Please install: pip install insightface")
        return None
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def load_calibration_images(calib_dir: str, input_size: int, max_images: int = 100) -> np.ndarray:
    """
    Load calibration images for INT8 quantization.

    Args:
        calib_dir: Directory containing calibration images
        input_size: Target input size (e.g., 160)
        max_images: Maximum number of images to load

    Returns:
        Numpy array of shape [N, H, W, 3] in NHWC format, float32, normalized to [0,1]
    """
    images = []
    calib_path = Path(calib_dir)

    if not calib_path.exists():
        print(f"Calibration directory not found: {calib_dir}")
        return None

    image_files = sorted(calib_path.glob("*.jpg"))[:max_images]
    print(f"Loading {len(image_files)} calibration images...")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)

    print(f"Loaded {len(images)} calibration images")
    return np.array(images) if images else None


def fix_onnx_for_target_size(onnx_path: str, output_path: str, input_size: int = 160) -> str:
    """
    Fix ONNX model for target input size.

    The original SCRFD model has dynamic input shapes and output shape metadata
    for 640x640 input. This function:
    1. Renames problematic input names (input.1 -> input_1)
    2. Sets fixed input shape
    3. Updates output shape metadata for target size
    4. Runs onnxsim to propagate shapes

    Args:
        onnx_path: Path to original ONNX model
        output_path: Path to save fixed model
        input_size: Target input size (default: 160)

    Returns:
        Path to fixed ONNX model
    """
    print(f"\nFixing ONNX model for {input_size}x{input_size} input...")

    model = onnx.load(onnx_path)

    # Calculate output anchor counts for target size
    # Original 640x640: stride 8 -> 80x80x2=12800, stride 16 -> 40x40x2=3200, stride 32 -> 20x20x2=800
    # For input_size: stride 8 -> (size/8)^2*2, etc.
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

    # Rename input (replace dots with underscores for TFLite compatibility)
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
    """
    Convert ONNX model to TFLite INT8 using onnx2tf.

    Args:
        onnx_path: Path to fixed ONNX model
        output_dir: Directory to save TFLite models
        calib_data: Calibration data array [N, H, W, 3] NHWC float32

    Returns:
        True if successful
    """
    print(f"\nConverting to TFLite with onnx2tf...")
    print(f"  Input: {onnx_path}")
    print(f"  Output dir: {output_dir}")

    # Save calibration data as NPY file
    calib_npy_path = "scrfd_calib_nhwc.npy"
    np.save(calib_npy_path, calib_data.astype(np.float32))
    print(f"  Calibration data: {calib_data.shape}")

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    try:
        # Use onnx2tf with INT8 quantization
        # mean=0, std=1 because data is already normalized to [0,1]
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
        # Cleanup calibration file
        if os.path.exists(calib_npy_path):
            os.remove(calib_npy_path)


def validate_tflite_model(model_path: str) -> dict:
    """
    Validate the converted TFLite model.

    Args:
        model_path: Path to TFLite model

    Returns:
        Dictionary with model info
    """
    print(f"\nValidating: {model_path}")

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
                desc = "(confidence scores)"
            elif shape[1] == 4:
                desc = "(bounding boxes)"
            elif shape[1] == 10:
                desc = "(keypoints)"
        print(f"    [{i}] {shape} {desc}")

    # Test inference
    test_input = np.random.randint(-128, 127, size=input_details[0]['shape']).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    print("  Inference test: PASSED")

    return {
        'input_shape': input_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'num_outputs': len(output_details),
        'file_size': os.path.getsize(model_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert SCRFD ONNX to TFLite INT8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_scrfd.py
    python convert_scrfd.py --input-size 320
    python convert_scrfd.py --onnx my_scrfd.onnx --output my_scrfd_int8.tflite
        """
    )
    parser.add_argument('--onnx', type=str, default='scrfd_500m_kps.onnx',
                        help='Input ONNX model (default: scrfd_500m_kps.onnx)')
    parser.add_argument('--output', type=str, default='scrfd_500m_kps_int8.tflite',
                        help='Output TFLite model (default: scrfd_500m_kps_int8.tflite)')
    parser.add_argument('--input-size', type=int, default=160,
                        help='Input size (default: 160)')
    parser.add_argument('--calib-dir', type=str, default='calibration_data/fd_160',
                        help='Calibration images directory')
    parser.add_argument('--download', action='store_true',
                        help='Download model if not present')
    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  Input ONNX: {args.onnx}")
    print(f"  Output TFLite: {args.output}")
    print(f"  Input size: {args.input_size}x{args.input_size}")
    print(f"  Calibration dir: {args.calib_dir}")

    # Step 1: Get ONNX model
    onnx_path = args.onnx
    if not os.path.exists(onnx_path) or args.download:
        onnx_path = download_scrfd_model(args.onnx)
        if onnx_path is None:
            print("\nError: Could not get SCRFD model.")
            print("Please download manually from InsightFace or provide --onnx path.")
            return 1

    # Step 2: Load calibration data
    calib_images = load_calibration_images(args.calib_dir, args.input_size)
    if calib_images is None or len(calib_images) == 0:
        print("\nNo calibration images found. Using random data (not recommended).")
        print("For better quantization, run: python prepare_calibration_data.py")
        calib_images = np.random.rand(100, args.input_size, args.input_size, 3).astype(np.float32)

    # Step 3: Fix ONNX model for target size
    fixed_onnx = args.onnx.replace(".onnx", f"_{args.input_size}_fixed.onnx")
    fix_onnx_for_target_size(onnx_path, fixed_onnx, args.input_size)

    # Step 4: Convert to TFLite
    output_dir = "scrfd_tf_output"
    success = convert_to_tflite(fixed_onnx, output_dir, calib_images)

    if not success:
        print("\n" + "=" * 60)
        print("Conversion failed!")
        print("=" * 60)
        return 1

    # Step 5: Find and copy INT8 model
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
        print("\nError: No TFLite model found in output directory")
        return 1

    # Step 6: Validate
    model_info = validate_tflite_model(args.output)

    # Cleanup
    if os.path.exists(fixed_onnx):
        os.remove(fixed_onnx)

    print("\n" + "=" * 60)
    print("Conversion successful!")
    print("=" * 60)
    print(f"\nOutput: {args.output}")
    print(f"Size: {model_info['file_size'] / 1024:.1f} KB")
    print(f"Input: {model_info['input_shape']} {model_info['input_dtype']}")
    print(f"Outputs: {model_info['num_outputs']} tensors")
    print("\nNext step: python convert_ghostfacenet.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
