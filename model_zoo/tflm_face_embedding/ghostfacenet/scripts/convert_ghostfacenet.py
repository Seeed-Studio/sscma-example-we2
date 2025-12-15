#!/usr/bin/env python3
"""
Convert GhostFaceNet-0.5 from Keras H5 to TFLite INT8.

GhostFaceNet uses Ghost modules for efficient face embedding generation.
Ghost modules create cheap feature maps through linear transformations,
significantly reducing computation while maintaining accuracy.

Model source: https://github.com/HamadYA/GhostFaceNets

GhostFaceNet-0.5 specifications:
    - Width multiplier: 0.5 (halved channel dimensions)
    - Parameters: ~722K
    - Input: 112x112x3 RGB
    - Output: 512-dimensional embedding

Conversion pipeline:
    1. Download GhostFaceNet H5 model from GitHub (if not present)
    2. Convert mixed precision (float16) model to float32
    3. Convert to TFLite with INT8 quantization
    4. Compile with Vela for Ethos-U55 NPU

Output specifications:
    - Input: [1, 112, 112, 3] INT8 (NHWC, normalized to [-1,1])
    - Output: [1, 512] INT8 (512-dimensional face embedding)

IMPORTANT: ArcFace/InsightFace models expect [-1, 1] input range, NOT [0, 1]!
    Preprocessing: (pixel - 127.5) / 128.0
    This maps uint8 [0, 255] to float [-1, 1]

Usage:
    # First, prepare calibration data:
    python prepare_calibration_data.py

    # Then convert:
    python convert_ghostfacenet.py

    # Or with custom options:
    python convert_ghostfacenet.py --h5 my_model.h5 --output my_model_int8.tflite
"""

import os
import sys
import argparse
import shutil
import numpy as np
from pathlib import Path

print("=" * 60)
print("GhostFaceNet to TFLite INT8 Converter")
print("=" * 60)

# Check dependencies
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Please install tensorflow: pip install tensorflow")
    sys.exit(1)

# Use tf_keras (legacy Keras 2) for loading old H5 models
# TF 2.16+ uses Keras 3 which has incompatible serialization
try:
    import tf_keras
    print(f"tf_keras version: {tf_keras.__version__}")
except ImportError:
    print("Please install tf-keras: pip install tf-keras")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)


# Paths - relative to script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_H5 = str(SCRIPT_DIR / "GN_W0.5_S2_ArcFace_epoch16.h5")
DEFAULT_CALIB_DIR = str(SCRIPT_DIR.parent.parent / "calibration_data" / "emb_112")
GHOSTFACENET_URL = "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.0/GN_W0.5_S2_ArcFace_epoch16.h5"


def download_model_if_needed(h5_path: str) -> bool:
    """Download GhostFaceNet H5 if not present."""
    if os.path.exists(h5_path):
        print(f"Model found: {h5_path}")
        return True

    print(f"\nDownloading GhostFaceNet model...")
    print(f"  URL: {GHOSTFACENET_URL}")

    try:
        import urllib.request

        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r  Progress: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(GHOSTFACENET_URL, h5_path, progress)
        print(f"\n  Downloaded: {h5_path}")
        return True

    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def load_calibration_images(calib_dir: str, input_size: int = 112, max_images: int = 1000) -> list:
    """Load calibration images for INT8 quantization."""
    calib_path = Path(calib_dir)

    if not calib_path.exists():
        print(f"Calibration directory not found: {calib_dir}")
        return []

    image_files = sorted(calib_path.glob("*.jpg"))[:max_images]
    if not image_files:
        print(f"No images found in {calib_dir}")
        return []

    print(f"Loading {len(image_files)} calibration images...")
    images = []

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (input_size, input_size):
            img = cv2.resize(img, (input_size, input_size))
        # ArcFace/InsightFace standard normalization: [-1, 1]
        # (pixel - 127.5) / 128.0 maps uint8 [0, 255] to float [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 128.0
        images.append(img)

    print(f"Loaded {len(images)} images")
    return images


def load_and_convert_to_float32(h5_path: str):
    """
    Load Keras H5 model and convert from mixed precision to float32.

    Revised Strategy: Deep Precision Reset
    Instead of clone_model (which often preserves policy attributes), we:
    1. Extract the model configuration (JSON/Dict).
    2. Recursively scrub 'dtype' and 'dtype_policy' from the config.
    3. Reconstruct a fresh model from the clean config.
    4. Inject weights cast to float32.

    Note: Uses tf_keras (legacy Keras 2) for loading old H5 models.
    TensorFlow 2.16+ ships with Keras 3 which has incompatible DTypePolicy serialization.
    """
    from tf_keras import mixed_precision

    print(f"\nLoading Keras model: {h5_path}")

    # Step 0: Ensure global policy is float32 before doing anything
    # This ensures new layers created during reconstruction default to float32
    mixed_precision.set_global_policy('float32')

    # Step 1: Load original model using tf_keras (legacy Keras 2)
    # We load it as-is first to get the weights and structure
    try:
        original_model = tf_keras.models.load_model(h5_path, compile=False)
    except TypeError:
        # Sometimes custom objects are needed, but usually generic loading works for structure
        original_model = tf_keras.models.load_model(h5_path, compile=False)

    print(f"  Input shape: {original_model.input_shape}")
    print(f"  Output shape: {original_model.output_shape}")
    print(f"  Parameters: {original_model.count_params():,}")
    print(f"  Original input dtype: {original_model.input.dtype}")

    # Step 2: Clean the Configuration
    print("  Reconstructing model architecture with strict float32 policy...")

    config = original_model.get_config()

    def clean_config_dtype(cfg):
        """Recursively remove/reset dtype info in the config dictionary."""
        if isinstance(cfg, dict):
            # Force dtype to float32
            if 'dtype' in cfg:
                cfg['dtype'] = 'float32'

            # Reset policy to standard float32
            if 'dtype_policy' in cfg:
                if isinstance(cfg['dtype_policy'], dict):
                    cfg['dtype_policy'] = {'name': 'float32'}
                else:
                    cfg['dtype_policy'] = 'float32'

            # Recursively check all dictionary values
            for key, value in cfg.items():
                clean_config_dtype(value)

        elif isinstance(cfg, list):
            # Recursively check list items (like layers list)
            for item in cfg:
                clean_config_dtype(item)

    # Apply the cleaning
    clean_config_dtype(config)

    # Step 3: Rebuild Model from Clean Config (using tf_keras)
    try:
        # Standard functional/sequential model reconstruction
        new_model = tf_keras.Model.from_config(config)
    except Exception as e:
        print(f"  Note: Model.from_config failed ({e}), trying generic model_from_config...")
        new_model = tf_keras.models.model_from_config(config)

    # Step 4: Transfer and Cast Weights
    print("  Migrating weights (float16 -> float32)...")

    original_weights = original_model.get_weights()
    # Explicitly cast every single weight tensor to float32
    new_weights = [w.astype(np.float32) for w in original_weights]

    # Load weights into new structure
    new_model.set_weights(new_weights)

    print(f"  New model input dtype: {new_model.input.dtype}")

    # Verify conversion
    f16_layers = []
    for layer in new_model.layers:
        # Check both the compute dtype and the variable dtype
        dtype_pol = getattr(layer, 'dtype_policy', None)
        policy_name = dtype_pol.name if dtype_pol else "unknown"

        if 'float16' in policy_name or 'float16' in str(layer.dtype):
            f16_layers.append(layer.name)

    if f16_layers:
        print(f"  WARNING: {len(f16_layers)} layers still have float16 traces! (Should be 0)")
        print(f"  First culprit: {f16_layers[0]}")
    else:
        print("  SUCCESS: All layers strictly converted to float32 policy")

    return new_model


def convert_to_tflite_int8(
    model: tf.keras.Model,
    output_path: str,
    calibration_images: list
) -> bool:
    """
    Convert Keras model to TFLite INT8 with fixed input shape.

    IMPORTANT: We use a concrete function with fixed input shape to avoid
    dynamic shape operations that would fall back to CPU on Ethos-U55.
    """
    print("\nConverting to TFLite INT8...")
    print(f"  Calibration images: {len(calibration_images)}")

    # Create a concrete function with fixed input shape [1, 112, 112, 3]
    # This is crucial for 100% NPU support - it eliminates dynamic shape ops
    print("  Creating concrete function with fixed input shape [1, 112, 112, 3]...")

    @tf.function(input_signature=[tf.TensorSpec([1, 112, 112, 3], tf.float32)])
    def fixed_shape_model(x):
        return model(x, training=False)

    # Get concrete function
    concrete_func = fixed_shape_model.get_concrete_function()

    # Create converter from concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # Enable optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Create representative dataset generator
    # Use all calibration images for better quantization statistics
    num_calib = min(len(calibration_images), 500)
    print(f"  Using {num_calib} calibration images for quantization...")

    def representative_dataset():
        for img in calibration_images[:num_calib]:
            yield [np.expand_dims(img, axis=0).astype(np.float32)]

    converter.representative_dataset = representative_dataset

    # Configure for full INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("  This may take a few minutes...")

    try:
        tflite_model = converter.convert()
        print("  INT8 conversion successful!")

        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        model_size_kb = len(tflite_model) / 1024
        print(f"\n  Model saved to: {output_path}")
        print(f"  Model size: {model_size_kb:.1f} KB ({model_size_kb/1024:.2f} MB)")

        return True

    except Exception as e:
        print(f"\n  INT8 conversion failed: {e}")
        print("\n  Trying with mixed INT8/float fallback...")

        # Fallback: allow some float ops
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback for unsupported ops
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        try:
            tflite_model = converter.convert()
            print("  Mixed INT8 conversion successful!")

            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            model_size_kb = len(tflite_model) / 1024
            print(f"\n  Model saved to: {output_path}")
            print(f"  Model size: {model_size_kb:.1f} KB")

            return True

        except Exception as e2:
            print(f"  Conversion failed: {e2}")
            return False


def validate_tflite_model(model_path: str) -> dict:
    """Validate the converted TFLite model."""
    print(f"\nValidating: {model_path}")

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except (RuntimeError, ValueError) as e:
        error_str = str(e).lower()
        if "ethos-u" in error_str or "invalidly specified" in error_str:
            print("  Note: Vela-compiled model cannot be validated on CPU")
            print("  (This is expected - the model contains ethos-u custom ops)")
            file_size = os.path.getsize(model_path)
            print(f"  File size: {file_size / 1024:.1f} KB")
            return {'file_size': file_size, 'vela_model': True}
        raise

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Inputs: {len(input_details)}")
    for inp in input_details:
        qp = inp.get('quantization_parameters', {})
        scales = qp.get('scales', [])
        zps = qp.get('zero_points', [])
        quant_info = f"scale={scales[0]:.6f}, zp={zps[0]}" if len(scales) > 0 else "not quantized"
        print(f"    {inp['shape']} {inp['dtype']} ({quant_info})")

    print(f"  Outputs: {len(output_details)}")
    for i, out in enumerate(output_details):
        shape = out['shape']
        # Identify embedding (512-dim)
        if len(shape) >= 2 and shape[-1] == 512:
            desc = "(embedding)"
        else:
            desc = ""
        qp = out.get('quantization_parameters', {})
        scales = qp.get('scales', [])
        quant_info = f"scale={scales[0]:.6f}" if len(scales) > 0 else "not quantized"
        print(f"    [{i}] {shape} {out['dtype']} {desc} ({quant_info})")

    # Test inference with correct input type
    inp = input_details[0]
    if inp['dtype'] == np.int8:
        test_input = np.random.randint(-128, 127, size=inp['shape']).astype(np.int8)
    elif inp['dtype'] == np.uint8:
        test_input = np.random.randint(0, 255, size=inp['shape']).astype(np.uint8)
    else:
        test_input = np.random.rand(*inp['shape']).astype(np.float32)

    interpreter.set_tensor(inp['index'], test_input)
    interpreter.invoke()

    embedding = interpreter.get_tensor(output_details[0]['index'])
    print(f"\n  Embedding output: {embedding.shape}")
    print("  Inference test: PASSED")

    # Check for potential issues
    issues = []
    if inp['dtype'] != np.int8:
        issues.append(f"WARNING: Input is {inp['dtype']}, expected int8")
    if inp['shape'].tolist() != [1, 112, 112, 3]:
        issues.append(f"WARNING: Input shape is {inp['shape']}, expected [1, 112, 112, 3]")
    if output_details[0]['shape'].tolist() != [1, 512]:
        issues.append(f"WARNING: Output shape is {output_details[0]['shape']}, expected [1, 512]")

    if issues:
        print("\n  Issues detected:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  Model format is correct")

    return {
        'input_shape': input_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'num_outputs': len(output_details),
        'file_size': os.path.getsize(model_path),
        'issues': issues
    }


def compile_with_vela(input_path: str, output_dir: str = ".") -> str:
    """Compile with Vela for Ethos-U55 NPU."""
    import subprocess

    print(f"\nCompiling with Vela...")

    vela_path = shutil.which("vela")
    if not vela_path:
        print("  Vela not found. Please run manually:")
        print(f"    vela {input_path} --accelerator-config ethos-u55-64 --optimise Performance")
        return None

    cmd = [
        "vela", input_path,
        "--accelerator-config", "ethos-u55-64",
        "--optimise", "Performance",
        "--output-dir", output_dir
    ]

    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print summary (last part of output)
    output_lines = result.stdout.strip().split('\n')
    summary_start = -1
    for i, line in enumerate(output_lines):
        if 'Network summary' in line:
            summary_start = i
            break

    if summary_start >= 0:
        print("\n  " + "\n  ".join(output_lines[summary_start:]))

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None

    base = Path(input_path).stem
    output = os.path.join(output_dir, f"{base}_vela.tflite")

    if os.path.exists(output):
        print(f"\n  Output: {output}")
        return output

    return None


def test_embedding_similarity(model_path: str, calib_dir: str) -> None:
    """
    Test embedding quality with real face images.

    Args:
        model_path: Path to TFLite model
        calib_dir: Calibration images directory
    """
    print("\n" + "-" * 40)
    print("Testing embedding similarity...")
    print("-" * 40)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Load test images
    calib_path = Path(calib_dir)
    test_images = sorted(calib_path.glob("*.jpg"))[:5]

    if len(test_images) < 2:
        print("  Not enough test images")
        return

    embeddings = []

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))

        # Prepare input based on dtype
        # ArcFace/InsightFace standard: normalize to [-1, 1]
        if input_details['dtype'] == np.int8:
            # Quantize: (value / scale) + zero_point
            # where value is in [-1, 1] range (ArcFace standard)
            quant = input_details.get('quantization_parameters', {})
            scale = quant.get('scales', [1.0/128.0])[0]
            zero_point = quant.get('zero_points', [0])[0]
            # Normalize to [-1, 1] first, then quantize
            normalized = (img.astype(np.float32) - 127.5) / 128.0
            input_data = (normalized / scale + zero_point).astype(np.int8)
        else:
            # Float model: normalize to [-1, 1]
            input_data = (img.astype(np.float32) - 127.5) / 128.0

        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()

        embedding = interpreter.get_tensor(output_details['index'])[0].astype(np.float32)

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        embeddings.append(embedding)

    # Compute pairwise cosine similarities
    print("\n  Pairwise cosine similarities:")
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j])
            print(f"    Image {i} vs {j}: {sim:.4f}")

    print("\n  Note: Similarity > 0.5 typically indicates same person")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GhostFaceNet H5 to TFLite INT8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_ghostfacenet.py
    python convert_ghostfacenet.py --download
    python convert_ghostfacenet.py --h5 my_model.h5 --output my_model_int8.tflite
        """
    )
    parser.add_argument('--h5', type=str, default=DEFAULT_H5,
                        help=f'Path to GhostFaceNet H5 model (default: {DEFAULT_H5})')
    parser.add_argument('--output', type=str, default='ghostfacenet_fixed_int8.tflite',
                        help='Output TFLite model path')
    parser.add_argument('--calib-dir', type=str, default=DEFAULT_CALIB_DIR,
                        help='Calibration images directory')
    parser.add_argument('--download', action='store_true',
                        help='Download model if not present')
    parser.add_argument('--skip-vela', action='store_true',
                        help='Skip Vela compilation')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip embedding similarity test')
    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  H5: {args.h5}")
    print(f"  Output: {args.output}")
    print(f"  Calibration: {args.calib_dir}")

    # Step 1: Download if needed
    if not download_model_if_needed(args.h5):
        return 1

    # Step 2: Load calibration images
    calib_images = load_calibration_images(args.calib_dir)
    if len(calib_images) == 0:
        print("Using random calibration data...")
        calib_images = [np.random.rand(112, 112, 3).astype(np.float32) for _ in range(100)]

    # Step 3: Load and convert model to float32
    model = load_and_convert_to_float32(args.h5)

    # Step 4: Convert to TFLite INT8
    if not convert_to_tflite_int8(model, args.output, calib_images):
        print("\nFailed to convert to TFLite!")
        return 1

    # Step 5: Validate
    model_info = validate_tflite_model(args.output)

    # Step 6: Test embedding similarity
    if not args.skip_test and len(calib_images) >= 2:
        try:
            test_embedding_similarity(args.output, args.calib_dir)
        except Exception as e:
            print(f"  Similarity test skipped: {e}")

    # Step 7: Compile with Vela
    if not args.skip_vela:
        vela_output = compile_with_vela(args.output)
        if vela_output:
            validate_tflite_model(vela_output)

    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("=" * 60)
    print(f"\nOutput: {args.output}")
    print(f"Size: {model_info['file_size'] / 1024:.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
