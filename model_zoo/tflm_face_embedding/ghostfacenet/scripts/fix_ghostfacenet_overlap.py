#!/usr/bin/env python3
"""
Fix GhostFaceNet Vela tensor overlap by adding an NPU-only dummy branch.

Problem:
    Vela's HillClimb memory allocator assigns output tensor within input tensor's
    address range (overlap), causing UsageFault when NPU tries to read input
    while simultaneously writing to output.

    Memory layout issue:
    Address              Content
    0x3405F000  ──┬── Input start (37632 bytes)
                  │
    0x3405F100  ──┼── Output start (512 bytes) ← Problem: Output inside Input!
                  │
    0x34068340  ──┴── Input end

Solution (NPU-only branch):
    Add a dummy branch using NPU-supported operators that:
    1. Takes the same input as the main network
    2. Uses DepthwiseConv2D with identity weights (1x1 kernel)
    3. Reduces spatial dimensions with AveragePooling2D
    4. Outputs a small tensor as second model output

    This creates two consumers of the input tensor, both on NPU.
    Vela's Live Range analysis will see that input must stay alive
    until both branches complete, preventing memory overlap.

    Key: All ops must be NPU-compatible to ensure proper liveness tracking.
    - DepthwiseConv2D: NPU supported ✓
    - AveragePooling2D: NPU supported ✓
    - NO Mean/ReduceMean (falls back to CPU)

Usage:
    python fix_ghostfacenet_overlap.py

    # After running, compile with Vela:
    vela ghostfacenet_nooverlap_int8.tflite --accelerator-config ethos-u55-64 --optimise Performance
"""

import os
import sys
import argparse
import shutil
import numpy as np
from pathlib import Path

print("=" * 60)
print("GhostFaceNet Vela Overlap Fix (Direct Keras approach)")
print("=" * 60)

# Check dependencies
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


# Default paths
DEFAULT_H5 = "GN_W0.5_S2_ArcFace_epoch16.h5"
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


def load_calibration_images(calib_dir: str, input_size: int = 112, max_images: int = 200) -> list:
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
        img = img.astype(np.float32) / 255.0
        images.append(img)

    print(f"Loaded {len(images)} images")
    return images


def load_and_convert_to_float32(h5_path: str) -> tf.keras.Model:
    """
    Load Keras H5 model and convert from mixed precision to float32.

    GhostFaceNet uses mixed_float16 policy which causes issues during
    TFLite conversion. This function uses clone_model to rebuild the model
    under float32 policy.

    Method:
    1. Load original model (may be f16)
    2. Set global policy to float32
    3. Clone model (new model uses float32 policy)
    4. Copy weights from original to cloned model
    """
    print(f"\nLoading Keras model: {h5_path}")

    # Step 1: Load original model first (with its original precision)
    original_model = tf.keras.models.load_model(h5_path, compile=False)

    print(f"  Input shape: {original_model.input_shape}")
    print(f"  Output shape: {original_model.output_shape}")
    print(f"  Parameters: {original_model.count_params():,}")
    print(f"  Original input dtype: {original_model.input.dtype}")

    # Step 2: Set global policy to float32 AFTER loading original
    tf.keras.mixed_precision.set_global_policy('float32')

    # Step 3: Clone model - this creates a new model with float32 policy
    print("  Cloning model with float32 policy...")

    def clone_function(layer):
        """Clone function that forces float32 dtype."""
        config = layer.get_config()
        # Remove any dtype policy settings
        if 'dtype' in config:
            config['dtype'] = 'float32'
        return layer.__class__.from_config(config)

    cloned_model = tf.keras.models.clone_model(
        original_model,
        clone_function=clone_function
    )

    # Step 4: Copy weights from original to cloned model
    print("  Copying weights (converting to float32)...")
    for cloned_layer, original_layer in zip(cloned_model.layers, original_model.layers):
        try:
            weights = original_layer.get_weights()
            if weights:
                # Convert all weights to float32
                weights_f32 = [w.astype(np.float32) for w in weights]
                cloned_layer.set_weights(weights_f32)
        except Exception as e:
            print(f"    Warning: {original_layer.name}: {e}")

    print(f"  Cloned model input dtype: {cloned_model.input.dtype}")

    # Verify conversion
    f16_layers = []
    for layer in cloned_model.layers:
        if hasattr(layer, 'dtype'):
            if 'float16' in str(layer.dtype):
                f16_layers.append(layer.name)

    if f16_layers:
        print(f"  Warning: {len(f16_layers)} layers still have float16 dtype")
    else:
        print("  ✓ All layers converted to float32")

    return cloned_model


def add_dummy_lifetime_node(model: tf.keras.Model) -> tf.keras.Model:
    """
    Add dependency chain to prevent Vela memory overlap.

    Problem Analysis:
    - Input tensor: [1, 112, 112, 3] = 37632 bytes
    - Embedding output: [1, 512] = 512 bytes
    - Vela places embedding inside input range because input liveness ends early

    PREVIOUS FAILED ATTEMPTS:
    1. Output2 = Input (passthrough) -> u55-64 generates illegal DMA → UsageFault
    2. Input * 0 (zero multiply) -> Gets optimized away → overlap still happens
    3. Input + Input -> Anchor doesn't depend on Embedding, still allows overlap

    FINAL SOLUTION - Dependency Chain Injection:
    - Output1 = Embedding (512 bytes) - the useful output
    - Output2 = Input + Zero(from Embedding) - anchor that depends on BOTH

    The key insight:
    - Anchor = Input + reduce_sum(Embedding * 0)
    - Anchor depends on Embedding (via the zero multiply)
    - Embedding depends on Input (via the network)
    - Therefore: Input must stay alive until Embedding is computed
    - And: Embedding cannot be placed in Input's memory!

    Vela's reasoning:
    "To compute Anchor, I need Input AND Embedding's child (Zero)."
    "Embedding is computed last in the main path."
    "So Input must stay alive until the very end, waiting to add with Zero."
    "Conclusion: Embedding CANNOT overwrite Input's memory!"

    C++ code only needs to read output[0] (embedding), can ignore output[1] (anchor).
    """
    print("\nAdding dependency chain to force memory separation...")
    print("  Strategy: 'Dependency Chain Injection'")

    input_tensor = model.input      # [1, 112, 112, 3]
    embedding_output = model.output  # [1, 512]

    # ===== DEPENDENCY CHAIN STRATEGY (ALL NPU) =====
    # Goal: Anchor depends on BOTH Input AND Embedding
    # All ops must be NPU-supported (no reduce_sum which falls to CPU)

    # Step A: Embedding * 0 -> keeps dependency but produces zeros [1, 512]
    # MUL op is NPU supported
    zero_emb = tf.keras.layers.Lambda(
        lambda x: x * 0.0,
        name='zero_from_embedding'
    )(embedding_output)

    # Step B: Reshape zero_emb to [1, 1, 1, 512] for broadcasting
    # RESHAPE is NPU supported
    zero_reshaped = tf.keras.layers.Reshape(
        (1, 1, 512),
        name='zero_reshape'
    )(zero_emb)

    # Step C: Tile to match spatial dims [1, 112, 112, 512]
    # But tiling may not be NPU supported. Instead, use Conv2D or just slice.
    #
    # Better approach: Take just the first 3 channels from zero_emb
    # zero_emb is [1, 512], we need [1, 1, 1, 3] to broadcast with [1, 112, 112, 3]
    zero_slice = tf.keras.layers.Lambda(
        lambda x: tf.reshape(x[:, :3], [1, 1, 1, 3]),
        name='zero_slice_3ch'
    )(zero_emb)

    # Step D: Input + Zero_Slice = Input (numerically)
    # ADD with broadcast [1,112,112,3] + [1,1,1,3] is NPU supported
    anchor_output = tf.keras.layers.Add(
        name='locked_anchor'
    )([input_tensor, zero_slice])

    # Create model with DUAL outputs
    # Output[0] = embedding (512 bytes) - use this in C++
    # Output[1] = anchor (37KB) - ignore in C++, just locks memory
    new_model = tf.keras.Model(
        inputs=input_tensor,
        outputs=[embedding_output, anchor_output],
        name='ghostfacenet_dependency_chain'
    )

    print(f"  Dependency chain: Input -> Embedding -> Zero -> Anchor")
    print(f"  Output[0]: Embedding {embedding_output.shape} (use this)")
    print(f"  Output[1]: Anchor {anchor_output.shape} (Input + 0_from_Embedding)")
    print(f"  Input: {input_tensor.shape}")
    print(f"  Effect: Anchor needs Embedding, so Input lives until Embedding done")
    print(f"  Result: Embedding CANNOT be placed in Input's memory region")
    print(f"  Note: C++ code reads output(0) only, ignores output(1)")

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
    def representative_dataset():
        for img in calibration_images[:100]:
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
    embedding_idx = -1
    ghost_idx = -1
    for i, out in enumerate(output_details):
        shape = out['shape']
        # Identify embedding (512-dim) vs ghost (1-dim)
        if len(shape) >= 2 and shape[-1] == 512:
            desc = "(embedding - use this)"
            embedding_idx = i
        elif len(shape) >= 2 and shape[-1] == 1:
            desc = "(ghost - ignore in C++)"
            ghost_idx = i
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

    # Get embedding output
    if embedding_idx >= 0:
        embedding = interpreter.get_tensor(output_details[embedding_idx]['index'])
        print(f"\n  Embedding output: {embedding.shape}")
    print("  Inference test: PASSED")

    # Check for potential issues
    issues = []
    if inp['dtype'] != np.int8:
        issues.append(f"WARNING: Input is {inp['dtype']}, expected int8")
    if inp['shape'].tolist() != [1, 112, 112, 3]:
        issues.append(f"WARNING: Input shape is {inp['shape']}, expected [1, 112, 112, 3]")
    # Single output is now expected (embedding via dummy_add)
    if embedding_idx < 0:
        issues.append("WARNING: No embedding output (512-dim) found")

    if issues:
        print("\n  Issues detected:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  ✓ Model has correct format with lifetime extension (single output)")

    return {
        'input_shape': input_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'num_outputs': len(output_details),
        'file_size': os.path.getsize(model_path),
        'issues': issues
    }


def compile_with_vela(input_path: str) -> str:
    """Compile with Vela.

    The model has dual outputs (embedding + anchor) which prevents Vela from
    overlapping embedding with input memory.

    Uses --disable-cascading as a safety measure for u55-64 stability.
    """
    import subprocess

    print(f"\nCompiling with Vela...")

    vela_path = shutil.which("vela")
    if not vela_path:
        print("  Vela not found. Please run manually:")
        print(f"    vela {input_path} --accelerator-config ethos-u55-64 --optimise Performance --disable-cascading")
        return None

    cmd = [
        "vela", input_path,
        "--accelerator-config", "ethos-u55-64",
        "--optimise", "Performance",
        "--disable-cascading",  # Safety measure for u55-64 stability
        "--output-dir", "."
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
    output = f"{base}_vela.tflite"

    if os.path.exists(output):
        print(f"\n  Output: {output}")
        return output

    return None


def main():
    parser = argparse.ArgumentParser(description='Fix GhostFaceNet Vela tensor overlap')
    parser.add_argument('--h5', type=str, default=DEFAULT_H5,
                        help=f'GhostFaceNet H5 model (default: {DEFAULT_H5})')
    parser.add_argument('--output', type=str, default='ghostfacenet_nooverlap_int8.tflite',
                        help='Output TFLite model')
    parser.add_argument('--calib-dir', type=str, default='calibration_data/emb_112',
                        help='Calibration images directory')
    parser.add_argument('--skip-vela', action='store_true',
                        help='Skip Vela compilation')
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

    # Step 4: Add dummy lifetime extension node
    model_with_dummy = add_dummy_lifetime_node(model)

    # Step 5: Convert to TFLite INT8
    if not convert_to_tflite_int8(model_with_dummy, args.output, calib_images):
        print("\nFailed to convert to TFLite!")
        return 1

    # Step 6: Validate
    model_info = validate_tflite_model(args.output)

    # Step 7: Compile with Vela
    if not args.skip_vela:
        vela_output = compile_with_vela(args.output)
        if vela_output:
            validate_tflite_model(vela_output)

    print("\n" + "=" * 60)
    print("Fix completed!")
    print("=" * 60)
    print(f"\nOutput: {args.output}")
    print(f"Size: {model_info['file_size'] / 1024:.1f} KB")

    if model_info.get('num_outputs', 0) >= 2:
        print("\n✓ Model has dummy lifetime extension output")
        print("  This should prevent input/output memory overlap in Vela")

    return 0


if __name__ == "__main__":
    sys.exit(main())
