#!/usr/bin/env python3
"""
Deep TFLite Layer Analysis - Compare intermediate tensor outputs between
Float32 Keras model and INT8 TFLite model to identify quantization error sources.
"""

import os
import sys
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


def load_test_images(calib_dir: str, num_samples: int = 20, target_size: int = 112):
    """Load test images."""
    import cv2
    image_files = sorted(Path(calib_dir).glob("*.jpg"))[:num_samples]
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size))
        img = (img.astype(np.float32) - 127.5) / 127.5
        images.append(img)
    return np.array(images)


def analyze_activation_distribution(model, images):
    """Analyze activation distribution for each layer."""
    print("\n" + "="*80)
    print("ACTIVATION DISTRIBUTION ANALYSIS")
    print("="*80)

    # Find layers to analyze
    layers_to_check = []
    for layer in model.layers:
        if any(x in layer.name.lower() for x in ['conv', 'activation', 'prelu', 'add', 'dense', 'depthwise']):
            try:
                # Test if we can get output
                inter_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
                layers_to_check.append((layer.name, inter_model))
            except:
                pass

    print(f"\nAnalyzing {len(layers_to_check)} layers...")
    print("\n" + "-"*100)
    print(f"{'Layer Name':<45} {'Min':>10} {'Max':>10} {'Range':>10} {'Mean':>10} {'|Max|':>10} {'Issue'}")
    print("-"*100)

    problematic_layers = []

    for layer_name, inter_model in layers_to_check:
        try:
            activations = inter_model.predict(images, verbose=0)

            act_min = float(np.min(activations))
            act_max = float(np.max(activations))
            act_range = act_max - act_min
            act_mean = float(np.mean(activations))
            act_abs_max = max(abs(act_min), abs(act_max))

            # Determine if problematic
            issue = ""
            if act_range > 50:
                issue = "LARGE RANGE"
                problematic_layers.append((layer_name, act_range, act_min, act_max))
            elif act_abs_max > 30:
                issue = "LARGE ABS"
                problematic_layers.append((layer_name, act_range, act_min, act_max))

            # Only print layers with issues or every 10th layer
            if issue or len(layers_to_check) < 50:
                print(f"{layer_name:<45} {act_min:>10.2f} {act_max:>10.2f} {act_range:>10.2f} {act_mean:>10.4f} {act_abs_max:>10.2f} {issue}")

        except Exception as e:
            pass

    # Summary of problematic layers
    print("\n" + "="*80)
    print("PROBLEMATIC LAYERS SUMMARY")
    print("="*80)

    if problematic_layers:
        print(f"\nFound {len(problematic_layers)} layers with large activation ranges:")
        print("\n" + "-"*80)
        for name, range_val, min_val, max_val in sorted(problematic_layers, key=lambda x: x[1], reverse=True)[:20]:
            # Calculate quantization error
            # INT8 range: -128 to 127 (256 levels)
            quant_step = range_val / 256
            print(f"  {name:<45} range={range_val:>8.2f} [{min_val:.1f}, {max_val:.1f}] quant_step={quant_step:.4f}")
    else:
        print("No problematic layers found!")

    return problematic_layers


def compare_float_vs_int8_embeddings(h5_path, tflite_path, images):
    """Compare final embeddings between Float32 Keras and INT8 TFLite."""
    print("\n" + "="*80)
    print("FLOAT32 vs INT8 EMBEDDING COMPARISON")
    print("="*80)

    # Float32 Keras model
    float_model = tf.keras.models.load_model(h5_path, compile=False)

    # INT8 TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"\nINT8 TFLite input: dtype={input_details['dtype']}, shape={input_details['shape']}")
    print(f"INT8 TFLite output: dtype={output_details['dtype']}, shape={output_details['shape']}")

    if 'quantization_parameters' in input_details:
        qp = input_details['quantization_parameters']
        if 'scales' in qp and len(qp['scales']) > 0:
            print(f"  Input scale={qp['scales'][0]:.6f}, zero_point={qp['zero_points'][0]}")

    if 'quantization_parameters' in output_details:
        qp = output_details['quantization_parameters']
        if 'scales' in qp and len(qp['scales']) > 0:
            print(f"  Output scale={qp['scales'][0]:.6f}, zero_point={qp['zero_points'][0]}")

    float_embeddings = []
    int8_embeddings = []

    for i, img in enumerate(images):
        # Float32 inference
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        float_emb = float_model.predict(input_data, verbose=0)[0]
        float_embeddings.append(float_emb)

        # INT8 inference
        if input_details['dtype'] == np.int8:
            scale = input_details['quantization_parameters']['scales'][0]
            zp = input_details['quantization_parameters']['zero_points'][0]
            int8_input = np.round(input_data / scale + zp).astype(np.int8)
        else:
            int8_input = input_data

        interpreter.set_tensor(input_details['index'], int8_input)
        interpreter.invoke()
        int8_emb = interpreter.get_tensor(output_details['index'])[0]

        # Dequantize output if needed
        if output_details['dtype'] == np.int8:
            scale = output_details['quantization_parameters']['scales'][0]
            zp = output_details['quantization_parameters']['zero_points'][0]
            int8_emb = (int8_emb.astype(np.float32) - zp) * scale

        int8_embeddings.append(int8_emb)

    float_embeddings = np.array(float_embeddings)
    int8_embeddings = np.array(int8_embeddings)

    # Normalize for cosine similarity
    float_norm = float_embeddings / (np.linalg.norm(float_embeddings, axis=1, keepdims=True) + 1e-8)
    int8_norm = int8_embeddings / (np.linalg.norm(int8_embeddings, axis=1, keepdims=True) + 1e-8)

    # Per-sample cosine similarity
    cosine_sims = np.sum(float_norm * int8_norm, axis=1)

    print(f"\n{'Sample':<10} {'Cosine Sim':>12} {'Float Norm':>12} {'INT8 Norm':>12} {'MSE':>12}")
    print("-"*60)
    for i in range(min(10, len(images))):
        float_n = np.linalg.norm(float_embeddings[i])
        int8_n = np.linalg.norm(int8_embeddings[i])
        mse = np.mean((float_embeddings[i] - int8_embeddings[i])**2)
        print(f"{i:<10} {cosine_sims[i]:>12.4f} {float_n:>12.4f} {int8_n:>12.4f} {mse:>12.6f}")

    print("-"*60)
    print(f"{'Average':<10} {np.mean(cosine_sims):>12.4f}")
    print(f"{'Std':<10} {np.std(cosine_sims):>12.4f}")
    print(f"{'Min':<10} {np.min(cosine_sims):>12.4f}")
    print(f"{'Max':<10} {np.max(cosine_sims):>12.4f}")

    # Analyze embedding dimension-wise errors
    print("\n" + "="*80)
    print("EMBEDDING DIMENSION ANALYSIS")
    print("="*80)

    dim_errors = np.mean((float_embeddings - int8_embeddings)**2, axis=0)
    worst_dims = np.argsort(dim_errors)[-20:][::-1]

    print(f"\nTop 20 dimensions with largest errors:")
    print("-"*40)
    for dim in worst_dims:
        print(f"  Dim {dim:3d}: MSE={dim_errors[dim]:.6f}, Float mean={np.mean(float_embeddings[:, dim]):.4f}, INT8 mean={np.mean(int8_embeddings[:, dim]):.4f}")

    return {
        'mean_cosine_sim': np.mean(cosine_sims),
        'std_cosine_sim': np.std(cosine_sims),
        'mse': np.mean((float_embeddings - int8_embeddings)**2),
    }


def analyze_tflite_all_tensors(tflite_path, images):
    """Analyze all intermediate tensors in TFLite model."""
    print("\n" + "="*80)
    print("TFLITE INTERMEDIATE TENSOR ANALYSIS")
    print("="*80)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    all_tensors = interpreter.get_tensor_details()

    # Run inference to populate tensors
    img = images[0:1]
    if input_details['dtype'] == np.int8:
        scale = input_details['quantization_parameters']['scales'][0]
        zp = input_details['quantization_parameters']['zero_points'][0]
        input_data = np.round(img / scale + zp).astype(np.int8)
    else:
        input_data = img.astype(np.float32)

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # Analyze each tensor
    print(f"\nAnalyzing {len(all_tensors)} tensors...")
    print("\n" + "-"*120)
    print(f"{'Tensor Name':<60} {'Shape':<20} {'Dtype':<12} {'Scale':>10} {'ZP':>6} {'Range'}")
    print("-"*120)

    tensor_stats = []
    for tensor_info in all_tensors:
        name = tensor_info['name']
        shape = tensor_info['shape']
        dtype = tensor_info['dtype']

        try:
            tensor_data = interpreter.get_tensor(tensor_info['index'])

            qp = tensor_info.get('quantization_parameters', {})
            scales = qp.get('scales', np.array([]))
            zps = qp.get('zero_points', np.array([]))

            if len(scales) > 0:
                scale = scales[0] if len(scales) == 1 else np.mean(scales)
                zp = zps[0] if len(zps) == 1 else int(np.mean(zps))

                # Dequantize
                if dtype == np.int8:
                    dequant = (tensor_data.astype(np.float32) - zp) * scale
                    t_min, t_max = np.min(dequant), np.max(dequant)
                    t_range = t_max - t_min
                else:
                    t_min, t_max = np.min(tensor_data), np.max(tensor_data)
                    t_range = t_max - t_min

                shape_str = str(list(shape))[:18]
                scale_str = f"{scale:.6f}"
                range_str = f"[{t_min:.2f}, {t_max:.2f}] r={t_range:.2f}"

                # Flag if range is very small (potential precision issue)
                flag = ""
                if dtype == np.int8:
                    # For INT8, check if the effective range uses few quantization levels
                    int8_min, int8_max = np.min(tensor_data), np.max(tensor_data)
                    int8_range = int8_max - int8_min
                    if int8_range < 50:
                        flag = f" [LOW RANGE: {int8_range}]"

                # Only show conv/activation tensors
                if any(x in name.lower() for x in ['conv', 'activation', 'prelu', 'add', 'dense', 'depthwise', 'output']):
                    print(f"{name[:58]:<60} {shape_str:<20} {str(dtype):<12} {scale_str:>10} {zp:>6} {range_str}{flag}")

                tensor_stats.append({
                    'name': name,
                    'scale': scale,
                    'zp': zp,
                    'range': t_range,
                    'dtype': dtype
                })

        except Exception as e:
            pass

    # Find tensors with concerning properties
    print("\n" + "="*80)
    print("TENSORS WITH POTENTIAL ISSUES")
    print("="*80)

    # Large scale (coarse quantization)
    large_scale = [t for t in tensor_stats if t['scale'] > 0.1 and t['dtype'] == np.int8]
    if large_scale:
        print(f"\nTensors with large scale (>0.1) - coarse quantization:")
        for t in sorted(large_scale, key=lambda x: x['scale'], reverse=True)[:10]:
            print(f"  {t['name'][:50]}: scale={t['scale']:.4f}")

    return tensor_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep TFLite layer analysis")
    parser.add_argument('--h5', default='GN_W0.5_S2_ArcFace_epoch16.h5')
    parser.add_argument('--tflite', default='ghostfacenet_qat_int8.tflite')
    parser.add_argument('--calib-dir', default='calibration_data/qat_112')
    parser.add_argument('--num-samples', type=int, default=20)
    args = parser.parse_args()

    print("="*80)
    print("DEEP QUANTIZATION ANALYSIS")
    print("="*80)

    # Load test images
    print("\nLoading test images...")
    images = load_test_images(args.calib_dir, args.num_samples)
    print(f"Loaded {len(images)} images")

    # 1. Analyze Float32 model activation distributions
    print("\n[1/3] Analyzing Float32 model activations...")
    float_model = tf.keras.models.load_model(args.h5, compile=False)
    problematic_layers = analyze_activation_distribution(float_model, images)

    # 2. Compare Float32 vs INT8 embeddings
    print("\n[2/3] Comparing Float32 vs INT8 embeddings...")
    comparison = compare_float_vs_int8_embeddings(args.h5, args.tflite, images)

    # 3. Analyze TFLite intermediate tensors
    print("\n[3/3] Analyzing TFLite intermediate tensors...")
    tensor_stats = analyze_tflite_all_tensors(args.tflite, images)

    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVING QUANTIZATION")
    print("="*80)

    if problematic_layers:
        print(f"""
1. ADDRESS LARGE ACTIVATION RANGES
   Found {len(problematic_layers)} layers with range > 50.

   Solutions:
   a) Add activation clipping in the model:
      x = tf.clip_by_value(x, -10, 10)  # Before quantization-sensitive layers

   b) Use BatchNormalization to normalize activations:
      x = layers.BatchNormalization()(x)

   c) Replace PReLU with LeakyReLU (fixed negative slope):
      # PReLU learns slope which can cause large ranges
      x = layers.LeakyReLU(alpha=0.1)(x)

2. MOST PROBLEMATIC LAYERS (by activation range):
""")
        for name, range_val, min_val, max_val in sorted(problematic_layers, key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {name}: range={range_val:.1f}")

    print(f"""
3. QUANTIZATION PRECISION
   Current Float32→INT8 similarity: {comparison['mean_cosine_sim']:.4f}

   To improve:
   a) Fine-tune QAT with more epochs and larger dataset
   b) Use mixed-precision: keep problematic layers in float16
   c) Try per-tensor vs per-channel quantization
   d) Increase calibration data diversity

4. ARCHITECTURE MODIFICATIONS
   Consider modifying the architecture to be more quantization-friendly:
   - Replace PReLU with ReLU6 (bounded activation)
   - Add skip connections to preserve information flow
   - Use depth-separable convolutions with BN after each
""")

    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)


if __name__ == "__main__":
    main()
