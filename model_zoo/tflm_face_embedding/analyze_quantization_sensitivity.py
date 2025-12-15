#!/usr/bin/env python3
"""
Quantization Sensitivity Analysis for GhostFaceNet

This tool analyzes which layers are most sensitive to INT8 quantization,
helping identify where precision loss occurs.

Analysis Methods:
1. Layer activation statistics (mean, std, min, max)
2. Per-layer output comparison between Float32 and QAT models
3. TFLite tensor analysis (scales, zero points, ranges)
4. Cumulative error propagation analysis

Usage:
    python analyze_quantization_sensitivity.py \
        --h5 GN_W0.5_S2_ArcFace_epoch16.h5 \
        --calib-dir calibration_data/qat_112 \
        --num-samples 100
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


def load_calibration_images(calib_dir: str, num_samples: int = 100, target_size: int = 112):
    """Load calibration images for analysis."""
    import cv2

    image_files = sorted(Path(calib_dir).glob("*.jpg"))[:num_samples]
    if not image_files:
        raise ValueError(f"No images found in {calib_dir}")

    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size))
        # Normalize to [-1, 1] (GhostFaceNet input range)
        img = (img.astype(np.float32) - 127.5) / 127.5
        images.append(img)

    return np.array(images)


def create_intermediate_models(model):
    """
    Create models that output intermediate layer activations.
    Returns dict of {layer_name: intermediate_model}
    """
    intermediate_models = {}

    for layer in model.layers:
        # Skip input layers and layers without meaningful output
        if 'input' in layer.name.lower():
            continue

        try:
            intermediate_model = tf.keras.Model(
                inputs=model.input,
                outputs=layer.output,
                name=f"intermediate_{layer.name}"
            )
            intermediate_models[layer.name] = intermediate_model
        except Exception as e:
            # Some layers may not be connectable
            pass

    return intermediate_models


def analyze_layer_activations(model, images, layer_names=None):
    """
    Analyze activation statistics for each layer.

    Returns dict of {layer_name: {mean, std, min, max, range, sparsity}}
    """
    print("\n" + "="*60)
    print("Layer Activation Statistics Analysis")
    print("="*60)

    intermediate_models = create_intermediate_models(model)

    if layer_names:
        intermediate_models = {k: v for k, v in intermediate_models.items() if k in layer_names}

    stats = {}

    for layer_name, inter_model in intermediate_models.items():
        try:
            activations = inter_model.predict(images, verbose=0)

            # Compute statistics
            act_mean = np.mean(activations)
            act_std = np.std(activations)
            act_min = np.min(activations)
            act_max = np.max(activations)
            act_range = act_max - act_min

            # Sparsity (percentage of zeros or near-zeros)
            sparsity = np.mean(np.abs(activations) < 1e-6) * 100

            stats[layer_name] = {
                'mean': act_mean,
                'std': act_std,
                'min': act_min,
                'max': act_max,
                'range': act_range,
                'sparsity': sparsity,
                'shape': activations.shape
            }

        except Exception as e:
            pass

    return stats


def compare_float_vs_qat(float_model, qat_model, images):
    """
    Compare outputs between Float32 and QAT models layer by layer.

    Returns dict of {layer_name: {mse, cosine_sim, relative_error}}
    """
    print("\n" + "="*60)
    print("Float32 vs QAT Layer-by-Layer Comparison")
    print("="*60)

    float_intermediates = create_intermediate_models(float_model)
    qat_intermediates = create_intermediate_models(qat_model)

    # Find common layers
    common_layers = set(float_intermediates.keys()) & set(qat_intermediates.keys())

    comparison = {}

    for layer_name in sorted(common_layers):
        try:
            float_out = float_intermediates[layer_name].predict(images, verbose=0)
            qat_out = qat_intermediates[layer_name].predict(images, verbose=0)

            # Flatten for comparison
            float_flat = float_out.reshape(float_out.shape[0], -1)
            qat_flat = qat_out.reshape(qat_out.shape[0], -1)

            # MSE
            mse = np.mean((float_flat - qat_flat) ** 2)

            # Cosine similarity (average across samples)
            cosine_sims = []
            for i in range(len(float_flat)):
                norm_f = np.linalg.norm(float_flat[i])
                norm_q = np.linalg.norm(qat_flat[i])
                if norm_f > 1e-8 and norm_q > 1e-8:
                    cos_sim = np.dot(float_flat[i], qat_flat[i]) / (norm_f * norm_q)
                    cosine_sims.append(cos_sim)
            avg_cosine = np.mean(cosine_sims) if cosine_sims else 0

            # Relative error
            float_norm = np.mean(np.abs(float_flat))
            relative_error = np.sqrt(mse) / (float_norm + 1e-8)

            comparison[layer_name] = {
                'mse': mse,
                'cosine_sim': avg_cosine,
                'relative_error': relative_error,
                'float_mean': np.mean(float_flat),
                'qat_mean': np.mean(qat_flat)
            }

        except Exception as e:
            pass

    return comparison


def analyze_tflite_tensors(tflite_path: str):
    """
    Analyze TFLite model tensor quantization parameters.

    Returns dict of {tensor_name: {scale, zero_point, dtype, shape}}
    """
    print("\n" + "="*60)
    print(f"TFLite Tensor Analysis: {os.path.basename(tflite_path)}")
    print("="*60)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()

    analysis = {}

    for tensor in tensor_details:
        name = tensor['name']
        quant_params = tensor.get('quantization_parameters', {})

        scales = quant_params.get('scales', np.array([]))
        zero_points = quant_params.get('zero_points', np.array([]))

        analysis[name] = {
            'shape': tensor['shape'].tolist(),
            'dtype': str(tensor['dtype']),
            'scales': scales.tolist() if len(scales) > 0 else [],
            'zero_points': zero_points.tolist() if len(zero_points) > 0 else [],
            'num_scales': len(scales),
        }

        # Compute quantization range if applicable
        if len(scales) > 0 and tensor['dtype'] == np.int8:
            min_scale = np.min(scales)
            max_scale = np.max(scales)
            analysis[name]['scale_range'] = (min_scale, max_scale)
            # Effective range: scale * (min_int8, max_int8) = scale * (-128, 127)
            analysis[name]['effective_range'] = (-128 * max_scale, 127 * max_scale)

    return analysis


def compare_tflite_outputs(float_tflite: str, int8_tflite: str, images):
    """
    Compare Float32 TFLite vs INT8 TFLite outputs.
    """
    print("\n" + "="*60)
    print("Float32 TFLite vs INT8 TFLite Comparison")
    print("="*60)

    # Load interpreters
    float_interp = tf.lite.Interpreter(model_path=float_tflite)
    float_interp.allocate_tensors()

    int8_interp = tf.lite.Interpreter(model_path=int8_tflite)
    int8_interp.allocate_tensors()

    float_input = float_interp.get_input_details()[0]
    float_output = float_interp.get_output_details()[0]

    int8_input = int8_interp.get_input_details()[0]
    int8_output = int8_interp.get_output_details()[0]

    float_embeddings = []
    int8_embeddings = []

    for img in images:
        # Float32 inference
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        float_interp.set_tensor(float_input['index'], input_data)
        float_interp.invoke()
        float_emb = float_interp.get_tensor(float_output['index'])[0]
        float_embeddings.append(float_emb)

        # INT8 inference (may need quantization)
        if int8_input['dtype'] == np.int8:
            scale = int8_input['quantization_parameters']['scales'][0]
            zp = int8_input['quantization_parameters']['zero_points'][0]
            int8_data = np.round(input_data / scale + zp).astype(np.int8)
        else:
            int8_data = input_data

        int8_interp.set_tensor(int8_input['index'], int8_data)
        int8_interp.invoke()
        int8_emb = int8_interp.get_tensor(int8_output['index'])[0]

        # Dequantize if needed
        if int8_output['dtype'] == np.int8:
            scale = int8_output['quantization_parameters']['scales'][0]
            zp = int8_output['quantization_parameters']['zero_points'][0]
            int8_emb = (int8_emb.astype(np.float32) - zp) * scale

        int8_embeddings.append(int8_emb)

    float_embeddings = np.array(float_embeddings)
    int8_embeddings = np.array(int8_embeddings)

    # Normalize and compute similarity
    float_norm = float_embeddings / (np.linalg.norm(float_embeddings, axis=1, keepdims=True) + 1e-8)
    int8_norm = int8_embeddings / (np.linalg.norm(int8_embeddings, axis=1, keepdims=True) + 1e-8)

    cosine_sims = np.sum(float_norm * int8_norm, axis=1)

    return {
        'mean_cosine_sim': np.mean(cosine_sims),
        'std_cosine_sim': np.std(cosine_sims),
        'min_cosine_sim': np.min(cosine_sims),
        'max_cosine_sim': np.max(cosine_sims),
        'mse': np.mean((float_embeddings - int8_embeddings) ** 2),
    }


def identify_sensitive_layers(comparison_results, threshold=0.95):
    """
    Identify layers that are most sensitive to quantization.

    Returns list of (layer_name, sensitivity_score, reason)
    """
    sensitive_layers = []

    for layer_name, metrics in comparison_results.items():
        reasons = []
        sensitivity_score = 0

        # Check cosine similarity drop
        if metrics['cosine_sim'] < threshold:
            sensitivity_score += (threshold - metrics['cosine_sim']) * 100
            reasons.append(f"cosine_sim={metrics['cosine_sim']:.4f}")

        # Check relative error
        if metrics['relative_error'] > 0.1:
            sensitivity_score += metrics['relative_error'] * 10
            reasons.append(f"rel_error={metrics['relative_error']:.4f}")

        # Check MSE
        if metrics['mse'] > 0.01:
            sensitivity_score += metrics['mse'] * 10
            reasons.append(f"mse={metrics['mse']:.6f}")

        if sensitivity_score > 0:
            sensitive_layers.append((layer_name, sensitivity_score, ", ".join(reasons)))

    # Sort by sensitivity score (most sensitive first)
    sensitive_layers.sort(key=lambda x: x[1], reverse=True)

    return sensitive_layers


def print_activation_stats(stats, top_n=20):
    """Print activation statistics in a formatted table."""
    print("\n" + "-"*100)
    print(f"{'Layer Name':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10} {'Sparse%':>8}")
    print("-"*100)

    # Sort by range (largest first - potentially problematic)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['range'], reverse=True)

    for layer_name, s in sorted_stats[:top_n]:
        print(f"{layer_name:<40} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['range']:>10.4f} {s['sparsity']:>7.2f}%")

    if len(sorted_stats) > top_n:
        print(f"... and {len(sorted_stats) - top_n} more layers")


def print_comparison_results(comparison, top_n=20):
    """Print layer comparison results."""
    print("\n" + "-"*100)
    print(f"{'Layer Name':<40} {'MSE':>12} {'Cosine Sim':>12} {'Rel Error':>12} {'Float Mean':>12} {'QAT Mean':>12}")
    print("-"*100)

    # Sort by cosine similarity (lowest first - most problematic)
    sorted_comp = sorted(comparison.items(), key=lambda x: x[1]['cosine_sim'])

    for layer_name, c in sorted_comp[:top_n]:
        print(f"{layer_name:<40} {c['mse']:>12.6f} {c['cosine_sim']:>12.4f} {c['relative_error']:>12.4f} {c['float_mean']:>12.4f} {c['qat_mean']:>12.4f}")

    if len(sorted_comp) > top_n:
        print(f"... and {len(sorted_comp) - top_n} more layers")


def print_tflite_analysis(analysis, show_all=False):
    """Print TFLite tensor analysis."""
    print("\n" + "-"*100)
    print(f"{'Tensor Name':<50} {'Shape':<20} {'Dtype':<10} {'Scales':>10} {'Range'}")
    print("-"*100)

    # Filter to show only quantized tensors
    quantized = {k: v for k, v in analysis.items() if v['num_scales'] > 0}

    # Sort by scale magnitude (smallest first - highest precision)
    sorted_tensors = sorted(quantized.items(),
                           key=lambda x: np.min(x[1]['scales']) if x[1]['scales'] else float('inf'))

    for name, info in sorted_tensors[:30] if not show_all else sorted_tensors:
        shape_str = str(info['shape'])[:18]
        scales = info['scales']
        if scales:
            if len(scales) == 1:
                scale_str = f"{scales[0]:.6f}"
            else:
                scale_str = f"{np.min(scales):.4f}-{np.max(scales):.4f}"

            if 'effective_range' in info:
                range_str = f"[{info['effective_range'][0]:.2f}, {info['effective_range'][1]:.2f}]"
            else:
                range_str = ""
        else:
            scale_str = "N/A"
            range_str = ""

        print(f"{name:<50} {shape_str:<20} {info['dtype']:<10} {scale_str:>10} {range_str}")


def print_sensitive_layers(sensitive_layers):
    """Print sensitive layer analysis."""
    print("\n" + "="*60)
    print("QUANTIZATION-SENSITIVE LAYERS (Sorted by Sensitivity)")
    print("="*60)

    if not sensitive_layers:
        print("No significantly sensitive layers detected!")
        return

    print(f"\n{'Rank':<6} {'Layer Name':<40} {'Score':>8} {'Reasons'}")
    print("-"*100)

    for i, (layer_name, score, reasons) in enumerate(sensitive_layers[:20], 1):
        print(f"{i:<6} {layer_name:<40} {score:>8.2f} {reasons}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze quantization sensitivity of GhostFaceNet layers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--h5', type=str, default='GN_W0.5_S2_ArcFace_epoch16.h5',
                        help='Path to Float32 Keras model (.h5)')
    parser.add_argument('--qat-h5', type=str, default=None,
                        help='Path to QAT Keras model (.h5) for comparison')
    parser.add_argument('--tflite-float', type=str, default=None,
                        help='Path to Float32 TFLite model')
    parser.add_argument('--tflite-int8', type=str, default='ghostfacenet_qat_int8.tflite',
                        help='Path to INT8 TFLite model')
    parser.add_argument('--calib-dir', type=str, default='calibration_data/qat_112',
                        help='Directory with calibration images')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples for analysis')
    parser.add_argument('--output', type=str, default='sensitivity_report.txt',
                        help='Output report file')

    args = parser.parse_args()

    print("="*60)
    print("Quantization Sensitivity Analysis")
    print("="*60)
    print(f"Float32 model: {args.h5}")
    print(f"INT8 TFLite: {args.tflite_int8}")
    print(f"Calibration dir: {args.calib_dir}")
    print(f"Samples: {args.num_samples}")

    # Load calibration images
    print("\nLoading calibration images...")
    images = load_calibration_images(args.calib_dir, args.num_samples)
    print(f"Loaded {len(images)} images")

    # Load Float32 model
    print("\nLoading Float32 model...")
    float_model = tf.keras.models.load_model(args.h5, compile=False)
    print(f"  Layers: {len(float_model.layers)}")

    # 1. Analyze layer activations
    print("\n[1/4] Analyzing layer activation statistics...")
    activation_stats = analyze_layer_activations(float_model, images)
    print_activation_stats(activation_stats)

    # 2. Compare Float32 vs QAT (if QAT model provided)
    comparison_results = None
    if args.qat_h5 and os.path.exists(args.qat_h5):
        print("\n[2/4] Comparing Float32 vs QAT model...")
        qat_model = tf.keras.models.load_model(args.qat_h5, compile=False)
        comparison_results = compare_float_vs_qat(float_model, qat_model, images)
        print_comparison_results(comparison_results)
    else:
        print("\n[2/4] Skipping Float32 vs QAT comparison (no QAT model provided)")

    # 3. Analyze TFLite tensors
    if os.path.exists(args.tflite_int8):
        print("\n[3/4] Analyzing INT8 TFLite tensor quantization...")
        tflite_analysis = analyze_tflite_tensors(args.tflite_int8)
        print_tflite_analysis(tflite_analysis)
    else:
        print(f"\n[3/4] Skipping TFLite analysis ({args.tflite_int8} not found)")
        tflite_analysis = None

    # 4. Identify sensitive layers
    print("\n[4/4] Identifying quantization-sensitive layers...")
    if comparison_results:
        sensitive_layers = identify_sensitive_layers(comparison_results)
        print_sensitive_layers(sensitive_layers)
    else:
        print("  Skipping (requires QAT model for comparison)")
        sensitive_layers = []

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Find layers with extreme activation ranges
    extreme_range_layers = [(name, stats['range']) for name, stats in activation_stats.items()
                            if stats['range'] > 10]
    if extreme_range_layers:
        print("\nLayers with large activation range (>10):")
        for name, range_val in sorted(extreme_range_layers, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {name}: range={range_val:.2f}")

    # PReLU layer analysis
    prelu_layers = [name for name in activation_stats.keys() if 'prelu' in name.lower()]
    if prelu_layers:
        print(f"\nPReLU layers found: {len(prelu_layers)}")
        prelu_stats = {k: v for k, v in activation_stats.items() if k in prelu_layers}
        avg_range = np.mean([s['range'] for s in prelu_stats.values()])
        print(f"  Average activation range: {avg_range:.2f}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    recommendations = []

    # Check for PReLU issues
    if prelu_layers and avg_range > 5:
        recommendations.append(
            "1. PReLU layers have large activation ranges. Consider:\n"
            "   - Using LeakyReLU with fixed negative slope instead\n"
            "   - Applying activation clipping before PReLU\n"
            "   - Using per-channel quantization for PReLU weights"
        )

    # Check for extreme ranges
    if extreme_range_layers:
        recommendations.append(
            "2. Some layers have extreme activation ranges (>10). Consider:\n"
            "   - Adding batch normalization after these layers\n"
            "   - Using activation clipping (tf.clip_by_value)\n"
            "   - Adjusting learning rate during QAT training"
        )

    # Check sensitive layers
    if sensitive_layers and sensitive_layers[0][1] > 5:
        top_sensitive = [l[0] for l in sensitive_layers[:5]]
        recommendations.append(
            f"3. Most sensitive layers: {', '.join(top_sensitive)}\n"
            "   - Try mixed-precision quantization (keep these in float32)\n"
            "   - Use higher-precision calibration for these layers\n"
            "   - Consider architecture modifications"
        )

    if recommendations:
        for rec in recommendations:
            print(f"\n{rec}")
    else:
        print("\nNo critical issues detected. Model appears well-suited for INT8 quantization.")

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
