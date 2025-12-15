#!/usr/bin/env python3
"""
SCRFD 量化模型验证脚本

比较 INT8 TFLite 模型与原始 Float32 ONNX 模型的输出精度。

Usage:
    python validate_quantization.py --onnx scrfd.onnx --tflite scrfd_int8.tflite
"""

import argparse
import numpy as np
import cv2
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("请安装 onnxruntime: pip install onnxruntime")
    exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("请安装 tensorflow: pip install tensorflow")
    exit(1)


def load_test_images(image_dir: str, input_size: int, max_images: int = 50):
    """加载测试图片"""
    images = []
    image_path = Path(image_dir)

    if not image_path.exists():
        print(f"图片目录不存在: {image_dir}")
        return None

    all_images = sorted(image_path.rglob("*.jpg"))[:max_images]

    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    return np.array(images) if images else None


def run_onnx_inference(session, images, input_name):
    """运行 ONNX 推理"""
    outputs_all = []

    for img in images:
        # ONNX 输入: NCHW, 归一化 [-1, 1]
        img_normalized = (img.astype(np.float32) - 127.5) / 128.0
        img_nchw = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
        outputs = session.run(None, {input_name: img_nchw})
        outputs_all.append(outputs)

    return outputs_all


def run_tflite_inference(interpreter, images, input_details, output_details):
    """运行 TFLite 推理"""
    outputs_all = []

    for img in images:
        # TFLite 输入: NHWC, 归一化 [0, 1]
        img_01 = img.astype(np.float32) / 255.0
        img_nhwc = img_01[np.newaxis, ...]

        # 量化输入
        scale, zp = input_details[0]['quantization']
        if scale != 0:
            int8_input = np.clip(np.round(img_nhwc / scale + zp), -128, 127).astype(np.int8)
        else:
            int8_input = img_nhwc.astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], int8_input)
        interpreter.invoke()

        # 获取并反量化输出
        outputs = []
        for out_detail in output_details:
            raw = interpreter.get_tensor(out_detail['index'])
            s, z = out_detail['quantization']
            if s != 0:
                dequant = (raw.astype(np.float32) - z) * s
            else:
                dequant = raw.astype(np.float32)
            outputs.append(dequant)

        outputs_all.append(outputs)

    return outputs_all


def calculate_metrics(onnx_outputs, tflite_outputs):
    """计算精度指标"""
    metrics = {
        'scores': {'corr': [], 'mae': [], 'mse': []},
        'boxes': {'corr': [], 'mae': [], 'mse': []},
        'keypoints': {'corr': [], 'mae': [], 'mse': []}
    }

    for onnx_out_list, tflite_out_list in zip(onnx_outputs, tflite_outputs):
        for onnx_out in onnx_out_list:
            if len(onnx_out.shape) != 3:
                continue

            n_channels = onnx_out.shape[-1]

            # 确定输出类型
            if n_channels == 1:
                key = 'scores'
            elif n_channels == 4:
                key = 'boxes'
            elif n_channels == 10:
                key = 'keypoints'
            else:
                continue

            # 找到匹配的 TFLite 输出
            for tflite_out in tflite_out_list:
                if tflite_out.shape == onnx_out.shape:
                    onnx_flat = onnx_out.flatten()
                    tflite_flat = tflite_out.flatten()

                    if np.std(onnx_flat) > 1e-6:
                        corr = np.corrcoef(onnx_flat, tflite_flat)[0, 1]
                        mae = np.mean(np.abs(onnx_flat - tflite_flat))
                        mse = np.mean((onnx_flat - tflite_flat) ** 2)

                        if not np.isnan(corr):
                            metrics[key]['corr'].append(corr)
                            metrics[key]['mae'].append(mae)
                            metrics[key]['mse'].append(mse)
                    break

    return metrics


def print_report(metrics):
    """打印验证报告"""
    print("\n" + "=" * 60)
    print("量化精度验证报告")
    print("=" * 60)

    for key in ['scores', 'boxes', 'keypoints']:
        if metrics[key]['corr']:
            avg_corr = np.mean(metrics[key]['corr'])
            avg_mae = np.mean(metrics[key]['mae'])
            avg_mse = np.mean(metrics[key]['mse'])

            print(f"\n{key.upper()}:")
            print(f"  相关系数 (Correlation): {avg_corr:.6f}")
            print(f"  平均绝对误差 (MAE): {avg_mae:.6f}")
            print(f"  均方误差 (MSE): {avg_mse:.6f}")

    print("\n" + "=" * 60)
    print("精度评估标准:")
    print("  - 相关系数 > 0.99: 优秀")
    print("  - 相关系数 > 0.95: 良好")
    print("  - 相关系数 > 0.90: 可接受")
    print("  - 相关系数 < 0.90: 需要优化")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='SCRFD 量化模型验证')
    parser.add_argument('--onnx', type=str, required=True, help='ONNX 模型路径')
    parser.add_argument('--tflite', type=str, required=True, help='TFLite 模型路径')
    parser.add_argument('--image-dir', type=str,
                        default='calibration_data/lfw/lfw-deepfunneled',
                        help='测试图片目录')
    parser.add_argument('--input-size', type=int, default=160, help='输入尺寸')
    parser.add_argument('--num-samples', type=int, default=50, help='测试样本数')
    args = parser.parse_args()

    print("=" * 60)
    print("SCRFD 量化模型验证")
    print("=" * 60)
    print(f"\nONNX 模型: {args.onnx}")
    print(f"TFLite 模型: {args.tflite}")
    print(f"输入尺寸: {args.input_size}x{args.input_size}")
    print(f"测试样本数: {args.num_samples}")

    # 加载测试图片
    print("\n加载测试图片...")
    test_images = load_test_images(args.image_dir, args.input_size, args.num_samples)
    if test_images is None or len(test_images) < 10:
        print("测试图片不足!")
        return 1

    print(f"  已加载 {len(test_images)} 张图片")

    # 加载 ONNX 模型
    print("\n加载 ONNX 模型...")
    onnx_session = ort.InferenceSession(args.onnx)
    onnx_input_name = onnx_session.get_inputs()[0].name
    print(f"  输入: {onnx_input_name}")

    # 加载 TFLite 模型
    print("加载 TFLite 模型...")
    tflite_interpreter = tf.lite.Interpreter(model_path=args.tflite)
    tflite_interpreter.allocate_tensors()
    tflite_input_details = tflite_interpreter.get_input_details()
    tflite_output_details = tflite_interpreter.get_output_details()
    print(f"  输入形状: {tflite_input_details[0]['shape']}")
    print(f"  输出数量: {len(tflite_output_details)}")

    # 运行推理
    print("\n运行 ONNX 推理...")
    onnx_outputs = run_onnx_inference(
        onnx_session, test_images, onnx_input_name
    )

    print("运行 TFLite 推理...")
    tflite_outputs = run_tflite_inference(
        tflite_interpreter, test_images,
        tflite_input_details, tflite_output_details
    )

    # 计算指标
    print("\n计算精度指标...")
    metrics = calculate_metrics(onnx_outputs, tflite_outputs)

    # 打印报告
    print_report(metrics)

    return 0


if __name__ == "__main__":
    exit(main())
