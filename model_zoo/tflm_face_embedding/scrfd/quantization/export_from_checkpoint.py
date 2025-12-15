#!/usr/bin/env python3
"""Export model from checkpoint without sigmoid/clamp."""

import torch
import numpy as np
import os
import sys
import argparse

# Import from main script
from qat_scrfd_enhanced import (
    SCRFD, load_pretrained_weights, fuse_model,
    export_to_onnx, convert_to_tflite, sample_ms1m_images
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="scrfd_qat_v4_best_checkpoint.pth")
    parser.add_argument('--output', default="scrfd_qat_v4_export")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")

    # Create model (without sigmoid/clamp in forward)
    model = SCRFD()

    # Load original weights first
    load_pretrained_weights(model, "scrfd_500m_kps.pth")

    # Fuse layers
    model = fuse_model(model)

    # Copy weights from checkpoint
    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    matched = 0
    for name, param in state_dict.items():
        clean_name = name.replace('module.', '')
        if clean_name in model_dict and model_dict[clean_name].shape == param.shape:
            model_dict[clean_name] = param
            matched += 1
    model.load_state_dict(model_dict)
    print(f"Loaded {matched} tensors from checkpoint")

    # Export ONNX
    model.eval()
    output_onnx = f"{args.output}.onnx"
    export_to_onnx(model, output_onnx, 160)

    # Load calibration data
    calib_data, _ = sample_ms1m_images("/Users/harvest/project/datasets/ms1m-arcface", 160, 1000)
    print(f"Calibration data: {calib_data.shape}, range [{calib_data.min():.2f}, {calib_data.max():.2f}]")

    # Convert to TFLite
    output_tflite = f"{args.output}.tflite"
    convert_to_tflite(output_onnx, output_tflite, calib_data)

    # Vela compile
    print("\nCompiling with Vela...")
    import subprocess
    result = subprocess.run(
        ["vela", "--accelerator-config", "ethos-u55-64", "--optimise", "Performance",
         output_tflite, "--output-dir", "."],
        capture_output=True, text=True
    )

    vela_output = f"{args.output}_vela.tflite"
    if result.returncode == 0:
        print(f"  Vela output: {vela_output}")
    else:
        print(f"  Vela failed: {result.stderr}")

    print(f"\nDone! Output: {vela_output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
