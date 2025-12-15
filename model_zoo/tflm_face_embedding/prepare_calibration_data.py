#!/usr/bin/env python3
"""
Prepare face datasets for quantization calibration and QAT training.

This script uses local LFW dataset (LFW_dataset.zip) and optionally downloads
additional datasets for QAT training (which requires more data).

Supported datasets:
- LFW (Labeled Faces in the Wild) - local zip file
- CelebA (optional download for QAT)

Usage:
    # Basic calibration data (500 images)
    python prepare_calibration_data.py

    # Extended data for QAT training (5000+ images)
    python prepare_calibration_data.py --qat

    # Custom image count
    python prepare_calibration_data.py --qat --num-images 10000

Output:
    calibration_data/fd_160/*.jpg  - 160x160 images for face detection
    calibration_data/emb_112/*.jpg - 112x112 images for embedding model
"""

import os
import sys
import zipfile
import argparse
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    import subprocess
    subprocess.check_call(["pip", "install", "opencv-python"])
    import cv2

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing requests and tqdm...")
    import subprocess
    subprocess.check_call(["pip", "install", "requests", "tqdm"])
    import requests
    from tqdm import tqdm

# Local LFW dataset path
LFW_ZIP_PATH = "LFW_dataset.zip"

# CelebA dataset URLs (img_align_celeba subset)
CELEBA_GDRIVE_ID = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
CELEBA_URL = "https://drive.google.com/uc?export=download&id=" + CELEBA_GDRIVE_ID


def extract_lfw(output_dir: str = "calibration_data") -> str:
    """
    Extract local LFW dataset.

    Args:
        output_dir: Directory to save the dataset

    Returns:
        Path to extracted LFW directory
    """
    os.makedirs(output_dir, exist_ok=True)
    extract_dir = os.path.join(output_dir, "lfw")

    # Check if already extracted
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"LFW dataset already extracted at {extract_dir}")
        return extract_dir

    # Check for local zip file
    if not os.path.exists(LFW_ZIP_PATH):
        print(f"ERROR: {LFW_ZIP_PATH} not found!")
        print("Please download LFW dataset and place it in the current directory.")
        raise FileNotFoundError(f"{LFW_ZIP_PATH} not found")

    # Extract archive
    print(f"Extracting {LFW_ZIP_PATH} to {extract_dir}...")
    with zipfile.ZipFile(LFW_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Find the actual extracted directory (may be nested)
    extracted_items = os.listdir(output_dir)
    for item in extracted_items:
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item not in ['fd_160', 'emb_112']:
            # Rename to 'lfw' for consistency
            if item != 'lfw':
                os.rename(item_path, extract_dir)
            break

    print("Extraction complete!")
    return extract_dir


def prepare_calibration_images(
    lfw_dir: str,
    output_dir: str,
    num_fd_images: int = 300,
    num_emb_images: int = 500
) -> tuple:
    """
    Prepare calibration images for model quantization.

    Args:
        lfw_dir: Path to LFW dataset directory
        output_dir: Output directory for calibration images
        num_fd_images: Number of images for face detection calibration
        num_emb_images: Number of images for embedding calibration

    Returns:
        Tuple of (fd_output_dir, emb_output_dir)
    """
    fd_output = os.path.join(output_dir, "fd_160")
    emb_output = os.path.join(output_dir, "emb_112")

    os.makedirs(fd_output, exist_ok=True)
    os.makedirs(emb_output, exist_ok=True)

    # Get all image paths
    all_images = sorted(Path(lfw_dir).rglob("*.jpg"))
    print(f"Found {len(all_images)} images in LFW dataset")

    # Select images for calibration
    fd_images = all_images[:num_fd_images]
    emb_images = all_images[:num_emb_images]

    # Process face detection calibration images (160x160)
    print(f"\nPreparing {len(fd_images)} images for face detection calibration (160x160)...")
    for i, img_path in enumerate(fd_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Resize to 160x160 (SCRFD input size)
        fd_img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)
        output_path = os.path.join(fd_output, f"{i:04d}.jpg")
        cv2.imwrite(output_path, fd_img)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(fd_images)}")

    # Process embedding calibration images (112x112)
    print(f"\nPreparing {len(emb_images)} images for embedding calibration (112x112)...")
    for i, img_path in enumerate(emb_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # LFW images are 250x250, center crop to get face region
        h, w = img.shape[:2]

        # Center crop to square
        crop_size = min(h, w)
        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        cropped = img[y_start:y_start + crop_size, x_start:x_start + crop_size]

        # Resize to 112x112 (GhostFaceNet input size)
        emb_img = cv2.resize(cropped, (112, 112), interpolation=cv2.INTER_LINEAR)
        output_path = os.path.join(emb_output, f"{i:04d}.jpg")
        cv2.imwrite(output_path, emb_img)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(emb_images)}")

    print(f"\nCalibration images prepared:")
    print(f"  Face detection: {fd_output} ({len(list(Path(fd_output).glob('*.jpg')))} images)")
    print(f"  Embedding: {emb_output} ({len(list(Path(emb_output).glob('*.jpg')))} images)")

    return fd_output, emb_output


def verify_calibration_data(output_dir: str) -> bool:
    """
    Verify that calibration data was created correctly.

    Args:
        output_dir: Calibration data directory

    Returns:
        True if verification passes
    """
    fd_dir = os.path.join(output_dir, "fd_160")
    emb_dir = os.path.join(output_dir, "emb_112")

    fd_images = list(Path(fd_dir).glob("*.jpg"))
    emb_images = list(Path(emb_dir).glob("*.jpg"))

    print("\nVerification:")

    # Check count
    if len(fd_images) < 100:
        print(f"  [WARN] Only {len(fd_images)} FD calibration images (recommend 200+)")
    else:
        print(f"  [OK] {len(fd_images)} FD calibration images")

    if len(emb_images) < 200:
        print(f"  [WARN] Only {len(emb_images)} embedding calibration images (recommend 300+)")
    else:
        print(f"  [OK] {len(emb_images)} embedding calibration images")

    # Check dimensions
    if fd_images:
        sample_fd = cv2.imread(str(fd_images[0]))
        if sample_fd.shape[:2] == (160, 160):
            print(f"  [OK] FD images are 160x160")
        else:
            print(f"  [ERROR] FD images are {sample_fd.shape[:2]}, expected (160, 160)")
            return False

    if emb_images:
        sample_emb = cv2.imread(str(emb_images[0]))
        if sample_emb.shape[:2] == (112, 112):
            print(f"  [OK] Embedding images are 112x112")
        else:
            print(f"  [ERROR] Embedding images are {sample_emb.shape[:2]}, expected (112, 112)")
            return False

    print("\nCalibration data ready for model conversion!")
    return True


def prepare_qat_data(
    lfw_dir: str,
    output_dir: str,
    num_images: int = 5000
) -> str:
    """
    Prepare extended training data for QAT.

    Uses all available LFW images (13000+) for QAT training.

    Args:
        lfw_dir: Path to LFW dataset directory
        output_dir: Output directory
        num_images: Target number of images

    Returns:
        Path to QAT training data directory
    """
    qat_output = os.path.join(output_dir, "qat_112")
    os.makedirs(qat_output, exist_ok=True)

    # Get all image paths from LFW
    all_images = sorted(Path(lfw_dir).rglob("*.jpg"))
    print(f"Found {len(all_images)} images in LFW dataset")

    # Limit to requested number
    images_to_process = all_images[:num_images]
    print(f"\nPreparing {len(images_to_process)} images for QAT training (112x112)...")

    processed = 0
    for i, img_path in enumerate(images_to_process):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # LFW images are 250x250, center crop to get face region
        h, w = img.shape[:2]

        # Center crop to square
        crop_size = min(h, w)
        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        cropped = img[y_start:y_start + crop_size, x_start:x_start + crop_size]

        # Resize to 112x112 (GhostFaceNet input size)
        resized = cv2.resize(cropped, (112, 112), interpolation=cv2.INTER_LINEAR)
        output_path = os.path.join(qat_output, f"{processed:05d}.jpg")
        cv2.imwrite(output_path, resized)
        processed += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(images_to_process)}")

    print(f"\nQAT training data prepared:")
    print(f"  Output: {qat_output}")
    print(f"  Images: {processed}")

    return qat_output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare calibration/training data for face recognition models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--qat', action='store_true',
                        help='Prepare extended data for QAT training (5000+ images)')
    parser.add_argument('--num-images', type=int, default=5000,
                        help='Number of images for QAT (default: 5000)')
    parser.add_argument('--num-fd', type=int, default=300,
                        help='Number of face detection calibration images (default: 300)')
    parser.add_argument('--num-emb', type=int, default=500,
                        help='Number of embedding calibration images (default: 500)')
    args = parser.parse_args()

    output_dir = "calibration_data"

    print("=" * 60)
    if args.qat:
        print("Face Dataset Preparation for QAT Training")
    else:
        print("LFW Calibration Data Preparation for INT8 Quantization")
    print("=" * 60)

    # Extract local LFW dataset
    lfw_dir = extract_lfw(output_dir)

    # Prepare basic calibration images
    prepare_calibration_images(
        lfw_dir=lfw_dir,
        output_dir=output_dir,
        num_fd_images=args.num_fd,
        num_emb_images=args.num_emb
    )

    # Verify basic calibration data
    verify_calibration_data(output_dir)

    # Prepare extended QAT data if requested
    if args.qat:
        print("\n" + "=" * 60)
        print("Preparing Extended QAT Training Data")
        print("=" * 60)
        qat_dir = prepare_qat_data(
            lfw_dir=lfw_dir,
            output_dir=output_dir,
            num_images=args.num_images
        )
        print(f"\nQAT training data ready at: {qat_dir}")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Run convert_scrfd.py to convert SCRFD model")
    print("  2. Run convert_ghostfacenet.py to convert GhostFaceNet model")
    if args.qat:
        print("  3. Run qat_ghostfacenet.py --calib-dir calibration_data/qat_112")
    print("=" * 60)


if __name__ == "__main__":
    main()
