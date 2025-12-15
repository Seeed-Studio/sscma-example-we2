#!/usr/bin/env python3
"""
Download and prepare face datasets for QAT training.

Supports:
- LFW (Labeled Faces in the Wild) - 13K images
- CASIA-WebFace aligned - 500K images (requires manual download)
- Custom directory of face images

Usage:
    # Download and prepare LFW
    python download_datasets.py --dataset lfw

    # Use existing CASIA-WebFace
    python download_datasets.py --dataset casia --input /path/to/CASIA-WebFace

    # Use custom face image directory
    python download_datasets.py --dataset custom --input /path/to/faces

    # Limit number of images
    python download_datasets.py --dataset lfw --max-images 50000
"""

import os
import sys
import argparse
import tarfile
import zipfile
from pathlib import Path

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "opencv-python", "numpy", "tqdm", "requests"])
    import cv2
    import numpy as np
    from tqdm import tqdm

# Dataset configurations
# Multiple mirror URLs for reliability
DATASETS = {
    "lfw": {
        "urls": [
            # Figshare - used by TensorFlow Datasets and scikit-learn (MOST RELIABLE)
            # Source: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/lfw/lfw_dataset_builder.py
            "https://ndownloader.figshare.com/files/5976018",
            # Original source
            "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        ],
        "filename": "lfw.tgz",
        "size": "173MB",
        "images": 13233,
        "description": "Labeled Faces in the Wild - standard benchmark"
    },
    "lfw_deepfunneled": {
        "urls": [
            "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
        ],
        "filename": "lfw-deepfunneled.tgz",
        "size": "103MB",
        "images": 13233,
        "description": "LFW with deep funneling alignment"
    }
}


def download_file(url: str, output_path: str, description: str = "Downloading"):
    """Download file with progress bar."""
    import requests

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_lfw(output_dir: str = "datasets", variant: str = "lfw") -> str:
    """
    Download LFW dataset.

    Args:
        output_dir: Directory to save dataset
        variant: "lfw" or "lfw_deepfunneled"

    Returns:
        Path to extracted directory
    """
    os.makedirs(output_dir, exist_ok=True)

    config = DATASETS[variant]
    archive_path = os.path.join(output_dir, config["filename"])
    extract_dir = os.path.join(output_dir, variant.replace("_", "-"))

    # Check if already extracted
    if os.path.exists(extract_dir) and len(list(Path(extract_dir).rglob("*.jpg"))) > 1000:
        print(f"LFW already extracted at {extract_dir}")
        return extract_dir

    # Download if needed - try multiple URLs
    if not os.path.exists(archive_path):
        # Support both "urls" (list) and legacy "url" (string)
        if "urls" in config:
            urls = config["urls"]
        elif "url" in config:
            urls = [config["url"]]
        else:
            raise RuntimeError(f"No URL configured for {variant}")

        downloaded = False
        for url in urls:
            try:
                print(f"Downloading {variant} ({config['size']}) from {url[:60]}...")
                download_file(url, archive_path, f"Downloading {variant}")
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
                print("  Trying next mirror...")
                continue

        if not downloaded:
            raise RuntimeError(f"Failed to download {variant} from all mirrors")

    # Verify and extract
    print(f"Extracting {archive_path}...")

    # Check file type and extract accordingly
    with open(archive_path, 'rb') as f:
        magic = f.read(4)

    if magic[:2] == b'\x1f\x8b':  # gzip magic number
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_dir)
    elif magic[:4] == b'PK\x03\x04':  # zip magic number
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        # Invalid file - delete and raise error
        with open(archive_path, 'r', errors='ignore') as f:
            content = f.read(500)
        os.remove(archive_path)
        print(f"  Invalid archive deleted. Content preview: {content[:100]}")
        raise RuntimeError(
            f"Downloaded file is not a valid archive.\n"
            f"Please download LFW manually and upload to cloud:\n"
            f"  wget http://vis-www.cs.umass.edu/lfw/lfw.tgz\n"
            f"  # Upload lfw.tgz to datasets/ directory\n"
            f"Or upload your local LFW_dataset.zip and use:\n"
            f"  unzip LFW_dataset.zip -d datasets/\n"
            f"  python download_datasets.py --dataset custom --input datasets/lfw --output calibration_data/qat_112"
        )

    # Find extracted directory
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("lfw"):
            if item != variant.replace("_", "-"):
                os.rename(item_path, extract_dir)
            break

    print(f"Extracted to {extract_dir}")
    return extract_dir


def collect_images(input_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """Recursively collect all image files."""
    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(extensions):
                images.append(os.path.join(root, f))
    return sorted(images)


def resize_and_save(img_path: str, output_path: str, target_size: int = 112) -> bool:
    """
    Load, resize (center crop), and save image.

    Returns True if successful.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False

        h, w = img.shape[:2]

        # Center crop to square
        if h != w:
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

        # Resize to target size
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # Save
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    target_size: int = 112,
    max_images: int = None
) -> int:
    """
    Prepare face dataset: resize to target_size x target_size.

    Args:
        input_dir: Directory containing face images
        output_dir: Output directory for processed images
        target_size: Target image size (default: 112 for GhostFaceNet)
        max_images: Maximum number of images to process

    Returns:
        Number of images processed
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect images
    print(f"Scanning {input_dir} for images...")
    images = collect_images(input_dir)
    print(f"Found {len(images)} images")

    if max_images and len(images) > max_images:
        # Shuffle and limit
        np.random.seed(42)
        np.random.shuffle(images)
        images = images[:max_images]
        print(f"Limited to {max_images} images")

    # Process images
    count = 0
    for img_path in tqdm(images, desc=f"Preparing {target_size}x{target_size} images"):
        output_path = os.path.join(output_dir, f"{count:06d}.jpg")
        if resize_and_save(img_path, output_path, target_size):
            count += 1

    print(f"\nPrepared {count} images in {output_dir}")
    return count


def prepare_with_face_detection(
    input_dir: str,
    output_dir: str,
    target_size: int = 112,
    max_images: int = None
) -> int:
    """
    Prepare dataset with face detection and alignment.
    Requires insightface package.

    Args:
        input_dir: Directory containing images (may have multiple faces)
        output_dir: Output directory for aligned faces
        target_size: Target image size
        max_images: Maximum number of faces to extract

    Returns:
        Number of faces extracted
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("insightface not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                              "insightface", "onnxruntime"])
        from insightface.app import FaceAnalysis

    os.makedirs(output_dir, exist_ok=True)

    # Initialize face detector
    print("Initializing face detector...")
    app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(160, 160))

    # Collect images
    images = collect_images(input_dir)
    print(f"Found {len(images)} images to process")

    count = 0
    for img_path in tqdm(images, desc="Detecting and aligning faces"):
        if max_images and count >= max_images:
            break

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Detect faces
        faces = app.get(img)

        for face in faces:
            if max_images and count >= max_images:
                break

            # Get aligned face (112x112)
            # insightface provides norm_crop for alignment
            try:
                from insightface.utils import face_align
                aligned = face_align.norm_crop(img, face.kps, image_size=target_size)

                output_path = os.path.join(output_dir, f"{count:06d}.jpg")
                cv2.imwrite(output_path, aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
                count += 1
            except Exception as e:
                # Fallback: crop bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                face_img = img[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                face_img = cv2.resize(face_img, (target_size, target_size))
                output_path = os.path.join(output_dir, f"{count:06d}.jpg")
                cv2.imwrite(output_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                count += 1

    print(f"\nExtracted {count} aligned faces in {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare face datasets for QAT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download LFW and prepare for training
    python download_datasets.py --dataset lfw --output calibration_data/qat_112

    # Use CASIA-WebFace (must download manually first)
    python download_datasets.py --dataset casia --input /path/to/CASIA-WebFace --max-images 100000

    # Use any directory of face images
    python download_datasets.py --dataset custom --input /path/to/faces --max-images 50000

    # Use face detection for non-aligned images
    python download_datasets.py --dataset custom --input /path/to/photos --detect-faces
        """
    )

    parser.add_argument('--dataset', type=str, default='lfw',
                        choices=['lfw', 'lfw_deepfunneled', 'casia', 'custom'],
                        help='Dataset to use (default: lfw)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input directory for casia/custom datasets')
    parser.add_argument('--output', type=str, default='calibration_data/qat_112',
                        help='Output directory for processed images')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to prepare')
    parser.add_argument('--size', type=int, default=112,
                        help='Target image size (default: 112)')
    parser.add_argument('--detect-faces', action='store_true',
                        help='Use face detection and alignment (slower but better)')
    parser.add_argument('--download-dir', type=str, default='datasets',
                        help='Directory to download datasets to')

    args = parser.parse_args()

    print("=" * 60)
    print("Face Dataset Preparation for QAT Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Target size: {args.size}x{args.size}")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    print("=" * 60)

    # Get input directory
    if args.dataset in ['lfw', 'lfw_deepfunneled']:
        input_dir = download_lfw(args.download_dir, args.dataset)
    elif args.dataset in ['casia', 'custom']:
        if not args.input:
            print(f"ERROR: --input required for {args.dataset} dataset")
            return 1
        if not os.path.exists(args.input):
            print(f"ERROR: Input directory not found: {args.input}")
            return 1
        input_dir = args.input
    else:
        print(f"ERROR: Unknown dataset: {args.dataset}")
        return 1

    # Prepare dataset
    if args.detect_faces:
        count = prepare_with_face_detection(
            input_dir, args.output, args.size, args.max_images
        )
    else:
        count = prepare_dataset(
            input_dir, args.output, args.size, args.max_images
        )

    print("\n" + "=" * 60)
    print(f"Dataset preparation complete!")
    print(f"  Images: {count}")
    print(f"  Output: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
