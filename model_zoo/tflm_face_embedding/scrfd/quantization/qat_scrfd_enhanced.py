#!/usr/bin/env python3
"""
SCRFD Enhanced QAT (Quantization-Aware Training) with MS1M-ArcFace Dataset

This enhanced version supports:
1. MS1M-ArcFace large-scale face dataset (5.8M images)
2. Cosine learning rate schedule with warmup
3. Gradient clipping for stability
4. Best model checkpointing
5. Validation during training

Key differences from original:
- Uses MS1M instead of LFW for more diverse training data
- Samples strategically across identities for diversity
- Implements cosine annealing LR schedule
- Adds validation loss tracking

Usage:
    python qat_scrfd_enhanced.py --epochs 10 --num-images 30000

Recommended configurations:
    Quick test:   --num-images 5000  --epochs 5   (~15 min)
    Standard:     --num-images 20000 --epochs 10  (~1 hour)
    Full:         --num-images 50000 --epochs 15  (~3 hours)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import copy
import random
from typing import List, Tuple, Optional

print("=" * 60)
print("SCRFD Enhanced QAT with MS1M-ArcFace")
print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
print(f"PyTorch: {torch.__version__}")

try:
    import cv2
except ImportError:
    print("Please install opencv-python")
    sys.exit(1)

try:
    import onnxruntime as ort
    # Suppress shape mismatch warnings (harmless, caused by dynamic shapes)
    ort.set_default_logger_severity(3)  # ERROR level only
except ImportError:
    print("Please install onnxruntime")
    sys.exit(1)


# Paths - relative to script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PTH_FILE = str(SCRIPT_DIR / "scrfd_500m_kps.pth")
ONNX_REF = str(SCRIPT_DIR / "scrfd_500m_kps.onnx")
MS1M_DIR = "/Users/harvest/project/datasets/ms1m-arcface"
LFW_DIR = str(SCRIPT_DIR / "calibration_data/lfw/lfw-deepfunneled")
INPUT_SIZE = 160


# ============================================================================
# Model Definition (same as qat_scrfd_native.py)
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activate=True):
        super().__init__()
        self.activate = activate
        self.depthwise_conv = nn.Sequential()
        self.depthwise_conv.add_module('conv',
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False))
        self.depthwise_conv.add_module('bn', nn.BatchNorm2d(in_ch))
        if activate:
            self.depthwise_conv.add_module('relu', nn.ReLU(inplace=True))
        self.pointwise_conv = nn.Sequential()
        self.pointwise_conv.add_module('conv',
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
        self.pointwise_conv.add_module('bn', nn.BatchNorm2d(out_ch))
        if activate:
            self.pointwise_conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class MobileNetV1Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential()
        self.stem.add_module('0', nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ))
        self.stem.add_module('1', DepthwiseSeparableConv(16, 16, stride=1))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('0', DepthwiseSeparableConv(16, 40, stride=2))
        self.layer1.add_module('1', DepthwiseSeparableConv(40, 40, stride=1))
        self.layer2 = nn.Sequential()
        self.layer2.add_module('0', DepthwiseSeparableConv(40, 72, stride=2))
        self.layer2.add_module('1', DepthwiseSeparableConv(72, 72, stride=1))
        self.layer2.add_module('2', DepthwiseSeparableConv(72, 72, stride=1))
        self.layer3 = nn.Sequential()
        self.layer3.add_module('0', DepthwiseSeparableConv(72, 152, stride=2))
        self.layer3.add_module('1', DepthwiseSeparableConv(152, 152, stride=1))
        self.layer4 = nn.Sequential()
        self.layer4.add_module('0', DepthwiseSeparableConv(152, 288, stride=2))
        for i in range(1, 6):
            self.layer4.add_module(str(i), DepthwiseSeparableConv(288, 288, stride=1))

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class PAFPN(nn.Module):
    def __init__(self, in_channels=[40, 72, 152, 288], out_channels=16):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            ConvModule(in_ch, out_channels, 1) for in_ch in in_channels[1:]
        ])
        self.fpn_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1) for _ in range(3)
        ])
        self.downsample_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, stride=2, padding=1) for _ in range(2)
        ])
        self.pafpn_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1) for _ in range(2)
        ])

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, [c2, c3, c4])]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        for i in range(len(fpn_outs) - 1):
            fpn_outs[i + 1] = fpn_outs[i + 1] + self.downsample_convs[i](fpn_outs[i])
        outs = [fpn_outs[0]]
        for i in range(1, len(fpn_outs)):
            outs.append(self.pafpn_convs[i - 1](fpn_outs[i]))
        return outs


class SCRFDHead(nn.Module):
    def __init__(self, in_channels=16, feat_channels=64, num_anchors=2):
        super().__init__()
        self.num_anchors = num_anchors
        self.strides = [8, 16, 32]
        self.cls_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        self.stride_kps = nn.ModuleDict()
        for stride in self.strides:
            key = f"({stride}, {stride})"
            self.cls_stride_convs[key] = nn.Sequential()
            self.cls_stride_convs[key].add_module('0',
                DepthwiseSeparableConv(in_channels, feat_channels, stride=1))
            self.cls_stride_convs[key].add_module('1',
                DepthwiseSeparableConv(feat_channels, feat_channels, stride=1))
            self.stride_cls[key] = nn.Conv2d(feat_channels, num_anchors, 3, padding=1)
            self.stride_reg[key] = nn.Conv2d(feat_channels, num_anchors * 4, 3, padding=1)
            self.stride_kps[key] = nn.Conv2d(feat_channels, num_anchors * 10, 3, padding=1)

    def forward(self, feats):
        all_cls, all_reg, all_kps = [], [], []
        for i, (feat, stride) in enumerate(zip(feats, self.strides)):
            key = f"({stride}, {stride})"
            x = self.cls_stride_convs[key](feat)
            cls = self.stride_cls[key](x)
            reg = self.stride_reg[key](x)
            kps = self.stride_kps[key](x)
            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 1)
            cls = torch.clamp(cls, 0.0, 1.0)  # Clamp to [0, 1] probability range
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
            kps = kps.permute(0, 2, 3, 1).reshape(B, -1, 10)
            all_cls.append(cls)
            all_reg.append(reg)
            all_kps.append(kps)
        return all_cls + all_reg + all_kps


class SCRFD(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV1Backbone()
        self.neck = PAFPN(in_channels=[40, 72, 152, 288], out_channels=16)
        self.bbox_head = SCRFDHead(in_channels=16, feat_channels=64)

    def forward(self, x):
        features = self.backbone(x)
        fpn_feats = self.neck(features)
        outputs = self.bbox_head(fpn_feats)
        return outputs


# ============================================================================
# Weight Loading (same as original)
# ============================================================================

def load_pretrained_weights(model, checkpoint_path):
    """Load pretrained weights with key mapping."""
    print(f"\nLoading weights from {checkpoint_path}...")
    import re

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    model_dict = model.state_dict()
    new_dict = {}
    matched, unmatched = 0, 0

    def map_key(ckpt_key):
        model_key = ckpt_key
        for prefix in ['backbone.stem.1', 'backbone.layer1', 'backbone.layer2',
                       'backbone.layer3', 'backbone.layer4']:
            if prefix in ckpt_key:
                pattern = rf'({prefix})\.(\d+)\.(\d+)\.(.*)'
                match = re.match(pattern, ckpt_key)
                if match:
                    block_prefix, block_idx, sub_idx, attr = match.groups()
                    sub_idx = int(sub_idx)
                    if sub_idx == 0:
                        model_key = f"{block_prefix}.{block_idx}.depthwise_conv.conv.{attr}"
                    elif sub_idx == 1:
                        model_key = f"{block_prefix}.{block_idx}.depthwise_conv.bn.{attr}"
                    elif sub_idx == 3:
                        model_key = f"{block_prefix}.{block_idx}.pointwise_conv.conv.{attr}"
                    elif sub_idx == 4:
                        model_key = f"{block_prefix}.{block_idx}.pointwise_conv.bn.{attr}"
                    return model_key
                if prefix == 'backbone.stem.1':
                    pattern = rf'(backbone\.stem\.1)\.(\d+)\.(.*)'
                    match = re.match(pattern, ckpt_key)
                    if match:
                        stem_prefix, sub_idx, attr = match.groups()
                        sub_idx = int(sub_idx)
                        if sub_idx == 0:
                            model_key = f"backbone.stem.1.depthwise_conv.conv.{attr}"
                        elif sub_idx == 1:
                            model_key = f"backbone.stem.1.depthwise_conv.bn.{attr}"
                        elif sub_idx == 3:
                            model_key = f"backbone.stem.1.pointwise_conv.conv.{attr}"
                        elif sub_idx == 4:
                            model_key = f"backbone.stem.1.pointwise_conv.bn.{attr}"
                        return model_key
        return model_key

    for ckpt_key, ckpt_val in state_dict.items():
        model_key = map_key(ckpt_key)
        if model_key in model_dict:
            if ckpt_val.shape == model_dict[model_key].shape:
                new_dict[model_key] = ckpt_val
                matched += 1
            else:
                unmatched += 1
        else:
            unmatched += 1

    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    model.load_state_dict(new_dict, strict=False)
    return model


# ============================================================================
# Data Loading - MS1M-ArcFace Support
# ============================================================================

def sample_ms1m_images(
    ms1m_dir: str,
    num_images: int,
    input_size: int,
    seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """
    Sample images from MS1M-ArcFace dataset.

    Strategy: Sample from diverse identities rather than many images per identity.
    This provides better coverage of face variations.

    Args:
        ms1m_dir: Path to MS1M-ArcFace dataset
        num_images: Total number of images to sample
        input_size: Target image size
        seed: Random seed for reproducibility

    Returns:
        Tuple of (images array, list of paths)
    """
    print(f"\nSampling {num_images} images from MS1M-ArcFace...")
    ms1m_path = Path(ms1m_dir)

    if not ms1m_path.exists():
        print(f"  MS1M directory not found: {ms1m_dir}")
        return None, []

    # Get all identity folders
    identity_dirs = [d for d in ms1m_path.iterdir() if d.is_dir()]
    print(f"  Found {len(identity_dirs)} identities")

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Strategy: Sample 1-3 images per identity from many identities
    # This maximizes diversity
    images_per_identity = 2
    num_identities = num_images // images_per_identity + 1

    # Sample identities
    if num_identities >= len(identity_dirs):
        selected_identities = identity_dirs
    else:
        selected_identities = random.sample(identity_dirs, num_identities)

    print(f"  Sampling from {len(selected_identities)} identities ({images_per_identity} imgs each)")

    images = []
    paths = []

    for i, identity_dir in enumerate(selected_identities):
        if len(images) >= num_images:
            break

        # Get images in this identity folder
        img_files = list(identity_dir.glob("*.jpg"))
        if not img_files:
            continue

        # Sample images from this identity
        sampled = random.sample(img_files, min(images_per_identity, len(img_files)))

        for img_path in sampled:
            if len(images) >= num_images:
                break

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # MS1M images are typically 112x112 aligned faces
            # Resize to target size
            img = cv2.resize(img, (input_size, input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to [-1, 1] range (matching SCRFD ONNX teacher model)
            img = (img.astype(np.float32) - 127.5) / 128.0

            images.append(img)
            paths.append(str(img_path))

        if (i + 1) % 5000 == 0:
            print(f"    Progress: {i+1}/{len(selected_identities)} identities, {len(images)} images")

    print(f"  Loaded {len(images)} images")
    return np.array(images), paths


def load_lfw_images(lfw_dir: str, input_size: int, max_images: int = 2000) -> np.ndarray:
    """Load images from LFW directory for validation."""
    images = []
    lfw_path = Path(lfw_dir)

    if not lfw_path.exists():
        print(f"LFW directory not found: {lfw_dir}")
        return None

    all_images = sorted(lfw_path.rglob("*.jpg"))
    print(f"Found {len(all_images)} images in LFW")

    if len(all_images) > max_images:
        step = len(all_images) // max_images
        all_images = all_images[::step][:max_images]

    print(f"Loading {len(all_images)} images...")

    for i, img_path in enumerate(all_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 128.0
        images.append(img)

    return np.array(images) if images else None


# ============================================================================
# ONNX Teacher
# ============================================================================

class ONNXTeacher:
    """ONNX model for output distillation."""
    def __init__(self, onnx_path: str, input_size: int):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        print(f"  Teacher input: {self.input_name}")

    def predict(self, images_nhwc: np.ndarray):
        """Get teacher outputs for NHWC normalized images."""
        images_nchw = np.transpose(images_nhwc, (0, 3, 1, 2)).astype(np.float32)
        outputs = []
        for i in range(len(images_nchw)):
            out = self.session.run(None, {self.input_name: images_nchw[i:i+1]})
            outputs.append(out)
        stacked = []
        for j in range(len(outputs[0])):
            batch_outputs = [np.expand_dims(o[j], axis=0) for o in outputs]
            stacked.append(np.concatenate(batch_outputs, axis=0))
        return stacked


# ============================================================================
# Model Fusion and QAT Setup
# ============================================================================

def fuse_model(model):
    """Fuse Conv+BN+ReLU patterns in the model."""
    print("\nFusing Conv+BN+ReLU layers...")
    from torch.ao.quantization import fuse_modules

    model.eval()
    fuse_modules(model.backbone.stem[0], ['0', '1', '2'], inplace=True)

    def fuse_dwsep(block):
        if hasattr(block.depthwise_conv, 'relu'):
            fuse_modules(block.depthwise_conv, ['conv', 'bn', 'relu'], inplace=True)
        else:
            fuse_modules(block.depthwise_conv, ['conv', 'bn'], inplace=True)
        if hasattr(block.pointwise_conv, 'relu'):
            fuse_modules(block.pointwise_conv, ['conv', 'bn', 'relu'], inplace=True)
        else:
            fuse_modules(block.pointwise_conv, ['conv', 'bn'], inplace=True)

    fuse_dwsep(model.backbone.stem[1])
    for layer in [model.backbone.layer1, model.backbone.layer2,
                  model.backbone.layer3, model.backbone.layer4]:
        for block in layer:
            fuse_dwsep(block)
    for stride in [8, 16, 32]:
        key = f"({stride}, {stride})"
        for block in model.bbox_head.cls_stride_convs[key]:
            fuse_dwsep(block)

    print("  Fusion complete")
    return model


def apply_qat(model):
    """Apply PyTorch native QAT after fusion."""
    print("\nApplying QAT configuration...")

    import platform
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        backend = 'qnnpack'
        torch.backends.quantized.engine = 'qnnpack'
    else:
        backend = 'fbgemm'
        torch.backends.quantized.engine = 'fbgemm'

    print(f"  Using backend: {backend}")
    model.train()
    qconfig = get_default_qat_qconfig(backend)
    model.qconfig = qconfig
    model_prepared = prepare_qat(model, inplace=False)
    print("  QAT preparation complete")
    return model_prepared


# ============================================================================
# Enhanced QAT Training
# ============================================================================

def train_qat_enhanced(
    model,
    teacher,
    train_data: np.ndarray,
    val_data: Optional[np.ndarray],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    warmup_epochs: int = 1,
    grad_clip: float = 1.0,
    output_prefix: str = "scrfd_qat"
):
    """
    Enhanced QAT training with:
    - Cosine annealing LR schedule with warmup
    - Gradient clipping
    - Validation loss tracking
    - Best model checkpointing
    """
    print(f"\n{'='*60}")
    print("Enhanced QAT Training")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial LR: {lr}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Gradient clip: {grad_clip}")
    print(f"  Train samples: {len(train_data)}")
    if val_data is not None:
        print(f"  Val samples: {len(val_data)}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate schedulers
    n_samples = len(train_data)
    n_batches = n_samples // batch_size
    total_steps = epochs * n_batches
    warmup_steps = min(warmup_epochs * n_batches, total_steps // 2)  # Cap warmup at half of total

    # Ensure we have enough steps for scheduler
    cosine_steps = max(1, total_steps - warmup_steps)

    # Warmup scheduler (linear)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(1, warmup_steps)
    )

    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=lr * 0.01
    )

    # Combined scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[max(1, warmup_steps)]
    )

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        indices = np.random.permutation(n_samples)

        for batch_idx in range(n_batches):
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_images_np = train_data[batch_indices]
            batch_images = torch.from_numpy(
                np.transpose(batch_images_np, (0, 3, 1, 2))
            ).float()

            # Get teacher outputs
            teacher_outputs = teacher.predict(batch_images_np)

            # Forward pass
            optimizer.zero_grad()
            student_outputs = model(batch_images)

            # Compute distillation loss
            total_loss = torch.tensor(0.0, requires_grad=True)
            n_matched = 0
            for t_out, s_out in zip(teacher_outputs, student_outputs):
                t_tensor = torch.from_numpy(t_out).float()
                if len(t_tensor.shape) == 2 and len(s_out.shape) == 3:
                    t_tensor = t_tensor.unsqueeze(0).expand_as(s_out)
                if t_tensor.shape == s_out.shape:
                    loss = F.mse_loss(s_out, t_tensor)
                    total_loss = total_loss + loss
                    n_matched += 1

            if n_matched == 0:
                continue

            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()

            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  E{epoch+1} B{batch_idx}/{n_batches} "
                      f"Loss: {total_loss.item():.6f} LR: {current_lr:.2e}")

        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0

        # Validation
        val_loss = 0.0
        if val_data is not None and len(val_data) > 0:
            model.eval()
            with torch.no_grad():
                val_batches = len(val_data) // batch_size
                for batch_idx in range(val_batches):
                    batch_images_np = val_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    batch_images = torch.from_numpy(
                        np.transpose(batch_images_np, (0, 3, 1, 2))
                    ).float()
                    teacher_outputs = teacher.predict(batch_images_np)
                    student_outputs = model(batch_images)

                    for t_out, s_out in zip(teacher_outputs, student_outputs):
                        t_tensor = torch.from_numpy(t_out).float()
                        if len(t_tensor.shape) == 2 and len(s_out.shape) == 3:
                            t_tensor = t_tensor.unsqueeze(0).expand_as(s_out)
                        if t_tensor.shape == s_out.shape:
                            val_loss += F.mse_loss(s_out, t_tensor).item()

                val_loss /= (val_batches * 9) if val_batches > 0 else 1  # 9 outputs

        print(f"\n  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            checkpoint_path = f"{output_prefix}_best_checkpoint.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  >>> Best model saved: {checkpoint_path}")

    print(f"\n  Training complete! Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")

    # Load best model
    best_checkpoint = f"{output_prefix}_best_checkpoint.pth"
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded best model from epoch {checkpoint['epoch']+1}")

    return model


# ============================================================================
# Export Functions
# ============================================================================

def export_to_onnx(model, output_path: str, input_size: int = 160):
    """Export model to ONNX with squeezed outputs (no batch dimension)."""
    import onnx
    from onnx import helper, TensorProto

    print(f"\nExporting to ONNX: {output_path}")
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # First export with batch dimension
    temp_path = output_path.replace(".onnx", "_temp.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        temp_path,
        input_names=['input'],
        output_names=[f'output_{i}' for i in range(9)],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )

    # Post-process: add Squeeze nodes to remove batch dimension
    print("  Adding Squeeze nodes to remove batch dimension...")
    onnx_model = onnx.load(temp_path)
    graph = onnx_model.graph

    # Create new outputs with Squeeze
    new_outputs = []
    for i, output in enumerate(graph.output):
        # Create squeeze axes constant
        axes_name = f"squeeze_axes_{i}"
        axes_tensor = helper.make_tensor(axes_name, TensorProto.INT64, [1], [0])
        graph.initializer.append(axes_tensor)

        # Create Squeeze node
        squeeze_output = f"{output.name}_squeezed"
        squeeze_node = helper.make_node(
            'Squeeze',
            inputs=[output.name, axes_name],
            outputs=[squeeze_output],
            name=f"Squeeze_{i}"
        )
        graph.node.append(squeeze_node)

        # Get output shape without batch dimension
        old_shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        new_shape = old_shape[1:]  # Remove first dimension

        # Create new output
        new_output = helper.make_tensor_value_info(
            squeeze_output,
            output.type.tensor_type.elem_type,
            new_shape
        )
        new_outputs.append(new_output)

    # Replace outputs
    while len(graph.output) > 0:
        graph.output.pop()
    for new_out in new_outputs:
        graph.output.append(new_out)

    # Save modified model
    onnx.save(onnx_model, output_path)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    print(f"  Saved: {output_path}")
    return output_path


def convert_to_tflite(onnx_path: str, output_path: str, calib_data: np.ndarray):
    """Convert ONNX to INT8 TFLite.

    Training uses [-1, 1] normalization (to match SCRFD teacher model).
    Device sends [0, 1] normalized data (uint8 / 255).

    We convert calibration data from [-1, 1] to [0, 1] so the TFLite model
    accepts [0, 1] input matching the device.
    """
    import onnx2tf
    import shutil

    print(f"\nConverting to TFLite INT8...")

    calib_npy = "qat_calib_temp.npy"
    # Training data is in [-1, 1], convert to [0, 1] for TFLite
    # This makes TFLite model accept [0, 1] input (matching device)
    calib_01 = (calib_data[:200] * 0.5 + 0.5).astype(np.float32)
    np.save(calib_npy, calib_01)
    print(f"  Training data range: [{calib_data.min():.2f}, {calib_data.max():.2f}]")
    print(f"  Calibration (converted to [0,1]): [{calib_01.min():.2f}, {calib_01.max():.2f}]")

    output_dir = "qat_tflite_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    try:
        # mean=0, std=1 because calibration data is now [0, 1]
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            copy_onnx_input_output_names_to_tflite=True,
            output_integer_quantized_tflite=True,
            custom_input_op_name_np_data_path=[
                ["input", calib_npy, [[[[0.0, 0.0, 0.0]]]], [[[[1.0, 1.0, 1.0]]]]]
            ],
            non_verbose=True,
        )

        for pattern in ["*full_integer_quant.tflite", "*integer_quant.tflite"]:
            matches = list(Path(output_dir).glob(pattern))
            if matches:
                shutil.copy(matches[0], output_path)
                print(f"  Saved: {output_path}")
                break
    finally:
        if os.path.exists(calib_npy):
            os.remove(calib_npy)

    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced SCRFD QAT with MS1M-ArcFace")
    parser.add_argument('--pth', default=PTH_FILE, help='Pretrained weights')
    parser.add_argument('--onnx-ref', default=ONNX_REF, help='ONNX reference for distillation')
    parser.add_argument('--output', default='scrfd_qat_enhanced.tflite', help='Output TFLite path')
    parser.add_argument('--ms1m-dir', default=MS1M_DIR, help='MS1M-ArcFace dataset path')

    # Training parameters
    parser.add_argument('--num-images', type=int, default=20000,
                        help='Number of training images (default: 20000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=1,
                        help='Warmup epochs (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Options
    parser.add_argument('--skip-vela', action='store_true', help='Skip Vela compilation')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Configuration")
    print(f"{'='*60}")
    print(f"  PTH: {args.pth}")
    print(f"  ONNX Reference: {args.onnx_ref}")
    print(f"  MS1M Dir: {args.ms1m_dir}")
    print(f"  Output: {args.output}")
    print(f"  Num Images: {args.num_images}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Val Split: {args.val_split}")

    # Check files
    if not os.path.exists(args.pth):
        print(f"\nError: PTH file not found: {args.pth}")
        return 1
    if not os.path.exists(args.onnx_ref):
        print(f"\nError: ONNX reference not found: {args.onnx_ref}")
        return 1

    # Load training data from MS1M
    train_data, _ = sample_ms1m_images(
        args.ms1m_dir,
        args.num_images,
        INPUT_SIZE,
        seed=args.seed
    )

    if train_data is None or len(train_data) < 100:
        print("\nNot enough training images from MS1M!")
        print("Falling back to LFW...")
        train_data = load_lfw_images(LFW_DIR, INPUT_SIZE, args.num_images)
        if train_data is None:
            print("No training data available!")
            return 1

    # Split train/val
    n_val = int(len(train_data) * args.val_split)
    indices = np.random.permutation(len(train_data))
    val_data = train_data[indices[:n_val]]
    train_data = train_data[indices[n_val:]]
    print(f"\nData split: Train={len(train_data)}, Val={len(val_data)}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating SCRFD model...")
    model = SCRFD()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Load pretrained weights
    model = load_pretrained_weights(model, args.pth)

    # Create teacher
    print("\nCreating ONNX teacher...")
    teacher = ONNXTeacher(args.onnx_ref, INPUT_SIZE)

    # Fuse and prepare for QAT
    model = fuse_model(model)
    model_qat = apply_qat(model)

    # Train with enhanced QAT
    output_prefix = args.output.replace('.tflite', '')
    model_qat = train_qat_enhanced(
        model_qat,
        teacher,
        train_data,
        val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        output_prefix=output_prefix
    )

    # Export
    print("\n" + "=" * 60)
    print("Exporting model...")

    # Create fresh model for export
    model_export = SCRFD()
    model_export = load_pretrained_weights(model_export, args.pth)
    model_export = fuse_model(model_export)

    # Copy trained weights
    import re
    qat_state = model_qat.state_dict()
    export_state = model_export.state_dict()

    def get_qat_key(export_key):
        qat_key = export_key
        qat_key = re.sub(r'\.conv\.0\.(weight|bias)', r'.conv.\1', qat_key)
        qat_key = re.sub(r'backbone\.stem\.0\.0\.0\.(weight|bias)', r'backbone.stem.0.0.\1', qat_key)
        return qat_key

    matched = 0
    for key in export_state.keys():
        qat_key = get_qat_key(key)
        if qat_key in qat_state:
            if export_state[key].shape == qat_state[qat_key].shape:
                export_state[key] = qat_state[qat_key]
                matched += 1
        elif key in qat_state:
            if export_state[key].shape == qat_state[key].shape:
                export_state[key] = qat_state[key]
                matched += 1

    print(f"  Copied {matched}/{len(export_state)} weight tensors from QAT model")
    model_export.load_state_dict(export_state)
    model_export.eval()

    onnx_qat = args.output.replace('.tflite', '.onnx')
    export_to_onnx(model_export, onnx_qat, INPUT_SIZE)

    # Convert to TFLite
    convert_to_tflite(onnx_qat, args.output, train_data)

    # Compile with Vela
    if not args.skip_vela:
        print("\nCompiling with Vela...")
        import subprocess
        vela_out = args.output.replace('.tflite', '_vela.tflite')
        result = subprocess.run([
            sys.executable, '-m', 'ethosu.vela',
            '--accelerator-config', 'ethos-u55-64',
            '--optimise', 'Performance',
            args.output,
            '--output-dir', '.'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  Vela output: {vela_out}")
        else:
            print(f"  Vela error: {result.stderr}")

    print("\n" + "=" * 60)
    print("Enhanced QAT Complete!")
    print(f"Output: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
