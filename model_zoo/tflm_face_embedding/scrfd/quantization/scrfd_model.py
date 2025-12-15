"""
SCRFD Model Implementation for QAT

Standalone PyTorch implementation matching the checkpoint structure.
No mmdetection dependency required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """Conv + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: DWConv3x3 + Conv1x1"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise conv
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        # Pointwise conv
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class MobileNetV1Backbone(nn.Module):
    """
    MobileNetV1-like backbone for SCRFD.
    Channels: 16 -> 16 -> 40 -> 72 -> 152 -> 288
    """
    def __init__(self):
        super().__init__()

        # Stem: Conv3x3 + DepthwiseSeparable
        self.stem = nn.Sequential(
            # stem.0: Conv3x3 -> BN -> ReLU
            ConvBNReLU(3, 16, 3, stride=2, padding=1),
            # stem.1: DWConv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> ReLU
            DepthwiseSeparableConv(16, 16, stride=1),
        )

        # Layer1: 16 -> 40, stride=2, 2 blocks
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv(16, 40, stride=2),
            DepthwiseSeparableConv(40, 40, stride=1),
        )

        # Layer2: 40 -> 72, stride=2, 3 blocks
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv(40, 72, stride=2),
            DepthwiseSeparableConv(72, 72, stride=1),
            DepthwiseSeparableConv(72, 72, stride=1),
        )

        # Layer3: 72 -> 152, stride=2, 2 blocks
        self.layer3 = nn.Sequential(
            DepthwiseSeparableConv(72, 152, stride=2),
            DepthwiseSeparableConv(152, 152, stride=1),
        )

        # Layer4: 152 -> 288, stride=2, 6 blocks
        self.layer4 = nn.Sequential(
            DepthwiseSeparableConv(152, 288, stride=2),
            DepthwiseSeparableConv(288, 288, stride=1),
            DepthwiseSeparableConv(288, 288, stride=1),
            DepthwiseSeparableConv(288, 288, stride=1),
            DepthwiseSeparableConv(288, 288, stride=1),
            DepthwiseSeparableConv(288, 288, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)       # /2, 16ch
        c1 = self.layer1(x)    # /4, 40ch
        c2 = self.layer2(c1)   # /8, 72ch
        c3 = self.layer3(c2)   # /16, 152ch
        c4 = self.layer4(c3)   # /32, 288ch
        return c1, c2, c3, c4  # Return multi-scale features


class PAFPN(nn.Module):
    """
    Path Aggregation Feature Pyramid Network.
    Input: [40, 72, 152, 288] channels at strides [4, 8, 16, 32]
    Output: 3 feature maps at strides [8, 16, 32] with 16 channels each
    """
    def __init__(self, in_channels=[40, 72, 152, 288], out_channels=16):
        super().__init__()

        # Lateral convs (1x1 to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels[1:]  # Skip first
        ])

        # FPN convs (3x3 after lateral)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(3)
        ])

        # Downsample convs for bottom-up path
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) for _ in range(2)
        ])

        # PAFPN output convs
        self.pafpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(2)
        ])

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs  # [40, 72, 152, 288] channels

        # Top-down path (FPN)
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, [c2, c3, c4])]

        # Build top-down
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )

        # FPN outputs
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Bottom-up path (PAN)
        for i in range(len(fpn_outs) - 1):
            fpn_outs[i + 1] = fpn_outs[i + 1] + self.downsample_convs[i](fpn_outs[i])

        # Final outputs
        outs = [fpn_outs[0]]  # stride 8
        for i in range(1, len(fpn_outs)):
            outs.append(self.pafpn_convs[i - 1](fpn_outs[i]))

        return outs  # 3 feature maps at strides [8, 16, 32]


class SCRFDHead(nn.Module):
    """
    SCRFD detection head with keypoints.
    Outputs: scores, boxes, keypoints for each stride
    """
    def __init__(self, in_channels=16, feat_channels=64, num_anchors=2):
        super().__init__()

        self.num_anchors = num_anchors
        self.strides = [8, 16, 32]

        # Shared convs for each stride
        self.cls_stride_convs = nn.ModuleList()
        for _ in self.strides:
            self.cls_stride_convs.append(nn.Sequential(
                # DWConv + Conv
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, feat_channels, 1, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
                # Second DWConv + Conv
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1, groups=feat_channels, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_channels, feat_channels, 1, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.ReLU(inplace=True),
            ))

        # Output heads for each stride
        self.stride_cls = nn.ModuleList([
            nn.Conv2d(feat_channels, num_anchors, 1) for _ in self.strides
        ])
        self.stride_reg = nn.ModuleList([
            nn.Conv2d(feat_channels, num_anchors * 4, 1) for _ in self.strides
        ])
        self.stride_kps = nn.ModuleList([
            nn.Conv2d(feat_channels, num_anchors * 10, 1) for _ in self.strides
        ])

    def forward(self, feats):
        """
        Args:
            feats: List of 3 feature maps from PAFPN

        Returns:
            List of (scores, boxes, keypoints) for each stride
        """
        outputs = []
        for i, feat in enumerate(feats):
            x = self.cls_stride_convs[i](feat)

            cls = self.stride_cls[i](x)  # [B, 2, H, W]
            reg = self.stride_reg[i](x)  # [B, 8, H, W]
            kps = self.stride_kps[i](x)  # [B, 20, H, W]

            # Reshape to [N, C] format
            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 1)  # [B, H*W*2, 1]
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, H*W*2, 4]
            kps = kps.permute(0, 2, 3, 1).reshape(B, -1, 10) # [B, H*W*2, 10]

            outputs.append((cls, reg, kps))

        return outputs


class SCRFD(nn.Module):
    """
    Complete SCRFD model for face detection with keypoints.
    """
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV1Backbone()
        self.neck = PAFPN(in_channels=[40, 72, 152, 288], out_channels=16)
        self.bbox_head = SCRFDHead(in_channels=16, feat_channels=64)

    def forward(self, x):
        # Backbone
        features = self.backbone(x)

        # Neck
        fpn_feats = self.neck(features)

        # Head
        outputs = self.bbox_head(fpn_feats)

        # Flatten outputs for ONNX export
        all_cls, all_reg, all_kps = [], [], []
        for cls, reg, kps in outputs:
            all_cls.append(cls)
            all_reg.append(reg)
            all_kps.append(kps)

        return all_cls + all_reg + all_kps  # 9 outputs total


def load_pretrained(model, checkpoint_path):
    """Load pretrained weights from mmdet checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    # Map mmdet keys to our model keys
    new_state_dict = {}
    for k, v in state_dict.items():
        # Handle DepthwiseSeparableConv mapping
        # mmdet: layer1.0.0 (dw), layer1.0.1 (bn), layer1.0.3 (pw), layer1.0.4 (bn)
        # ours: layer1.0.dw.0 (dw), layer1.0.dw.1 (bn), layer1.0.pw.0 (pw), layer1.0.pw.1 (bn)

        new_k = k

        # Map stem
        if 'stem.0.0' in k:
            new_k = k.replace('stem.0.0', 'stem.0.0')  # Conv
        elif 'stem.0.1' in k:
            new_k = k.replace('stem.0.1', 'stem.0.1')  # BN
        elif 'stem.1.0' in k:
            new_k = k.replace('stem.1.0', 'stem.1.dw.0')  # DWConv
        elif 'stem.1.1' in k:
            new_k = k.replace('stem.1.1', 'stem.1.dw.1')  # BN
        elif 'stem.1.3' in k:
            new_k = k.replace('stem.1.3', 'stem.1.pw.0')  # PWConv
        elif 'stem.1.4' in k:
            new_k = k.replace('stem.1.4', 'stem.1.pw.1')  # BN

        # Map layers (layer1, layer2, layer3, layer4)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if f'backbone.{layer_name}' in k:
                # Pattern: layerX.Y.0 -> layerX.Y.dw.0
                #          layerX.Y.1 -> layerX.Y.dw.1
                #          layerX.Y.3 -> layerX.Y.pw.0
                #          layerX.Y.4 -> layerX.Y.pw.1
                import re
                match = re.match(rf'backbone\.{layer_name}\.(\d+)\.(\d+)(.*)', k)
                if match:
                    block_idx, conv_idx, rest = match.groups()
                    conv_idx = int(conv_idx)
                    if conv_idx == 0:
                        new_k = f'backbone.{layer_name}.{block_idx}.dw.0{rest}'
                    elif conv_idx == 1:
                        new_k = f'backbone.{layer_name}.{block_idx}.dw.1{rest}'
                    elif conv_idx == 3:
                        new_k = f'backbone.{layer_name}.{block_idx}.pw.0{rest}'
                    elif conv_idx == 4:
                        new_k = f'backbone.{layer_name}.{block_idx}.pw.1{rest}'

        new_state_dict[new_k] = v

    # Load with strict=False to handle any mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights: {len(new_state_dict)} tensors")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return model


if __name__ == "__main__":
    # Test model
    model = SCRFD()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    x = torch.randn(1, 3, 160, 160)
    outputs = model(x)
    print(f"Outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  [{i}] {out.shape}")

    # Load pretrained
    import os
    if os.path.exists("scrfd_500m_kps.pth"):
        model = load_pretrained(model, "scrfd_500m_kps.pth")
