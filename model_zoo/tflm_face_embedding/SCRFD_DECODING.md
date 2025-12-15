# SCRFD Model Decoding Guide

## Model Overview

SCRFD (Sample and Computation Redistribution for Face Detection) is an efficient anchor-free face detector. The 500M_KPS variant outputs bounding boxes and 5 facial keypoints.

- **Model**: SCRFD_500M_KPS
- **Input**: `[1, 160, 160, 3]` NHWC, normalized to `[0, 1]`
- **Parameters**: ~500K

## Output Structure

The model outputs **9 tensors** across 3 scales (strides 8, 16, 32):

| Stride | Feature Size | Anchors | Scores Shape | Boxes Shape | Keypoints Shape |
|--------|--------------|---------|--------------|-------------|-----------------|
| 8      | 20x20        | 800     | [800, 1]     | [800, 4]    | [800, 10]       |
| 16     | 10x10        | 200     | [200, 1]     | [200, 4]    | [200, 10]       |
| 32     | 5x5          | 50      | [50, 1]      | [50, 4]     | [50, 10]        |

**Note**: Each grid cell has 2 anchors, so `anchors = (input_size / stride)^2 * 2`

## Anchor Generation

Anchors are generated per stride with 2 anchors per grid cell:

```python
def generate_anchors(input_size, stride):
    feat_size = input_size // stride
    anchors = []
    for y in range(feat_size):
        for x in range(feat_size):
            anchors.append((x, y))  # anchor 1
            anchors.append((x, y))  # anchor 2
    return anchors
```

For 160x160 input:
- Stride 8: 20x20x2 = 800 anchors
- Stride 16: 10x10x2 = 200 anchors
- Stride 32: 5x5x2 = 50 anchors

## Decoding

### Reference Points

**Box and Keypoints use the SAME reference point (anchor corner):**

| Output | Reference Point | Formula |
|--------|-----------------|---------|
| Box    | Anchor **corner** | `ax * stride` |
| Keypoints | Anchor **corner** | `ax * stride` |

### Box Decoding

Box output format: `[left, top, right, bottom]` as distances from anchor corner.

```python
# Anchor corner (no offset)
cx = ax * stride
cy = ay * stride

# Box decoding
x1 = cx - box[0] * stride  # left edge
y1 = cy - box[1] * stride  # top edge
x2 = cx + box[2] * stride  # right edge
y2 = cy + box[3] * stride  # bottom edge
```

### Keypoints Decoding

Keypoints output format: `[x0, y0, x1, y1, x2, y2, x3, y3, x4, y4]` (5 points x 2 coords)

```python
# Anchor corner (NO 0.5 offset)
anchor_x = ax * stride
anchor_y = ay * stride

# Keypoint decoding
for k in range(5):
    kp_x = anchor_x + kps[k*2] * stride
    kp_y = anchor_y + kps[k*2+1] * stride
```

### Keypoint Order

The 5 facial keypoints are in this order:
| Index | Landmark |
|-------|----------|
| 0     | Left eye |
| 1     | Right eye |
| 2     | Nose |
| 3     | Left mouth corner |
| 4     | Right mouth corner |

## Complete Decoding Example

```python
def decode_scrfd(outputs, input_size=160, score_threshold=0.5):
    """
    Decode SCRFD outputs to detections.

    Args:
        outputs: List of 9 output tensors
        input_size: Input image size (default 160)
        score_threshold: Detection confidence threshold

    Returns:
        List of detections with 'score', 'box', 'keypoints'
    """
    strides = [8, 16, 32]
    detections = []

    # Group outputs by stride based on anchor count
    anchor_to_stride = {
        800: 8,   # (160/8)^2 * 2
        200: 16,  # (160/16)^2 * 2
        50: 32,   # (160/32)^2 * 2
    }

    grouped = {}
    for out in outputs:
        if len(out.shape) == 3:
            out = out[0]  # Remove batch dim

        count = out.shape[0]
        stride = anchor_to_stride.get(count)
        if stride is None:
            continue

        if stride not in grouped:
            grouped[stride] = {}

        feat_dim = out.shape[-1]
        if feat_dim == 1:
            grouped[stride]['scores'] = out
        elif feat_dim == 4:
            grouped[stride]['boxes'] = out
        elif feat_dim == 10:
            grouped[stride]['keypoints'] = out

    # Decode each stride
    for stride in strides:
        if stride not in grouped:
            continue

        data = grouped[stride]
        scores = data['scores'].flatten()
        boxes = data['boxes']
        kps = data.get('keypoints')

        feat_size = input_size // stride

        # Generate anchors
        anchors = []
        for y in range(feat_size):
            for x in range(feat_size):
                anchors.append((x, y))
                anchors.append((x, y))

        for i, score in enumerate(scores):
            if score < score_threshold:
                continue

            ax, ay = anchors[i]

            # Both box and keypoints use anchor CORNER
            cx = ax * stride
            cy = ay * stride

            box = boxes[i]
            x1 = cx - box[0] * stride
            y1 = cy - box[1] * stride
            x2 = cx + box[2] * stride
            y2 = cy + box[3] * stride

            det = {
                'score': float(score),
                'box': [x1, y1, x2, y2],
                'keypoints': None
            }

            # Keypoints: also use anchor CORNER
            if kps is not None:
                kp = kps[i]
                landmarks = []
                for k in range(5):
                    lx = ax * stride + kp[k*2] * stride
                    ly = ay * stride + kp[k*2+1] * stride
                    landmarks.append([lx, ly])
                det['keypoints'] = landmarks

            detections.append(det)

    return detections
```

## Int8 Quantization Notes

### Input Quantization
- Scale: `0.003921569` (≈ 1/255)
- Zero Point: `-128`
- Conversion: `int8_value = float_value / scale + zero_point`

### Output Quantization
Each output tensor has its own scale and zero point. Dequantize before decoding:

```python
float_value = (int8_value - zero_point) * scale
```

### Quantization Quality

| Output | Correlation | Notes |
|--------|-------------|-------|
| Scores | >0.99 | Excellent |
| Boxes | >0.99 | Excellent |
| Keypoints | >0.99 | Excellent |

The int8 model produces nearly identical results to float32.

## Common Mistakes

1. **Using different references for box and keypoints**: Both should use anchor corner (no +0.5 offset)

2. **Wrong anchor order**: Anchors iterate as `for y: for x: [anchor1, anchor2]`

3. **Forgetting batch dimension**: Some TFLite outputs have shape `[1, N, C]`, need to squeeze

4. **Not scaling for original image**: If input was resized, scale coordinates back:
   ```python
   scale_x = original_width / input_size
   scale_y = original_height / input_size
   ```
