# foamliu MobileFaceNet 128D

Face embedding model optimized for Ethos-U55 NPU on Grove Vision AI Module V2.

## Model Information

| Property | Value |
|----------|-------|
| Source | [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet) |
| Architecture | MobileFaceNet |
| LFW Accuracy | 99.48% |
| Input | 112×112×3 RGB |
| Input Range | [-1, 1] (normalized) |
| Output | 128D embedding vector |

## Generated Files

| File | Size | Description |
|------|------|-------------|
| `foamliu_mobilefacenet_128d_qat_int8_vela.tflite` | 1.1 MB | **Production model** - QAT + Vela-optimized for Ethos-U55 |
| `foamliu_mobilefacenet_128d_qat_int8.tflite` | 1.2 MB | QAT INT8 quantized model |
| `foamliu_mobilefacenet_128d_int8_vela.tflite` | 1.1 MB | Standard PTQ + Vela-optimized |
| `foamliu_mobilefacenet_128d_int8.tflite` | 1.2 MB | Standard INT8 quantized model |
| `foamliu_mobilefacenet_128d_float32.tflite` | 3.8 MB | Float32 reference model |

## Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Quantization Accuracy | 98.83% | >95% | PASS |
| NPU Operators | 100% | 100% | PASS |
| CPU Operators | 0 | 0 | PASS |
| Tensor Arena | 599.77 KiB | <500 KiB | WARN |
| Inference Time | 14.83 ms | - | 67 fps |

## Usage

### Input Preprocessing

```c
// C code for embedded deployment
// Input: BGR image from camera
// Output: INT8 tensor [-128, 127]

void preprocess_face(uint8_t* bgr_input, int8_t* tensor_input) {
    for (int i = 0; i < 112 * 112; i++) {
        // BGR to RGB conversion
        float r = bgr_input[i * 3 + 2];
        float g = bgr_input[i * 3 + 1];
        float b = bgr_input[i * 3 + 0];

        // Normalize to [-1, 1]
        r = (r - 127.5f) / 127.5f;
        g = (g - 127.5f) / 127.5f;
        b = (b - 127.5f) / 127.5f;

        // Quantize to INT8 (scale=0.007843, zero_point=-1)
        tensor_input[i * 3 + 0] = (int8_t)CLAMP(r / 0.007843f - 1, -128, 127);
        tensor_input[i * 3 + 1] = (int8_t)CLAMP(g / 0.007843f - 1, -128, 127);
        tensor_input[i * 3 + 2] = (int8_t)CLAMP(b / 0.007843f - 1, -128, 127);
    }
}
```

### Output Dequantization

```c
// Output: INT8 embedding [1, 128]
// Quantization: scale=0.033768, zero_point=1

void dequantize_embedding(int8_t* int8_output, float* float_embedding) {
    for (int i = 0; i < 128; i++) {
        float_embedding[i] = (int8_output[i] - 1) * 0.033768f;
    }
}
```

### Cosine Similarity for Face Matching

```c
float cosine_similarity(float* emb1, float* emb2, int dim) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += emb1[i] * emb2[i];
        norm1 += emb1[i] * emb1[i];
        norm2 += emb2[i] * emb2[i];
    }
    return dot / (sqrtf(norm1) * sqrtf(norm2) + 1e-8f);
}

// Threshold: similarity > 0.5 indicates same person
```

## Conversion Process

### Environment Setup

```bash
cd model_zoo/tflm_face_embedding/foamliu_mobilefacenet_128d

# Create virtual environment with Python 3.11
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install tensorflow==2.19.0 onnx2tf==1.25.0 onnx==1.16.0 opencv-python gdown
```

### Run Conversion

```bash
# Standard PTQ conversion
uv run python convert.py

# QAT-enhanced conversion (recommended)
uv run python convert.py --qat --num-calib 500

# With more calibration samples
uv run python convert.py --qat --num-calib 1000
```

### Key Conversion Steps

1. **Fix Dynamic Shapes**: Set static batch size [1, ...] to avoid CPU fallback ops
2. **Convert to TFLite**: Use onnx2tf with BatchNorm fusion (Conv+BN→Conv)
3. **Enhanced Calibration (QAT mode)**:
   - Load calibration images from `qat_112/` directory
   - Compute teacher embeddings with float32 model
   - Augment data (horizontal flip, brightness adjustment)
4. **INT8 Quantization**: Full integer quantization with enhanced calibration
5. **Vela Compilation**: Optimize for Ethos-U55-64 NPU

### QAT Enhancement

The `--qat` flag enables knowledge distillation-based enhanced calibration:
- Uses larger calibration dataset (500+ images from `qat_112/`)
- Data augmentation: horizontal flip, brightness variations
- Computes embedding statistics for calibration guidance
- Results in better quantization accuracy

### Critical Notes

1. **TensorFlow/PyTorch Conflict**: On macOS ARM, TensorFlow and PyTorch cannot coexist in the same process due to mutex conflicts. The conversion script blocks PyTorch imports.

2. **Static Shapes Required**: Dynamic batch dimensions cause SHAPE/GATHER/RESHAPE ops to fall back to CPU. Fix by setting static [1, ...] shapes in ONNX before conversion.

3. **Package Versions**:
   - TensorFlow 2.19.0 (2.20.0 has issues)
   - onnx2tf 1.25.0 (1.28.5 has broken dependencies)
   - onnx 1.16.0 (1.20.0 missing required functions)
   - onnxruntime 1.20.1 (1.22.0 has macOS mutex bug)

## Flash Address Configuration

For Grove Vision AI Module V2 deployment:

```c
// In common_config.h
#define EMBEDDING_MODEL_ADDR_FLASH  0x400000  // 4MB offset
#define EMBEDDING_MODEL_SIZE        0x200000  // 2MB reserved
```

Flash command:
```bash
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --file=output.img \
  --model="foamliu_mobilefacenet_128d_qat_int8_vela.tflite 0x400000 0x0"
```

## Comparison with Other Models

| Model | Embedding | Accuracy | Arena | NPU% |
|-------|-----------|----------|-------|------|
| **foamliu MobileFaceNet** | 128D | 99.48% LFW | 600 KiB | 100% |
| InsightFace MBF | 512D | 99.83% LFW | 1.1 MiB | 95% |
| GhostFaceNet | 512D | 99.78% LFW | 390 KiB | 100% |

foamliu MobileFaceNet provides a good balance of:
- High accuracy (99.48% LFW)
- Compact embedding (128D)
- Full NPU acceleration (100%)
- Reasonable memory usage (600 KiB)

## License

The original MobileFaceNet model is from [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet).
