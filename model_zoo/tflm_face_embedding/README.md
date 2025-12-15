# Face Embedding Models for Grove Vision AI V2

Face detection (SCRFD) + Face embedding (GhostFaceNet/MobileFaceNet) for Ethos-U55 NPU.

## Final Models (Ready to Flash)

| Model | File | Size | Flash Address |
|-------|------|------|---------------|
| SCRFD-500M-KPS | `scrfd/models/scrfd_500m_kps_int8_vela.tflite` | 701 KB | 0x200000 |
| GhostFaceNet-0.5 | `ghostfacenet/models/ghostfacenet_fixed_int8_vela.tflite` | 849 KB | 0x400000 |

## Model Specifications

### SCRFD-500M-KPS (Face Detection)
- Input: [1, 160, 160, 3] INT8
- Output: Multi-scale detection (bounding boxes + 5-point landmarks)
- SRAM: ~201 KB
- Inference: ~4.4ms @ 500MHz

### GhostFaceNet-0.5 (Face Embedding)
- Input: [1, 112, 112, 3] INT8
- Output: [1, 512] INT8 (512-dimensional embedding)
- SRAM: ~245 KB
- Inference: ~17ms @ 500MHz

## Directory Structure

```
tflm_face_embedding/
├── scrfd/                           # SCRFD Face Detection
│   ├── models/                      # Model files
│   │   ├── scrfd_500m_kps_int8_vela.tflite  # Final model (flash to 0x200000)
│   │   ├── scrfd_500m_kps_int8.tflite       # INT8 quantized
│   │   ├── scrfd_500m_kps.onnx              # Original ONNX
│   │   └── scrfd_500m_kps.pth               # PyTorch weights
│   ├── scripts/                     # Conversion scripts
│   │   ├── convert_scrfd.py         # PTQ conversion
│   │   ├── convert_scrfd_enhanced.py # Enhanced conversion
│   │   └── scrfd_model.py           # Model definition
│   └── quantization/                # QAT training
│       ├── qat_scrfd_enhanced.py    # Main QAT training script
│       ├── run_qat_enhanced.sh      # Training runner
│       ├── export_from_checkpoint.py # Export trained model
│       ├── validate_quantization.py # Validation script
│       └── README.md                # QAT documentation
│
├── ghostfacenet/                    # GhostFaceNet Face Embedding
│   ├── models/                      # Model files
│   │   ├── ghostfacenet_fixed_int8_vela.tflite  # Final model (flash to 0x400000)
│   │   ├── ghostfacenet_fixed_int8.tflite       # INT8 quantized
│   │   ├── ghostfacenet_float32.onnx            # Float ONNX
│   │   └── GN_W0.5_S2_ArcFace_epoch16.h5        # Original H5
│   └── scripts/                     # Conversion scripts
│       ├── convert_ghostfacenet.py  # Main conversion script
│       ├── qat_ghostfacenet.py      # QAT training
│       └── fix_ghostfacenet_overlap.py  # Memory overlap fix
│
├── calibration_data/                # Calibration data for INT8 quantization
│   ├── fd_160/                      # Face detection calibration (160x160)
│   ├── emb_112/                     # Embedding calibration (112x112)
│   └── lfw/                         # LFW dataset for validation
│
├── foamliu_mobilefacenet_128d/      # MobileFaceNet alternative
│
├── prepare_calibration_data.py      # Generate calibration images
├── download_datasets.py             # Download training datasets
├── analyze_*.py                     # Analysis scripts
├── SCRFD_DECODING.md               # SCRFD output format documentation
└── pyproject.toml                   # Python dependencies
```

## Quick Start

```bash
# Setup environment
cd model_zoo/tflm_face_embedding
uv sync

# Convert GhostFaceNet (PTQ)
cd ghostfacenet/scripts
uv run python convert_ghostfacenet.py

# Convert SCRFD (PTQ)
cd ../../scrfd/scripts
uv run python convert_scrfd.py

# SCRFD QAT Training (for better accuracy)
cd ../quantization
./run_qat_enhanced.sh
```

## SCRFD QAT Training

For improved SCRFD detection accuracy, use Quantization-Aware Training:

```bash
cd scrfd/quantization

# Quick test (10 min)
NUM_IMAGES=5000 EPOCHS=5 ./run_qat_enhanced.sh

# Standard training (45 min, recommended)
NUM_IMAGES=30000 EPOCHS=10 ./run_qat_enhanced.sh

# Full training (2 hours)
NUM_IMAGES=50000 EPOCHS=15 ./run_qat_enhanced.sh
```

See `scrfd/quantization/README.md` for detailed QAT documentation.

## Flashing Models

```bash
# Flash with firmware
./build_and_flash.sh

# Or manually:
python3 xmodem/xmodem_send.py \
  --port=/dev/tty.usbmodem* \
  --baudrate=921600 \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="model_zoo/tflm_face_embedding/scrfd/models/scrfd_500m_kps_int8_vela.tflite 0x200000 0x0" \
  --model="model_zoo/tflm_face_embedding/ghostfacenet/models/ghostfacenet_fixed_int8_vela.tflite 0x400000 0x0"
```

## Troubleshooting

See `scrfd/quantization/README.md` for common issues:
- Q1: Calibration data format requirements
- Q2: ONNX opset version compatibility
- Q3: Input value range normalization
- Q4-Q10: Various conversion and quantization issues
