#!/bin/bash
#
# flash_only.sh - Flash firmware and/or models without rebuilding
# Usage:
#   ./flash_only.sh              # Flash firmware + models
#   ./flash_only.sh --model-only # Flash models only (no firmware)
#   ./flash_only.sh --fw-only    # Flash firmware only (no models)
#
# Models:
#   - SCRFD: Face detection (160x160, ~700KB)
#   - MobileFaceNet: Face embedding 128D (112x112, ~400KB)
#

set -e

# Configuration
PROJECT_ROOT="$(pwd)"
OUTPUT_IMG="${PROJECT_ROOT}/we2_image_gen_local/output_case1_sec_wlcsp/output.img"

# Model paths
SCRFD_MODEL="${PROJECT_ROOT}/model_zoo/tflm_face_embedding/scrfd/models/scrfd_500m_kps_int8_vela.tflite"
SCRFD_ADDR="0x200000"
EMBEDDING_MODEL="${PROJECT_ROOT}/model_zoo/tflm_face_embedding/foamliu_mobilefacenet_128d/foamliu_mobilefacenet_128d_qat_int8_vela.tflite"
EMBEDDING_ADDR="0x400000"

# Serial port (auto-detect)
SERIAL_PORT=$(ls /dev/tty.usbmodem* 2>/dev/null | head -1)

# Parse arguments
MODEL_ONLY=0
FW_ONLY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-only)
            MODEL_ONLY=1
            shift
            ;;
        --fw-only)
            FW_ONLY=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "  Grove Vision AI Module V2 Flash Tool"
echo "========================================"
echo ""
echo "Models: SCRFD + MobileFaceNet 128D"
echo ""

# Check serial port
if [ -z "${SERIAL_PORT}" ]; then
    echo "✗ No USB modem device found. Please connect the device."
    exit 1
fi
echo "Using port: ${SERIAL_PORT}"
echo ""

cd "${PROJECT_ROOT}"

# Use uv environment for xmodem
XMODEM_DIR="${PROJECT_ROOT}/xmodem"
PYTHON_CMD="uv run --directory ${XMODEM_DIR} python"

if [ ${MODEL_ONLY} -eq 1 ]; then
    # Flash models only
    echo "Flashing models only..."
    echo "  - SCRFD: ${SCRFD_ADDR}"
    echo "  - MobileFaceNet: ${EMBEDDING_ADDR}"
    echo ""
    ${PYTHON_CMD} ${XMODEM_DIR}/xmodem_send.py \
        --port="${SERIAL_PORT}" \
        --baudrate=921600 \
        --protocol=xmodem \
        --model="${SCRFD_MODEL} ${SCRFD_ADDR} 0x00000" \
        --model="${EMBEDDING_MODEL} ${EMBEDDING_ADDR} 0x00000"
elif [ ${FW_ONLY} -eq 1 ]; then
    # Flash firmware only
    if [ ! -f "${OUTPUT_IMG}" ]; then
        echo "✗ Firmware image not found: ${OUTPUT_IMG}"
        echo "  Run ./build_and_flash.sh first to build firmware."
        exit 1
    fi
    echo "Flashing firmware only..."
    echo "  - Image: ${OUTPUT_IMG}"
    echo ""
    ${PYTHON_CMD} ${XMODEM_DIR}/xmodem_send.py \
        --port="${SERIAL_PORT}" \
        --baudrate=921600 \
        --protocol=xmodem \
        --file="${OUTPUT_IMG}"
else
    # Flash firmware + models
    if [ ! -f "${OUTPUT_IMG}" ]; then
        echo "✗ Firmware image not found: ${OUTPUT_IMG}"
        echo "  Run ./build_and_flash.sh first to build firmware."
        exit 1
    fi
    echo "Flashing firmware + models..."
    echo "  - Image: ${OUTPUT_IMG}"
    echo "  - SCRFD: ${SCRFD_ADDR}"
    echo "  - MobileFaceNet: ${EMBEDDING_ADDR}"
    echo ""
    ${PYTHON_CMD} ${XMODEM_DIR}/xmodem_send.py \
        --port="${SERIAL_PORT}" \
        --baudrate=921600 \
        --protocol=xmodem \
        --file="${OUTPUT_IMG}" \
        --model="${SCRFD_MODEL} ${SCRFD_ADDR} 0x00000" \
        --model="${EMBEDDING_MODEL} ${EMBEDDING_ADDR} 0x00000"
fi

echo ""
echo "========================================"
echo "  Flash complete!"
echo "  Press RESET button on the device"
echo "========================================"
