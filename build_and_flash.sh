#!/bin/bash
#
# build_and_flash.sh - One-click build, generate image, and flash script
# Usage: ./build_and_flash.sh [--no-clean] [--no-flash] [--no-model]
#
# Options:
#   --no-clean              Skip 'make clean'
#   --no-flash              Build only, don't flash
#   --no-model              Flash firmware without models
#
# Models:
#   - SCRFD: Face detection (160x160, ~700KB)
#   - MobileFaceNet: Face embedding 128D (112x112, ~400KB)
#

set -e

# Configuration
PROJECT_ROOT="/Users/harvest/project/Seeed_Grove_Vision_AI_Module_V2"
APP_DIR="${PROJECT_ROOT}/EPII_CM55M_APP_S"
IMAGE_GEN_DIR="${PROJECT_ROOT}/we2_image_gen_local"
OUTPUT_DIR="${IMAGE_GEN_DIR}/output_case1_sec_wlcsp"
ELF_FILE="${APP_DIR}/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"

# App configuration
MAKEFILE_APP_TYPE="tflm_face_embedding"

# Model paths
SCRFD_MODEL="${PROJECT_ROOT}/model_zoo/tflm_face_embedding/scrfd/models/scrfd_500m_kps_int8_vela.tflite"
SCRFD_ADDR="0x200000"
EMBEDDING_MODEL="${PROJECT_ROOT}/model_zoo/tflm_face_embedding/foamliu_mobilefacenet_128d/foamliu_mobilefacenet_128d_qat_int8_vela.tflite"
EMBEDDING_ADDR="0x400000"

# Serial port (auto-detect)
SERIAL_PORT=$(ls /dev/tty.usbmodem* 2>/dev/null | head -1)

# Parse arguments
NO_CLEAN=0
NO_FLASH=0
NO_MODEL=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-clean)
            NO_CLEAN=1
            shift
            ;;
        --no-flash)
            NO_FLASH=1
            shift
            ;;
        --no-model)
            NO_MODEL=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "  Grove Vision AI Module V2 Build Tool"
echo "========================================"
echo ""
echo "App: ${MAKEFILE_APP_TYPE}"
echo "Models: SCRFD + MobileFaceNet 128D"
echo ""

# Step 1: Build firmware
echo "[1/3] Building firmware..."
echo "----------------------------------------"
cd "${APP_DIR}"

# Update APP_TYPE in makefile
sed -i.bak "s/^APP_TYPE = .*/APP_TYPE = ${MAKEFILE_APP_TYPE}/" makefile && rm -f makefile.bak
echo "Set APP_TYPE = ${MAKEFILE_APP_TYPE} in makefile"
echo ""
if [ ${NO_CLEAN} -eq 0 ]; then
    gmake clean
    echo ""
else
    echo "(Skipping clean - incremental build)"
    echo ""
fi
echo "Compiling (parallel, $(sysctl -n hw.ncpu) cores)..."
echo ""
gmake -j$(sysctl -n hw.ncpu)

# Verify ELF file
if [ ! -f "${ELF_FILE}" ]; then
    echo ""
    echo "✗ ELF file not found: ${ELF_FILE}"
    exit 1
fi
ELF_SIZE=$(ls -la "${ELF_FILE}" | awk '{print $5}')
echo ""
echo "✓ Build successful (ELF size: ${ELF_SIZE} bytes)"
echo ""

# Step 2: Generate firmware image
echo "========================================"
echo "[2/3] Generating firmware image..."
echo "----------------------------------------"
cd "${IMAGE_GEN_DIR}"
cp "${ELF_FILE}" input_case1_secboot/
./we2_local_image_gen_macOS_arm64 project_case1_blp_wlcsp.json

# Verify output image
OUTPUT_IMG="${OUTPUT_DIR}/output.img"
if [ ! -f "${OUTPUT_IMG}" ]; then
    echo ""
    echo "✗ Output image not found: ${OUTPUT_IMG}"
    exit 1
fi
IMG_SIZE=$(ls -la "${OUTPUT_IMG}" | awk '{print $5}')
echo ""
echo "✓ Image generation successful (size: ${IMG_SIZE} bytes)"
echo ""

# Step 3: Flash firmware
if [ ${NO_FLASH} -eq 1 ]; then
    echo "========================================"
    echo "[3/3] Skipping flash (--no-flash specified)"
    echo ""
    echo "To flash manually:"
    echo "  python3 xmodem/xmodem_send.py --port=${SERIAL_PORT} --baudrate=921600 --protocol=xmodem --file=${OUTPUT_IMG}"
    exit 0
fi

echo "========================================"
echo "[3/3] Flashing firmware..."
echo "----------------------------------------"

# Check serial port
if [ -z "${SERIAL_PORT}" ]; then
    echo "✗ No USB modem device found. Please connect the device."
    echo ""
    echo "After connecting, run:"
    echo "  python3 xmodem/xmodem_send.py --port=/dev/tty.usbmodemXXXX --baudrate=921600 --protocol=xmodem --file=${OUTPUT_IMG}"
    exit 1
fi
echo "Using port: ${SERIAL_PORT}"
echo ""

cd "${PROJECT_ROOT}"

# Use uv environment for xmodem
XMODEM_DIR="${PROJECT_ROOT}/xmodem"
PYTHON_CMD="uv run --directory ${XMODEM_DIR} python"

# Build flash command
if [ ${NO_MODEL} -eq 0 ]; then
    # Flash with models
    if [ -f "${SCRFD_MODEL}" ] && [ -f "${EMBEDDING_MODEL}" ]; then
        echo "Flashing firmware + SCRFD + MobileFaceNet models..."
        echo ""
        ${PYTHON_CMD} ${XMODEM_DIR}/xmodem_send.py \
            --port="${SERIAL_PORT}" \
            --baudrate=921600 \
            --protocol=xmodem \
            --file="${OUTPUT_IMG}" \
            --model="${SCRFD_MODEL} ${SCRFD_ADDR} 0x00000" \
            --model="${EMBEDDING_MODEL} ${EMBEDDING_ADDR} 0x00000"
    else
        echo "Warning: Model files not found, flashing firmware only..."
        echo "  Expected SCRFD: ${SCRFD_MODEL}"
        echo "  Expected Embedding: ${EMBEDDING_MODEL}"
        echo ""
        ${PYTHON_CMD} ${XMODEM_DIR}/xmodem_send.py \
            --port="${SERIAL_PORT}" \
            --baudrate=921600 \
            --protocol=xmodem \
            --file="${OUTPUT_IMG}"
    fi
else
    echo "Flashing firmware only (--no-model specified)..."
    echo ""
    ${PYTHON_CMD} ${XMODEM_DIR}/xmodem_send.py \
        --port="${SERIAL_PORT}" \
        --baudrate=921600 \
        --protocol=xmodem \
        --file="${OUTPUT_IMG}"
fi

echo ""
echo "========================================"
echo "  Flash complete!"
echo "  Press RESET button on the device"
echo "========================================"
