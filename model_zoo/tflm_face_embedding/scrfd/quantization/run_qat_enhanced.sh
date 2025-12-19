#!/bin/bash
# SCRFD Enhanced QAT - MS1M-ArcFace Large Scale Training
#
# 推荐配置:
#   Quick (测试):    --num-images 5000  --epochs 5   --batch-size 64   (~10 min)
#   Standard (推荐): --num-images 30000 --epochs 10  --batch-size 64   (~45 min)
#   Full (完整):     --num-images 50000 --epochs 15  --batch-size 128  (~2 hours)

set -e

echo "=============================================="
echo "SCRFD Enhanced QAT - MS1M-ArcFace"
echo "=============================================="

# 检查依赖
if ! command -v uv &> /dev/null; then
    echo "请先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 检查预训练权重
if [ ! -f "scrfd_500m_kps.pth" ]; then
    echo "请先下载预训练权重 scrfd_500m_kps.pth"
    echo "下载地址: https://github.com/deepinsight/insightface/releases"
    exit 1
fi

# 检查 MS1M 数据集
MS1M_DIR="./datasets/ms1m-arcface"
if [ ! -d "$MS1M_DIR" ]; then
    echo "MS1M-ArcFace 数据集未找到: $MS1M_DIR"
    echo "将回退到 LFW 数据集"
fi

# 默认使用 Standard 配置
NUM_IMAGES=${NUM_IMAGES:-30000}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}

echo ""
echo "Training Configuration:"
echo "  Images:     $NUM_IMAGES"
echo "  Epochs:     $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  LR:         $LR"
echo ""

# 执行 Enhanced QAT 训练
echo "[1/2] 执行 Enhanced QAT 训练..."
uv run python qat_scrfd_enhanced.py \
    --pth scrfd_500m_kps.pth \
    --onnx-ref scrfd_500m_kps.onnx \
    --output scrfd_qat_enhanced.tflite \
    --ms1m-dir "$MS1M_DIR" \
    --num-images $NUM_IMAGES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --warmup-epochs 1 \
    --val-split 0.1

# 验证模型
echo ""
echo "[2/2] 验证量化精度..."
if [ -f "validate_quantization.py" ]; then
    uv run python validate_quantization.py \
        --onnx scrfd_qat_enhanced.onnx \
        --tflite scrfd_qat_enhanced.tflite \
        --num-samples 50
fi

echo ""
echo "=============================================="
echo "Enhanced QAT 训练完成!"
echo "输出文件: scrfd_qat_enhanced_vela.tflite"
echo "=============================================="
