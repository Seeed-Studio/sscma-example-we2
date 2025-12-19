# SCRFD 模型量化指南

本文档介绍如何将 SCRFD 人脸检测模型从 Float32 量化为 INT8，并编译为 Ethos-U55 NPU 可执行格式。

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [量化方法对比](#量化方法对比)
- [PTQ 量化流程](#ptq-量化流程)
- [QAT 量化流程](#qat-量化流程)
- [模型验证](#模型验证)
- [文件说明](#文件说明)

## 概述

SCRFD (Sample and Computation Redistribution for Face Detection) 是一个高效的人脸检测模型。本项目针对 Grove Vision AI Module V2 (Ethos-U55 NPU) 进行量化优化。

### 模型规格

| 属性 | 值 |
|------|-----|
| 模型 | SCRFD-500M-KPS |
| 输入尺寸 | 160×160×3 (RGB) |
| 输出 | 9 个张量 (3 个尺度 × 3 种输出) |
| 参数量 | ~500K |
| 原始格式 | ONNX (Float32) |
| 目标格式 | TFLite INT8 + Vela |

### 输出结构

SCRFD 在 3 个不同尺度 (stride 8/16/32) 输出检测结果：

| 尺度 | Anchors | Scores | Boxes | Keypoints |
|------|---------|--------|-------|-----------|
| Stride 8 | 800 | [1,800,1] | [1,800,4] | [1,800,10] |
| Stride 16 | 200 | [1,200,1] | [1,200,4] | [1,200,10] |
| Stride 32 | 50 | [1,50,1] | [1,50,4] | [1,50,10] |

## 环境准备

### 依赖安装

```bash
cd model_zoo/tflm_face_embedding
uv sync  # 使用 uv 包管理器
```

### 主要依赖

- Python 3.10+
- PyTorch 2.x
- TensorFlow 2.x
- onnx2tf
- ethos-u-vela
- onnxruntime
- opencv-python

### 校准数据准备

使用 LFW (Labeled Faces in the Wild) 数据集作为校准数据：

```bash
python prepare_calibration_data.py
```

数据将下载到 `calibration_data/lfw/lfw-deepfunneled/` 目录。

## 量化方法对比

| 方法 | 精度 (相关系数) | 复杂度 | 推荐场景 |
|------|----------------|--------|----------|
| **PTQ (推荐)** | Scores: 0.9946, Boxes: 0.9987 | 低 | 生产环境 |
| QAT | Scores: 0.9489, Boxes: 0.9962 | 高 | 研究实验 |

**结论**: 对于 SCRFD，增强版 PTQ（使用 5000 张校准图片）已能达到优秀的精度，QAT 并未带来显著提升。

## PTQ 量化流程

PTQ (Post-Training Quantization) 是推荐的量化方法。

### 步骤 1: 准备 ONNX 模型

确保 `scrfd_500m_kps.onnx` 文件存在于当前目录。

### 步骤 2: 修复输入尺寸

原始 ONNX 模型输入为 640×640，需要修改为 160×160：

```bash
python convert_scrfd.py --fix-input-size
```

### 步骤 3: ONNX 转 TFLite INT8

```bash
python convert_scrfd.py --quantize --max-calib 5000
```

关键参数：
- `--max-calib`: 校准图片数量，推荐 5000 张
- 输入归一化: `[0, 1]` 范围
- 输出: 全 INT8 量化

### 步骤 4: Vela 编译

```bash
vela --accelerator-config ethos-u55-64 \
     --optimise Performance \
     scrfd_500m_kps_int8.tflite
```

输出文件: `scrfd_500m_kps_int8_vela.tflite`

### 完整 PTQ 脚本

```bash
# 一键执行 PTQ 量化
python convert_scrfd.py \
    --onnx scrfd_500m_kps.onnx \
    --output scrfd_500m_kps_int8.tflite \
    --max-calib 5000
```

## QAT 量化流程

QAT (Quantization-Aware Training) 通过训练让模型适应量化误差。

### 原理

1. **构建 PyTorch 模型**: 从 mmdetection checkpoint 加载权重
2. **Conv+BN+ReLU 融合**: 训练前必须融合这些层
3. **应用 QAT**: 插入 fake quantization 节点
4. **输出蒸馏训练**: 让 QAT 模型输出匹配原始 Float32 模型
5. **导出**: QAT → ONNX → TFLite INT8 → Vela

### 步骤 1: 准备预训练权重

下载 `scrfd_500m_kps.pth`:

```bash
# 从 InsightFace 官方仓库下载
wget https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.zip
unzip scrfd_500m_bnkps.zip
```

### 步骤 2: Conv+BN+ReLU 融合

融合是 QAT 的关键步骤，必须在训练前完成：

```python
from torch.ao.quantization import fuse_modules

def fuse_model(model):
    model.eval()  # 必须在 eval 模式下融合

    # 融合 stem 层
    fuse_modules(model.backbone.stem[0], ['0', '1', '2'], inplace=True)

    # 融合 DepthwiseSeparableConv
    for block in get_all_dwsep_blocks(model):
        fuse_modules(block.depthwise_conv, ['conv', 'bn', 'relu'], inplace=True)
        fuse_modules(block.pointwise_conv, ['conv', 'bn', 'relu'], inplace=True)

    return model
```

### 步骤 3: 应用 QAT

```python
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat

# 设置量化后端 (ARM 设备使用 qnnpack)
torch.backends.quantized.engine = 'qnnpack'

# 配置 QAT
model.qconfig = get_default_qat_qconfig('qnnpack')

# 准备 QAT 训练
model_qat = prepare_qat(model, inplace=False)
```

### 步骤 4: 输出蒸馏训练

```python
# Teacher: 原始 Float32 ONNX 模型
# Student: QAT 模型
# Loss: MSE(teacher_output, student_output)

for epoch in range(epochs):
    for batch in dataloader:
        teacher_out = onnx_inference(batch)
        student_out = model_qat(batch)

        loss = sum(F.mse_loss(t, s) for t, s in zip(teacher_out, student_out))
        loss.backward()
        optimizer.step()
```

### 步骤 5: 导出和编译

```python
# 保存 QAT checkpoint
torch.save(model_qat.state_dict(), 'scrfd_qat_checkpoint.pth')

# 导出 ONNX
torch.onnx.export(model_export, dummy_input, 'scrfd_qat.onnx')

# 转换 TFLite INT8
onnx2tf.convert(...)

# Vela 编译
vela --accelerator-config ethos-u55-64 scrfd_qat_int8.tflite
```

### 完整 QAT 脚本

```bash
# 一键执行 QAT 量化
python qat_scrfd_native.py \
    --pth scrfd_500m_kps.pth \
    --onnx-ref scrfd_500m_kps.onnx \
    --output scrfd_qat_int8.tflite \
    --epochs 5 \
    --batch-size 16 \
    --max-calib 3000 \
    --lr 5e-5
```

## 模型验证

### 验证脚本

```bash
python validate_quantization.py \
    --onnx scrfd_500m_kps.onnx \
    --tflite scrfd_500m_kps_int8.tflite \
    --num-samples 50
```

### 验证指标

1. **相关系数 (Correlation)**: 衡量输出趋势一致性，越接近 1 越好
2. **MAE (Mean Absolute Error)**: 平均绝对误差，越小越好
3. **MSE (Mean Squared Error)**: 均方误差，越小越好

### 验证结果示例

```
============================================================
Quantization Accuracy Report
============================================================

SCORES:
  Correlation: 0.9946
  MAE: 0.0312

BOXES:
  Correlation: 0.9987
  MAE: 0.0198

KEYPOINTS:
  Correlation: 0.9986
  MAE: 0.0215
```

## 文件说明

### 脚本文件

| 文件 | 说明 |
|------|------|
| `convert_scrfd.py` | PTQ 量化主脚本 |
| `qat_scrfd_native.py` | QAT 量化主脚本 |
| `scrfd_model.py` | SCRFD PyTorch 模型定义 |
| `validate_quantization.py` | 模型验证脚本 |
| `prepare_calibration_data.py` | 校准数据准备脚本 |

### 模型文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `scrfd_500m_kps.onnx` | ONNX Float32 | 原始模型 (640×640) |
| `scrfd_500m_kps.pth` | PyTorch | 预训练权重 |
| `scrfd_500m_kps_int8.tflite` | TFLite INT8 | PTQ 量化模型 |
| `scrfd_500m_kps_int8_vela.tflite` | TFLite Vela | PTQ + NPU 优化 |
| `scrfd_qat_int8.tflite` | TFLite INT8 | QAT 量化模型 |
| `scrfd_qat_int8_vela.tflite` | TFLite Vela | QAT + NPU 优化 |

### 数据文件

| 文件 | 说明 |
|------|------|
| `calibration_data/` | LFW 校准数据集 |
| `scrfd_calib_nhwc.npy` | 预处理后的校准数据 |

## 常见问题

### Q1: 为什么 QAT 精度不如 PTQ？

对于 SCRFD 这类已经很好优化的模型，增强版 PTQ（使用足够多的校准数据）已经能达到很高的精度。QAT 的优势主要体现在：
- 精度损失较大的模型
- 需要极低比特量化（如 INT4）的场景

### Q2: 如何选择校准数据数量？

- **最小**: 500 张 (基本可用)
- **推荐**: 2000-5000 张 (最佳性价比)
- **更多**: >5000 张 (边际收益递减)

### Q3: Vela 编译失败怎么办？

1. 确保 TFLite 模型是全 INT8 量化
2. 检查是否有不支持的算子
3. 尝试添加 `--system-config Ethos_U55_High_End_Embedded`

### Q4: 如何在设备上部署？

1. 将 `*_vela.tflite` 模型烧录到 Flash
2. 在固件中设置正确的模型地址
3. 确保输入预处理与量化时一致

### Q5: QAT 模型在设备上检测不到人脸？

**问题现象**: QAT 模型在 PC 验证时精度正常，但烧录到设备后检测不到人脸。

**根本原因**: PyTorch 导出的 ONNX 输出格式是 3D (`[1, N, C]`)，而 PTQ 转换后的 TFLite 输出是 2D (`[N, C]`)。固件期望 2D 格式。

**解决方案**: 在 ONNX 模型末尾添加 Squeeze 节点去掉 batch 维度：

```python
import onnx
from onnx import helper, numpy_helper

model = onnx.load("scrfd_qat.onnx")

# 为每个输出添加 Squeeze 节点
for i, out in enumerate(model.graph.output):
    old_name = out.name
    new_name = f"output_{i}_squeezed"
    axes_name = f"squeeze_axes_{i}"

    # 创建 axes 常量 (opset 13+)
    axes_tensor = numpy_helper.from_array(
        np.array([0], dtype=np.int64), axes_name
    )
    model.graph.initializer.append(axes_tensor)

    # 创建 Squeeze 节点
    squeeze_node = helper.make_node(
        'Squeeze',
        inputs=[old_name, axes_name],
        outputs=[new_name],
    )
    model.graph.node.append(squeeze_node)

# 更新输出定义...
onnx.save(model, "scrfd_qat_fixed.onnx")
```

**已修复的模型**: `scrfd_qat_int8_fixed_vela.tflite`

### Q6: PTQ 和 QAT 输出范围不一致？

**现象**:
- PTQ scores: min=0.00, max=0.92 (全正)
- QAT scores: min=-1.14, max=1.53 (有正有负)

**原因**: 这是正常的。SCRFD 的 scores 输出是 logits（未经 sigmoid），需要在后处理中应用 sigmoid。PTQ 可能由于量化参数的选择导致负值被截断。

**影响**: 只要后处理正确应用 sigmoid，两种模型都能正常工作。

### Q7: 校准数据尺寸必须匹配模型输入

**错误信息**:
```
RuntimeError: Given shapes, [1,13,13,16] and [1,10,10,16], are not broadcastable.
```

**原因**: 校准数据尺寸与模型输入不匹配（如 200×200 vs 160×160）。

**解决**: 确保 `sample_ms1m_images()` 的 `input_size` 参数与模型一致：
```python
calib_data, _ = sample_ms1m_images(ms1m_dir, num_images=200, input_size=160)
```

### Q8: QAT 训练带 Sigmoid 导致设备检测失败

**现象**: 训练时加了 sigmoid，设备上检测不到人脸。

**解决**: 从 checkpoint 重新导出（去掉 sigmoid）：
```bash
python export_from_checkpoint.py --checkpoint scrfd_qat_v4_best_checkpoint.pth --output scrfd_qat_v4_nosigmoid
```

模型输出为 logits（可 >1），设备端直接与阈值比较即可。

### Q9: 训练归一化 [-1,1] vs 设备 [0,1]

**问题**: 训练用 `[-1, 1]`，设备发送 `[0, 1]`。

**解决**: 在 `convert_to_tflite()` 中转换校准数据：
```python
calib_01 = (calib_data * 0.5 + 0.5).astype(np.float32)  # [-1,1] → [0,1]
```

### Q10: 前端显示置信度超过 100%

**现象**: 检测框显示 `115%` 等超过 100% 的值。

**原因**: 模型输出 logits（可 >1），前端直接 `× 100` 显示。

**解决**: 前端 clamp 到 [0, 1]：
```javascript
const conf = Math.min(Math.max(face.confidence, 0), 1.0);
ctx.fillText(`${(conf * 100).toFixed(0)}%`, ...);
```

## 参考资料

- [SCRFD 论文](https://arxiv.org/abs/2105.04714)
- [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [Ethos-U Vela](https://pypi.org/project/ethos-u-vela/)
- [TensorFlow Lite 量化](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [PyTorch QAT](https://pytorch.org/docs/stable/quantization.html)
