# tflm_face_embedding

Face Embedding 应用 - 使用 SCRFD + MobileFaceNet 进行人脸检测和特征提取。

## 模型

| 模型 | 输入 | 输出 | 精度 | NPU |
|------|------|------|------|-----|
| SCRFD-500M-KPS | 160x160 RGB | Bbox + 5 landmarks | - | 100% |
| MobileFaceNet | 112x112 RGB | 128D embedding | 99.25% LFW | 99.6% |

## Flash 地址

| 模型 | 地址 | 大小 |
|------|------|------|
| SCRFD | 0x200000 | ~600 KB |
| MobileFaceNet | 0x400000 | ~1.2 MB |

## 调试模式

### DEBUG_DETECTION_ONLY

仅运行人脸检测，跳过 embedding 模型。用于调试检测效果。

**文件:** `cvapp_face_embedding.cpp`

```c
/* 0 = 正常模式 (检测 + embedding) */
/* 1 = 仅检测模式 (跳过 embedding，发送所有检测到的人脸) */
#define DEBUG_DETECTION_ONLY 0
```

### DEBUG_VERBOSE

控制日志详细程度。

```c
/* 0 = 最小日志 (仅 timing 总结) */
/* 1 = 正常日志 (step timing + 关键信息) */
/* 2 = 详细日志 (所有调试信息) */
#define DEBUG_VERBOSE 0
```

### FRAME_CHECK_DEBUG

启用帧检查调试，发送 JPEG 图像到主机。

**文件:** `common_config.h`

```c
/* 1 = 启用 (发送 JPEG 帧) */
/* 0 = 禁用 */
#define FRAME_CHECK_DEBUG 1
```

### DBG_APP_LOG

启用应用层事件日志。

```c
/* 1 = 启用事件日志 */
/* 0 = 禁用 */
#define DBG_APP_LOG 0
```

## 配置参数

**文件:** `common_config.h`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FACE_CONF_THRESHOLD` | 0.70 | 人脸检测置信度阈值 |
| `FACE_NMS_THRESHOLD` | 0.4 | NMS IoU 阈值 |
| `MIN_FACE_SIZE` | 40 | 最小人脸尺寸（像素）|
| `EMBEDDING_OUTPUT_DIM` | 128 | 输出 embedding 维度 |

## 与 tflm_face_recognition 的区别

| 特性 | face_embedding | face_recognition |
|------|----------------|------------------|
| Embedding 模型 | MobileFaceNet | GhostFaceNet |
| 输出维度 | 128D | 512D |
| 内存需求 | 1420 KB | 520 KB |
| 精度 | 99.25% LFW | ~95% LFW |

## 编译

```bash
# 修改 APP_TYPE
# EPII_CM55M_APP_S/makefile: APP_TYPE = tflm_face_embedding

cd EPII_CM55M_APP_S
gmake clean && gmake -j8
```

## 烧录

```bash
./build_and_flash.sh
```

确保更新 `build_and_flash.sh` 中的模型路径：
- SCRFD: `model_zoo/tflm_face_embedding/scrfd_500m_kps_int8_vela.tflite`
- MobileFaceNet: `model_zoo/tflm_face_embedding/mobilefacenet_128d_int8_vela.tflite`
