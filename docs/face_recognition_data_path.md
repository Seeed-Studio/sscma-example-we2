# Face Recognition 数据路径文档

本文档详细描述了从视频采集到前端渲染的完整数据路径，用于调试 bbox/landmark 与视频不对齐问题。

## 目录
1. [整体数据流图](#1-整体数据流图)
2. [摄像头配置与采集](#2-摄像头配置与采集)
3. [SCRFD 人脸检测模型](#3-scrfd-人脸检测模型)
4. [后处理与坐标转换](#4-后处理与坐标转换)
5. [数据传输格式](#5-数据传输格式)
6. [前端解析与渲染](#6-前端解析与渲染)
7. [关键坐标系统](#7-关键坐标系统)
8. [潜在问题点](#8-潜在问题点)

---

## 1. 整体数据流图

```
传感器 (OV5647 640x480)
    ↓
硬件图像处理 (INP 4TO2 下采样)
    ↓
原始图像 (320x240 BGR planar) ← JPEG编码
    ↓
预处理 (缩放 + BGR→RGB)
    ↓
SCRFD 模型输入 (160x160 RGB)
    ↓
SCRFD 推理 (NPU)
    ↓
后处理 (解码 + 坐标映射)  ← 坐标从 160x160 映射到 320x240
    ↓
JSON 打包 (bbox, landmarks, embedding, JPEG)
    ↓
串口传输 (921600 baud)
    ↓
Python 后端解析
    ↓
WebSocket 传输
    ↓
前端渲染 (640x480 Canvas)  ← 坐标从 320x240 缩放到 Canvas
```

---

## 2. 摄像头配置与采集

### 配置文件
**路径**: `EPII_CM55M_APP_S/app/scenario_app/tflm_face_recognition/cis_sensor/cis_ov5647/cisdp_cfg.h`

### 关键配置参数

```c
// 传感器原始分辨率 (binning模式)
#define OV5647_SENSOR_WIDTH         640
#define OV5647_SENSOR_HEIGHT        480

// INP 下采样配置
#define DP_INP_CASE                 2
#define DP_INP_SUBSAMPLE            INP_SUBSAMPLE_4TO2  // 4:2 下采样

// INP 输出分辨率 (下采样后)
#define DP_INP_OUT_WIDTH            320   // 640 / 2
#define DP_INP_OUT_HEIGHT           240   // 480 / 2

// HW5x5 输出分辨率
#define DP_HW5X5_OUT_WIDTH          320
#define DP_HW5X5_OUT_HEIGHT         240

// 颜色空间
#define DP_HW5X5_DEMOS_COLORMODE    DEMOS_COLORMODE_YUV420
```

### 图像格式
- **原始格式**: BGR planar (分离的 B、G、R 平面)
- **输出分辨率**: 320x240
- **JPEG编码**: 320x240 YUV420

### 获取图像接口
**路径**: `cisdp_sensor.c`

```c
uint32_t app_get_raw_width() {
    return 320;  // DP_INP_CASE == 2 时
}

uint32_t app_get_raw_height() {
    return 240;  // DP_INP_CASE == 2 时
}

uint32_t app_get_raw_addr() {
    return g_wdma3_baseaddr;  // RGB/YUV 数据地址
}
```

---

## 3. SCRFD 人脸检测模型

### 模型配置
**路径**: `EPII_CM55M_APP_S/app/scenario_app/tflm_face_recognition/common_config.h`

```c
// SCRFD 输入尺寸
#define FD_INPUT_TENSOR_WIDTH       160
#define FD_INPUT_TENSOR_HEIGHT      160
#define FD_INPUT_TENSOR_CHANNEL     3

// 检测参数
#define SCRFD_NUM_STRIDES           3     // stride 8, 16, 32
#define SCRFD_NUM_ANCHORS           2     // 每个位置 2 个 anchor
#define SCRFD_NUM_LANDMARKS         5     // 5 点 landmarks

// 阈值
#define FACE_CONF_THRESHOLD         0.52f
#define FACE_NMS_THRESHOLD          0.4f
#define MIN_FACE_SIZE               40
```

### 预处理 (图像缩放)
**路径**: `cvapp_face_recognition.cpp:571-613`

```c
// 计算缩放因子 (原始图像 → 模型输入)
float w_scale = (float)(img_w - 1) / (FD_INPUT_TENSOR_WIDTH - 1);  // (320-1)/(160-1) ≈ 2.0
float h_scale = (float)(img_h - 1) / (FD_INPUT_TENSOR_HEIGHT - 1);  // (240-1)/(160-1) ≈ 1.5

// BGR planar → RGB interleaved 缩放
hx_lib_image_resize_BGR8U3C_to_RGB24_helium(
    (uint8_t *)raw_addr,           // 320x240 BGR planar
    (uint8_t *)fd_resized_img,     // 160x160 RGB interleaved
    img_w, img_h, FD_INPUT_TENSOR_CHANNEL,
    FD_INPUT_TENSOR_WIDTH, FD_INPUT_TENSOR_HEIGHT,
    w_scale, h_scale);
```

**关键点**:
- 使用 **stretch resize** (非等比例缩放)
- 无 letterbox padding
- X 缩放因子: ~2.0
- Y 缩放因子: ~1.5 (因为 320x240 → 160x160 不是等比例)

---

## 4. 后处理与坐标转换

### 后处理流程
**路径**: `scrfd_postprocessing.cc:128-352`

#### 步骤 1: 解码 bbox (模型空间)

```c
// anchor 中心点 (模型空间 160x160)
float cx = (w + 0.5f) * stride;    // grid位置转中心
float cy = (h + 0.5f) * stride;

// 距离解码 (distance-based)
float x1 = cx - d_left * stride;
float y1 = cy - d_top * stride;
float x2 = cx + d_right * stride;
float y2 = cy + d_bottom * stride;

// 裁剪到模型输入边界
x1 = fmaxf(0.0f, x1);
y1 = fmaxf(0.0f, y1);
x2 = fminf((float)net->input_w, x2);  // 160
y2 = fminf((float)net->input_h, y2);  // 160
```

#### 步骤 2: 映射到原始图像空间

```c
// 缩放因子: 模型空间 → 原始图像空间
float scale_x = (float)image_w / net->input_w;  // 320/160 = 2.0
float scale_y = (float)image_h / net->input_h;  // 240/160 = 1.5

// 坐标映射
float orig_x1 = x1 * scale_x;
float orig_y1 = y1 * scale_y;
float orig_x2 = x2 * scale_x;
float orig_y2 = y2 * scale_y;

// 存储为 (x, y, w, h) 格式
det.bbox.x = orig_x1;
det.bbox.y = orig_y1;
det.bbox.w = orig_x2 - orig_x1;
det.bbox.h = orig_y2 - orig_y1;
```

#### 步骤 3: Landmark 解码与映射

```c
// Landmark 相对于 anchor 中心的偏移
float lm_x = cx + kp_dx * stride;  // 模型空间
float lm_y = cy + kp_dy * stride;

// 映射到原始图像空间
det.landmarks[k].x = lm_x * scale_x;  // × 2.0
det.landmarks[k].y = lm_y * scale_y;  // × 1.5
```

### Landmark 顺序
```c
#define SCRFD_LM_LEFT_EYE       0   // 左眼
#define SCRFD_LM_RIGHT_EYE      1   // 右眼
#define SCRFD_LM_NOSE           2   // 鼻尖
#define SCRFD_LM_LEFT_MOUTH     3   // 左嘴角
#define SCRFD_LM_RIGHT_MOUTH    4   // 右嘴角
```

---

## 5. 数据传输格式

### JSON 构建
**路径**: `send_result.cpp:894-994`

```c
void send_face_recognition_json(
    el_img_t* jpeg_img,      // JPEG 图像
    el_box_t* face_box,      // bbox [x, y, w, h]
    const float* embedding,  // 512D embedding
    int embedding_dim,
    const float* landmarks,  // 5 点 landmarks
    float confidence,
    ...
)
```

### JSON 格式

```json
{
  "type": 1,
  "name": "FACE_RESULT",
  "code": 0,
  "data": {
    "image": "<base64_jpeg>",
    "resolution": [320, 240],
    "faces": [{
      "bbox": [x, y, w, h],
      "confidence": 0.695,
      "landmarks": [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5]],
      "embedding": "<base64_float32>",
      "embedding_format": "float32_base64"
    }]
  }
}
```

### 关键点
- **resolution**: JPEG 图像的实际尺寸 (320x240)
- **bbox**: 原始图像空间的坐标 (相对于 320x240)
- **landmarks**: 原始图像空间的坐标 (相对于 320x240)

### 串口帧格式
```
\r{...json...}\n
```

---

## 6. 前端解析与渲染

### 前端代码
**路径**: `tools/face_recognition_debug/frontend/app.js`

### 坐标转换流程 (lines 273-308)

```javascript
// 图像绘制到 Canvas 的缩放计算
const canvasW = 640;  // Canvas 固定宽度
const canvasH = 480;  // Canvas 固定高度

const imgW = actual_jpeg_width;   // 320 (从 JPEG 解码)
const imgH = actual_jpeg_height;  // 240

// 等比例缩放 + 居中 (letterbox)
const canvasRatio = canvasW / canvasH;  // 1.333
const imgRatio = imgW / imgH;            // 1.333 (320/240)

if (imgRatio > canvasRatio) {
    drawWidth = canvasW;              // 640
    drawHeight = drawWidth / imgRatio; // 480
    offsetX = 0;
    offsetY = (canvasH - drawHeight) / 2;  // 0
} else {
    drawHeight = canvasH;             // 480
    drawWidth = drawHeight * imgRatio; // 640
    offsetX = (canvasW - drawWidth) / 2;   // 0
    offsetY = 0;
}

// 缩放因子
const scale = drawWidth / imgW;  // 640 / 320 = 2.0
```

### bbox 绘制 (lines 344-428)

```javascript
// 接收的 bbox 是原始图像空间坐标 (320x240)
const [x, y, w, h] = face.bbox;

// 转换到 Canvas 坐标
const canvasX = offsetX + x * scale;  // 0 + x * 2.0
const canvasY = offsetY + y * scale;  // 0 + y * 2.0
const canvasW = w * scale;            // w * 2.0
const canvasH = h * scale;            // h * 2.0

// 绘制矩形框
ctx.strokeRect(canvasX, canvasY, canvasW, canvasH);
```

### Landmark 绘制 (lines 389-415)

```javascript
face.landmarks.forEach((lm, i) => {
    // lm[0], lm[1] 是原始图像空间坐标 (320x240)
    const lmX = offsetX + lm[0] * scale;  // 0 + lm[0] * 2.0
    const lmY = offsetY + lm[1] * scale;  // 0 + lm[1] * 2.0

    ctx.arc(lmX, lmY, 4, 0, 2 * Math.PI);
    ctx.fill();
});
```

---

## 7. 关键坐标系统

| 坐标系 | 尺寸 | 用途 |
|--------|------|------|
| 传感器原始 | 640×480 | OV5647 输出 (binning) |
| 下采样后 | 320×240 | 实际处理的图像尺寸 |
| 模型输入 | 160×160 | SCRFD 检测空间 |
| JSON 传输 | 320×240 | bbox/landmarks 坐标基准 |
| Canvas 显示 | 640×480 | 前端渲染空间 |

### 坐标转换链

```
模型空间 (160×160)
    ↓ ×scale_x(2.0), ×scale_y(1.5)
原始图像空间 (320×240) ← JSON 传输
    ↓ ×scale(2.0)
Canvas 空间 (640×480)
```

---

## 8. 潜在问题点

### 问题 1: 非等比例缩放
**位置**: `cvapp_face_recognition.cpp:579-580`

```c
float w_scale = (float)(img_w - 1) / (FD_INPUT_TENSOR_WIDTH - 1);  // 2.0
float h_scale = (float)(img_h - 1) / (FD_INPUT_TENSOR_HEIGHT - 1);  // 1.5
```

- 320×240 → 160×160 使用 stretch resize (非等比例)
- X 方向缩放 2.0，Y 方向缩放 1.5
- 这会导致人脸检测时产生形变

### 问题 2: 后处理坐标映射
**位置**: `scrfd_postprocessing.cc:286-287`

```c
float scale_x = (float)image_w / net->input_w;  // 320/160 = 2.0
float scale_y = (float)image_h / net->input_h;  // 240/160 = 1.5
```

- 正确使用了不同的 X/Y 缩放因子
- 应该能正确还原坐标

### 问题 3: JSON resolution 字段
**位置**: `send_result.cpp:229`

```c
std::string img_res_2_json_str(const el_img_t* img) {
    return concat_strings("\"resolution\": [",
        std::to_string(img->width), ", ",   // JPEG 宽度
        std::to_string(img->height), "]");  // JPEG 高度
}
```

- resolution 来自 JPEG 图像尺寸
- **需要验证**: JPEG 尺寸是否与 bbox 坐标空间一致

### 问题 4: 前端缩放计算
**位置**: `app.js:301`

```javascript
const scale = drawWidth / img.width;
```

- 使用 JPEG 解码后的实际宽度
- **关键**: img.width 应该与 JSON 中的 resolution 一致

### 问题 5: JPEG 图像来源
**位置**: `cvapp_face_recognition.cpp:647-649`

```c
cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);
jpeg_img.width = img_w;   // 320
jpeg_img.height = img_h;  // 240
```

- JPEG 尺寸手动设置为 img_w × img_h
- **需要验证**: cisdp_get_jpginfo 返回的 JPEG 实际编码尺寸

---

## 调试检查清单

1. [ ] 验证 `app_get_raw_width()` 返回 320
2. [ ] 验证 `app_get_raw_height()` 返回 240
3. [ ] 验证 JPEG 编码尺寸与 resolution 字段一致
4. [ ] 检查前端 `img.width` 与 JSON `resolution[0]` 是否匹配
5. [ ] 在串口输出中检查 bbox 坐标范围是否在 [0, 320] × [0, 240]
6. [ ] 验证前端 `scale` 计算: 应该是 640/320 = 2.0

---

## 关键代码文件

| 文件 | 作用 |
|------|------|
| `cis_sensor/cis_ov5647/cisdp_cfg.h` | 摄像头分辨率配置 |
| `cis_sensor/cis_ov5647/cisdp_sensor.c` | 图像获取接口 |
| `common_config.h` | 模型输入尺寸、阈值 |
| `cvapp_face_recognition.cpp` | 预处理、推理调用 |
| `scrfd_postprocessing.cc` | bbox/landmark 解码与坐标映射 |
| `send_result.cpp` | JSON 构建与传输 |
| `frontend/app.js` | 前端渲染与坐标转换 |
