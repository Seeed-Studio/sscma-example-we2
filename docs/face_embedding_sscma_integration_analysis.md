# 工作量评估：将 tflm_face_embedding 集成到 SSCMA 框架

> 评估日期: 2024-12-15
> 状态: 待确认

## 概述

**目标**: 将 `tflm_face_embedding` 的人脸识别功能集成到 `sscma` 应用的框架中，替换现有的模型推理部分。

**评估结论**: 中等偏大工作量，预计需要 **9-14 人天**（不含调试）。

---

## 架构差异分析

| 特性 | tflm_face_embedding | sscma |
|------|---------------------|-------|
| **运行环境** | 裸机 (bare-metal) | FreeRTOS |
| **推理引擎** | 直接使用 TFLite Micro | 通过 sscma_micro 抽象层 |
| **模型管理** | 硬编码地址 | ModelFactory 动态创建 |
| **输出协议** | 自定义二进制 UART (594 bytes) | AT 命令 + JSON |
| **摄像头** | 事件驱动回调 | AT 命令触发 |
| **内存管理** | 自定义 mm_reserve_align | sscma_micro 内置 |

### SSCMA 应用架构

```
sscma.cpp:app_main()
    ↓
创建 FreeRTOS 任务 → xTaskCreate(app, "app", 20480, NULL, 3, NULL)
    ↓
sscma::main_task::run()
    ↓
StaticResource 单例初始化
    ├─ Device::getInstance()
    ├─ EngineTFLite 初始化
    └─ Executor 初始化 (线程池)
    ↓
ATServer 启动
    ├─ AT+SAMPLE → 采样图像
    ├─ AT+INVOKE → 推理
    ├─ AT+MODEL  → 加载模型
    └─ AT+CONFIG → 配置阈值
```

### tflm_face_embedding 架构

```
app_main()
    ↓
cisdp 摄像头初始化
    ↓
cv_face_embedding_init()
    ├─ 分配 SCRFD tensor arena (220 KB)
    ├─ 分配 MobileFaceNet tensor arena (1200 KB)
    ├─ 加载两个模型
    └─ 创建两个解释器
    ↓
事件循环 (dp_app_cv_fd_fm_eventhdl_cb)
    ↓
cv_face_embedding_run()
    ├─ SCRFD 推理 (人脸检测)
    ├─ 人脸对齐 (相似变换)
    ├─ MobileFaceNet 推理 (特征提取)
    └─ UART 发送结果
```

---

## 工作模块分解

### 模块 1: 新增 SCRFD 模型类 (sscma_micro 扩展)

**工作量**: ⭐⭐⭐ (中等，2-3 天)

**新建文件**:
```
sscma_micro/sscma/core/model/
├── ma_model_scrfd.h
└── ma_model_scrfd.cpp
```

**主要工作**:
1. 定义 `MA_MODEL_TYPE_SCRFD` 枚举值
2. 继承 `Detector` 基类
3. 实现 `isValid()` - 验证 9 个输出张量 (3 stride × 3 output)
4. 实现 `postprocess()` - 移植后处理逻辑:
   - 多尺度特征解码 (stride 8, 16, 32)
   - Anchor-free 检测框解码
   - 5 点关键点解码
   - NMS (非极大值抑制)
5. 扩展输出结构支持 landmarks

**复用代码**:
- `scrfd_postprocessing.cc` (~400 行)
- `scrfd_types.h`

**需修改**:
- `ma_types.h` - 添加 `MA_MODEL_TYPE_SCRFD`
- `ma_model_factory.cpp` - 注册 SCRFD

---

### 模块 2: 新增 FaceEmbedding 模型类 (sscma_micro 扩展)

**工作量**: ⭐⭐ (较小，1 天)

**新建文件**:
```
sscma_micro/sscma/core/model/
├── ma_model_face_embedding.h
└── ma_model_face_embedding.cpp
```

**主要工作**:
1. 定义 `MA_MODEL_TYPE_FACE_EMBEDDING` 枚举值
2. 继承 `Model` 基类
3. 实现 `isValid()` - 验证 112x112x3 输入, 128D 输出
4. 实现 `preprocess()` - INT8 量化输入转换
5. 实现 `postprocess()` - 反量化 + L2 归一化
6. 添加 `getEmbedding()` 方法返回 128D 向量

**复用代码**:
- `cvapp_face_embedding.cpp` 中的量化/反量化逻辑 (~50 行)

---

### 模块 3: 人脸对齐模块 (sscma_micro 扩展)

**工作量**: ⭐⭐⭐ (中等，1-2 天)

**新建文件**:
```
sscma_micro/sscma/core/cv/
├── ma_face_alignment.h
└── ma_face_alignment.cpp
```

**主要工作**:
1. 实现 `compute_face_alignment()` - 计算相似变换矩阵
2. 实现 `apply_face_alignment()` - 双线性插值 + 颜色转换
3. 定义 ArcFace 标准参考点 (112x112)
4. 支持 BGR planar → RGB interleaved 转换

**复用代码**:
- `face_alignment.c` (~200 行) - 几乎可以直接移植

**ArcFace 参考点**:
```c
static const Point2f REFERENCE_LANDMARKS[5] = {
    {38.2946f, 51.6963f},   // 左眼
    {73.5318f, 51.5014f},   // 右眼
    {56.0252f, 71.7366f},   // 鼻子
    {41.5493f, 92.3655f},   // 左嘴角
    {70.7299f, 92.2041f}    // 右嘴角
};
```

---

### 模块 4: 双模型串联推理管道 (核心难点)

**工作量**: ⭐⭐⭐⭐ (较大，3-5 天)

**修改文件**:
- `sscma_micro/sscma/server/at/callback/invoke.hpp`
- 或新建 `invoke_face.hpp`

**主要工作**:
1. 支持加载两个模型 (SCRFD + MobileFaceNet)
2. 修改推理流程:
   ```
   Camera Frame → SCRFD 检测 → 人脸对齐 → MobileFaceNet → Embedding
   ```
3. 管理两个 tensor arena (220KB + 1200KB)
4. 处理 D-Cache 一致性 (NPU 交互)
5. 支持 "最佳人脸" 选择逻辑

**挑战**:
- **sscma_micro 当前仅支持单模型推理**
- 需要重构 `prepareModel()` 和 `eventLoopCamera()`
- Engine 类需要支持多实例或模型热切换
- 内存管理需要适配

**可能的解决方案**:
```cpp
// 方案1: 创建专用的 FaceRecognizer 类
class FaceRecognizer {
    Engine* scrfd_engine_;
    Engine* embedding_engine_;
    SCRFD* scrfd_model_;
    FaceEmbedding* embedding_model_;

    ma_err_t run(const ma_img_t* img);
};

// 方案2: 修改 Engine 支持多模型
class MultiModelEngine : public Engine {
    std::vector<Engine*> engines_;
    ma_err_t loadMultiple(const std::vector<ModelConfig>& configs);
};
```

---

### 模块 5: 输出序列化扩展

**工作量**: ⭐⭐ (较小，0.5 天)

**修改文件**:
- `invoke.hpp` 中的 `serializeAlgorithmOutput()`

**主要工作**:
1. 添加 embedding 向量序列化 (128 floats)
2. 添加 landmarks 序列化 (5 points)
3. 添加质量分数、姿态角等元数据
4. 保持与现有 JSON 格式兼容

**输出格式示例**:
```json
{
  "type": 1,
  "name": "FACE_RESULT",
  "code": 0,
  "data": {
    "image": "<base64_jpeg>",
    "resolution": [640, 480],
    "faces": [{
      "bbox": [120, 80, 100, 120],
      "confidence": 0.95,
      "landmarks": [[145, 110], [175, 108], [160, 135], [148, 160], [172, 158]],
      "embedding": [0.1, 0.2, ..., 0.05],
      "quality": 0.85,
      "pose": {"yaw": 5.2, "pitch": -3.1, "roll": 1.5}
    }],
    "perf": {
      "preprocess": 5,
      "inference": 150,
      "postprocess": 10
    }
  }
}
```

---

### 模块 6: 新增 AT 命令支持 (可选)

**工作量**: ⭐⭐ (较小，1-2 天)

**新建/修改文件**:
- `sscma_micro/sscma/server/at/callback/face.hpp` (新建)
- `ma_server_at.cpp` (注册命令)

**主要工作**:
1. `AT+FACE` - 启用人脸识别模式
2. `AT+ENROLL=<name>` - 注册人脸 (存储 embedding)
3. `AT+IDENTIFY` - 识别人脸 (比对 embedding)
4. `AT+FACELIST` - 列出已注册人脸
5. `AT+FACEDEL=<name>` - 删除人脸
6. 可选: 人脸数据库存储 (Flash/SD)

---

### 模块 7: 配置和构建系统

**工作量**: ⭐ (较小，0.5 天)

**修改文件**:
- `sscma.mk` - 添加人脸相关编译选项
- `sscma_micro` 的编译配置

**主要工作**:
1. 添加编译开关 `MA_MODEL_FACE_EMBEDDING_ENABLE`
2. 条件编译人脸相关模块
3. 配置模型 Flash 地址
4. 调整 linker script 内存分配

---

## 工作量汇总

| 模块 | 复杂度 | 预估工作量 | 依赖 |
|------|--------|------------|------|
| 1. SCRFD 模型类 | 中 | 2-3 天 | 无 |
| 2. FaceEmbedding 模型类 | 小 | 1 天 | 无 |
| 3. 人脸对齐模块 | 中 | 1-2 天 | 无 |
| 4. 双模型推理管道 | **大** | 3-5 天 | 1, 2, 3 |
| 5. 输出序列化 | 小 | 0.5 天 | 4 |
| 6. AT 命令 (可选) | 小 | 1-2 天 | 4, 5 |
| 7. 构建系统 | 小 | 0.5 天 | 全部 |

**总计**: 9-14 人天 (不含调试)

---

## 关键风险分析

### 1. 内存约束 (高风险)

**问题**:
- SCRFD + MobileFaceNet 需要 ~1.5 MB tensor arena
- sscma 已有其他内存开销:
  - FreeRTOS 任务栈 (~20KB per task)
  - AT Server 缓冲区
  - JPEG 编码缓冲区

**缓解措施**:
- 调整 linker script，优化内存分配
- 考虑模型热切换 (不同时加载两个模型)
- 使用更小的 tensor arena (Vela 优化)

### 2. 双模型管理 (高风险)

**问题**:
- sscma_micro 当前设计为单模型
- `StaticResource` 只有一个 `engine` 指针
- `ModelFactory` 不支持同时创建多个模型

**缓解措施**:
- 创建专用的 `FaceRecognizer` 类封装双模型
- 或修改 sscma_micro 架构支持多模型

### 3. 实时性能 (中等风险)

**问题**:
- 人脸对齐增加延迟 (~20ms)
- 双模型推理总延迟 ~150-200ms
- FreeRTOS 调度开销

**缓解措施**:
- 可以接受 5-7 FPS，满足实时需求
- 优化人脸对齐算法 (使用 Helium SIMD)

### 4. D-Cache 一致性 (中等风险)

**问题**:
- NPU (Ethos-U55) 直接访问内存
- CPU 写入后需要 Clean D-Cache
- NPU 写入后需要 Invalidate D-Cache
- sscma_micro 可能没有完善的 Cache 管理

**缓解措施**:
- 复用 tflm_face_embedding 的 Cache flush/invalidate 代码
- 在 Engine 类中添加 Cache 管理接口

---

## 替代方案比较

### 方案 A: 深度集成 (推荐)

**描述**: 在 sscma_micro 库中添加完整的人脸识别支持

**优点**:
- 统一架构，代码可维护
- 可复用 AT 命令系统
- 支持动态切换检测/识别模式

**缺点**:
- 工作量大 (9-14 天)
- 需要修改 sscma_micro 库代码
- 需要深入理解 sscma 架构

### 方案 B: 轻量集成

**描述**: 保持 tflm_face_embedding 独立，仅将输出协议改为 sscma 兼容的 JSON 格式

**优点**:
- 工作量小 (2-3 天)
- 不需要修改 sscma_micro
- 可以快速部署

**缺点**:
- 两套代码维护
- 无法使用 AT 命令
- 无法与其他 sscma 功能集成

### 方案 C: 混合方案

**描述**: 使用 sscma 的 AT Server 和传输层，人脸推理逻辑保持独立模块

**优点**:
- 平衡工作量和功能 (5-7 天)
- 可以使用 AT 命令
- 推理代码相对独立

**缺点**:
- 架构不够优雅
- 需要处理两套内存管理

---

## 建议的实施顺序

### Phase 1: 基础模块 (可独立开发测试)
- 模块 1: SCRFD 模型类
- 模块 2: FaceEmbedding 模型类
- 模块 3: 人脸对齐模块

**里程碑**: 各模块可以独立编译和单元测试

### Phase 2: 核心集成
- 模块 4: 双模型推理管道
- 模块 5: 输出序列化

**里程碑**: 人脸识别功能在 sscma 框架中运行

### Phase 3: 完善功能
- 模块 6: AT 命令支持
- 模块 7: 构建系统

**里程碑**: 完整功能，可发布

---

## 待确认问题

1. **集成深度**: 是否需要完整的 AT 命令支持，还是只需要输出格式兼容？
2. **人脸数据库**: 是否需要在设备端存储已注册人脸？
3. **多人脸支持**: 是否需要同时处理多个人脸，还是只处理最佳人脸？
4. **性能要求**: 可接受的最大延迟是多少？(当前 ~150-200ms)
5. **内存预算**: 是否有其他功能需要同时运行？

---

## 参考文件

### tflm_face_embedding 核心文件
- `EPII_CM55M_APP_S/app/scenario_app/tflm_face_embedding/cvapp_face_embedding.cpp` - 推理管道
- `EPII_CM55M_APP_S/app/scenario_app/tflm_face_embedding/scrfd_postprocessing.cc` - SCRFD 后处理
- `EPII_CM55M_APP_S/app/scenario_app/tflm_face_embedding/face_alignment.c` - 人脸对齐
- `EPII_CM55M_APP_S/app/scenario_app/tflm_face_embedding/common_config.h` - 配置

### sscma_micro 核心文件
- `sscma_micro/sscma/core/model/ma_model_base.h` - 模型基类
- `sscma_micro/sscma/core/model/ma_model_factory.cpp` - 模型工厂
- `sscma_micro/sscma/core/engine/ma_engine_tflite.h` - TFLite 引擎
- `sscma_micro/sscma/server/at/callback/invoke.hpp` - 推理回调
- `sscma_micro/sscma/server/at/callback/resource.hpp` - 全局资源
