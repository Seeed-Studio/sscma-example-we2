# Grove Vision AI Module V2 - 视觉应用开发参考指南

本文档整合了现有项目的固件参数、内存布局和关键配置信息，为后续开发视觉应用提供参考。

---

## 1. 硬件内存布局 (HX6538/WiseEye2)

### 1.1 物理 SRAM 配置

| 内存区域 | 基地址 | 别名地址 | 大小 | 用途 |
|----------|--------|----------|------|------|
| **ITCM** | 0x00000000 | - | 256 KB | 指令紧耦合内存 |
| **DTCM** | 0x20000000 | - | 256 KB | 数据紧耦合内存 |
| **SRAM0** | 0x24000000 | 0x34000000 | 1 MB | 主 SRAM |
| **SRAM1** | 0x24100000 | 0x34100000 | 1 MB | 辅助 SRAM |
| **SRAM2** | 0x26000000 | 0x36000000 | 384 KB | 第三 SRAM |
| **总计** | - | - | **2.5+ MB** | - |

### 1.2 Flash 配置

| Flash 区域 | 基地址 | 大小 | 用途 |
|------------|--------|------|------|
| **QSPI Flash** | 0x2A000000 | 16 MB | 应用固件 + 模型 |
| **OSPI Flash** | 0x2C000000 | 16 MB | 扩展存储 |

### 1.3 模型存储布局

```
Flash 地址空间:
0x00000000 - 0x00200000: 固件保留区 (2 MB, 实际限制 1 MB)
0x00200000+:            模型存储区 (4KB 对齐)

常见模型地址:
├── 0x200000 - 第一个模型位置
├── 0x280000 - 第二个模型位置
├── 0x32A000 - 第三个模型位置
├── 0x400000 - 大型模型位置
└── 0xB7B000 - YOLOv8 默认位置
```

---

## 2. 链接器脚本内存分配

### 2.1 标准配置 (NoTrustZone.ld)

| 区域 | 地址 | 大小 | 用途 |
|------|------|------|------|
| CM55M_S_APP_ROM | 0x10000000 | 256 KB | 固件代码 |
| CM55M_S_APP_DATA | 0x30000000 | 256 KB | 数据/BSS/堆/栈 |
| Heap | - | 8 KB | 动态分配 |
| Stack | - | 16 KB | 函数调用栈 |

### 2.2 算法/ML 配置 (Algo_NoTrustZone.ld) - **推荐用于视觉应用**

| 区域 | 地址 | 大小 | 用途 |
|------|------|------|------|
| CM55M_S_APP_ROM | 0x10000000 | 256 KB | 固件代码 |
| CM55M_S_APP_DATA | 0x30000000 | 256 KB | 数据/栈 |
| CM55M_S_SRAM0 | 0x3404D000 | 724 KB | 只读数据 |
| **CM55M_S_SRAM1** | 0x340E0000 | **1.125 MB** | **Tensor Arena** |
| Heap | - | 64 KB | 动态分配 |
| Stack | - | 64 KB | 函数调用栈 |

---

## 3. 图像/视频格式配置

### 3.1 支持的图像格式

| 格式 | 描述 | 每像素字节 | 640x480 大小 |
|------|------|------------|--------------|
| **YUV400** | 灰度 (RAW8) | 1 | 307 KB |
| **YUV420** | 4:2:0 采样 | 1.5 | 461 KB |
| **YUV422** | 4:2:2 采样 | 2 | 614 KB |
| **RGB** | RGB888 | 3 | 922 KB |

### 3.2 JPEG 编码配置

**硬件约束:**
- 最大分辨率: 1023 x 1023
- 最小分辨率: 16 x 16
- 宽度/高度: 必须是 16 的倍数
- ROI 位置: 必须是 16 的倍数

**量化表选项:**
- `JPEG_ENC_QTABLE_4X`: 标准质量 (更高压缩)
- `JPEG_ENC_QTABLE_10X`: 高质量 (更低压缩)

**JPEG 循环缓冲帧数:**
- YUV400: 10 帧
- YUV420: 6 帧
- YUV422: 3 帧

### 3.3 Bayer 模式支持 (HW5x5 去马赛克)

- BGGR, GBRG, GRBG, RGGB

---

## 4. WDMA 帧缓冲配置

### 4.1 默认缓冲布局

```
SRAM0_ALIAS (0x34000000)
    │
    ├── 0x30000: WDMA1 起始地址
    │            大小: 0x5F400 (391 KB)
    │            用途: HW2x2 输出 / CDM 数据
    │
    ├── 0x8F400: WDMA2 起始地址
    │            大小: 0x4B000 (299 KB)
    │            用途: JPEG 压缩输出
    │
    └── 0xDA400: WDMA3 起始地址
                 大小: 依格式而定
                 用途: HW5x5 输出 (YUV/RGB)
```

### 4.2 WDMA3 缓冲大小计算 (640x480)

| 格式 | 计算公式 | 大小 |
|------|----------|------|
| YUV400 | W × H | 307,200 B |
| YUV420 | W × H × 1.5 | 460,800 B |
| YUV422 | W × H × 2 | 614,400 B |
| RGB | W × H × 3 | 921,600 B |

---

## 5. 传感器数据路径 (SensorDP)

### 5.1 可用数据路径

| 路径名称 | 描述 | 输出 |
|----------|------|------|
| `PATH_INP_WDMA2` | 传感器 → INP → WDMA2 | RAW 直通 |
| `PATH_INP_HW2x2_CDM` | 传感器 → INP → 2x2 → CDM → WDMA1 | 降采样 + 运动检测 |
| `PATH_INP_HW5x5` | 传感器 → INP → 5x5 → WDMA3 | 去马赛克 |
| `PATH_INP_HW5x5_JPEG` | 传感器 → INP → 5x5 → JPEG → WDMA2 | 全压缩流程 |
| `PATH_INT1` | HW2x2 + HW5x5 + JPEG 并行 | 完整处理 |
| `PATH_INTNOJPEG` | HW2x2 + HW5x5 无 JPEG | 仅图像处理 |

### 5.2 INP 子采样模式

| 模式 | 缩减比例 |
|------|----------|
| `INP_SUBSAMPLE_1X` | 无缩减 |
| `INP_SUBSAMPLE_2X` | 1/2 |
| `INP_SUBSAMPLE_4X` | 1/4 |
| `INP_SUBSAMPLE_8TO2` | 8:2 |
| `INP_BINNING_6TO2_B` | 6:2 |

---

## 6. 摄像头传感器配置

### 6.1 支持的传感器

| 传感器 | 配置宏 | 额外标志 | 典型分辨率 |
|--------|--------|----------|------------|
| OV5647 | `cis_ov5647` | - | 640x480 (binning) |
| IMX219 | `cis_imx219` | `-DCIS_IMX` | 640x480 |
| IMX477 | `cis_imx477` | `-DCIS_IMX` | 640x480 |
| IMX708 | `cis_imx708` | `-DCIS_IMX` | 640x480 |
| HM0360 | `cis_hm0360` | - | 640x480 |

### 6.2 OV5647 默认配置

```c
I2C 地址:       0x36
MIPI 时钟:      220 MHz
MIPI 通道数:    2
像素深度:       10 bit
传感器分辨率:   1280 x 960 (binning 后 640x480)
输出分辨率:     640 x 480
SPI 时钟:       12 MHz
```

### 6.3 HW5x5 分辨率约束

| 参数 | 值 |
|------|-----|
| 最大裁剪宽度 | 644 |
| 最大裁剪高度 | 484 |
| 最小裁剪宽度 | 8 |
| 最小裁剪高度 | 8 |
| JPEG 路径宽度倍数 | 16 |
| JPEG 路径高度倍数 | 16 |
| FIR 路径宽度倍数 | 8 |
| FIR 路径高度倍数 | 4 |

---

## 7. TensorFlow Lite 推理配置

### 7.1 Tensor Arena 配置

```c
// 典型配置 (从链接器脚本)
Tensor Arena 基地址: 0x340E0000 (SRAM1)
Tensor Arena 大小:   1.125 MB (0x120000)

// 代码中定义
__attribute__((section(".tensor_arena")))
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
```

### 7.2 推理初始化模板

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// 1. 加载模型
const tflite::Model* model = tflite::GetModel((const void*)MODEL_FLASH_ADDR);

// 2. 设置操作解析器
static tflite::MicroMutableOpResolver<N> resolver;
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
// ... 添加所需操作

// 3. 创建解释器
static tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

// 4. 分配张量
interpreter.AllocateTensors();

// 5. 运行推理
TfLiteStatus status = interpreter.Invoke();
```

### 7.3 Vela 编译 (NPU 加速)

```bash
# 标准编译
vela --accelerator-config ethos-u55-64 \
     --optimise Performance \
     model.tflite

# 解决张量缓冲区重叠问题
vela --accelerator-config ethos-u55-64 \
     --optimise Performance \
     --cop-format COP2 \
     --separate-io-regions \
     model.tflite
```

---

## 8. 数据输出协议

### 8.1 SPI 协议格式

```
Header (7 bytes):
├── ID:   2 bytes
├── Type: 1 byte
└── Size: 4 bytes

Payload: 最大 976 KB
```

### 8.2 算法结果类型码

| 任务 | 类型码 | 结构体 |
|------|--------|--------|
| YOLOv8 目标检测 | 0x9A | `struct_yolov8_ob_algoResult` |
| 人脸检测+网格 | 0x9D | `struct_fm_algoResult_with_fps` |
| 人脸关键点 | 0x92 | `struct_fd_fl_t` |
| 姿态估计 | 0x94 | `struct_hp_algoResult` |

### 8.3 检测框结构

```c
struct__box {
    uint32_t x, y, width, height;
};

struct_yolov8_ob {
    struct__box bbox;
    float confidence;
    uint16_t class_idx;
};
```

---

## 9. 常用库选择 (LIB_SEL)

### 9.1 必需库

| 库名称 | 用途 |
|--------|------|
| `tflmtag2412_u55tag2411` | TensorFlow Lite Micro (当前版本) |
| `sensordp` | 传感器数据路径 |
| `spi_ptl` | SPI 协议传输 |

### 9.2 可选库

| 库名称 | 用途 |
|--------|------|
| `img_proc` | 图像处理 |
| `hxevent` | 事件处理中间件 |
| `pwrmgmt` | 电源管理 |
| `JPEGENC` | JPEG 编码 |
| `cmsis_nn_7_0_0` | CMSIS-NN 加速 |

---

## 10. Makefile 关键配置

### 10.1 构建变量

```makefile
# 应用选择
APP_TYPE = my_vision_app

# 工具链
TOOLCHAIN = gnu

# 优化级别
OLEVEL = O2

# RTOS 支持
OS_SEL =         # 裸机
OS_SEL = freertos # FreeRTOS

# CMSIS-NN 加速
LIB_CMSIS_NN_ENALBE = 1
LIB_CMSIS_NN_VERSION = 7_0_0

# TrustZone
TRUSTZONE = y
TRUSTZONE_TYPE = security
TRUSTZONE_FW_TYPE = 1

# 传感器选择
CIS_SUPPORT_INAPP_MODEL = cis_ov5647
```

### 10.2 应用定义

```makefile
# 在 app_name.mk 中
override SCENARIO_APP_SUPPORT_LIST := $(APP_TYPE)
LIB_SEL = tflmtag2412_u55tag2411 sensordp spi_ptl
CIS_SUPPORT_INAPP_MODEL = cis_ov5647
APPL_DEFINES += -DMY_APP_FLAG
```

---

## 11. 视觉应用目录结构模板

```
my_vision_app/
├── my_vision_app.mk           # 构建配置
├── common_config.h            # 模型地址、特性标志
├── my_vision_app.c/.h         # 主入口 (app_main)
├── cvapp_inference.cpp        # TFLite 推理引擎
├── send_result.cpp            # SPI 协议格式化
├── postprocessing.cc          # NMS、结果过滤
├── memory_manage.c/.h         # 自定义内存分配器
├── cis_sensor/
│   └── cis_ov5647/
│       ├── cisdp_cfg.h        # 传感器数据路径配置
│       └── cisdp_sensor.h     # 传感器参数
├── Algo_NoTrustZone.ld        # GNU 链接器脚本
└── hardfault_handler.c        # 异常处理
```

---

## 12. 关键约束总结

| 约束项 | 值 |
|--------|-----|
| 最大固件大小 | 1 MB |
| 最大模型大小 | ~2-4 MB (量化后) |
| Tensor Arena | 1.125 MB |
| 堆/栈 (算法配置) | 64 KB / 64 KB |
| JPEG 最大分辨率 | 1023 x 1023 |
| 模型地址对齐 | 4 KB (0x1000) |
| 串口波特率 | 921600 |

---

## 13. 关键文件位置

| 文件/目录 | 路径 |
|-----------|------|
| 设备内存映射 | `EPII_CM55M_APP_S/device/inc/WE2_device_addr.h` |
| 链接器脚本 | `EPII_CM55M_APP_S/linker_script/gcc/` |
| SensorDP 库 | `EPII_CM55M_APP_S/library/sensordp/inc/` |
| JPEG 编码器 | `EPII_CM55M_APP_S/library/JPEGENC/` |
| 硬件驱动 | `EPII_CM55M_APP_S/drivers/inc/hx_drv_*.h` |
| 应用示例 | `EPII_CM55M_APP_S/app/scenario_app/` |

---

## 附录 A: 内存布局可视化

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flash (16 MB)                            │
├─────────────────────────────────────────────────────────────────┤
│ 0x00000000 ─ 0x00100000: 固件区 (1 MB 限制)                     │
│ 0x00200000+:             模型存储区                             │
│   ├── Model 1 @ 0x200000                                        │
│   ├── Model 2 @ 0x400000                                        │
│   └── ...                                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        SRAM 布局                                │
├─────────────────────────────────────────────────────────────────┤
│ SRAM0 (1 MB) @ 0x34000000                                       │
│   ├── 0x30000:  WDMA1 (391 KB) - HW2x2/CDM                      │
│   ├── 0x8F400:  WDMA2 (299 KB) - JPEG 输出                      │
│   └── 0xDA400:  WDMA3 (变化)   - HW5x5 YUV/RGB                  │
├─────────────────────────────────────────────────────────────────┤
│ SRAM1 (1 MB) @ 0x34100000                                       │
│   └── 0xE0000:  Tensor Arena (1.125 MB)                         │
├─────────────────────────────────────────────────────────────────┤
│ SRAM2 (384 KB) @ 0x36000000                                     │
│   └── 预留/其他用途                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 附录 B: 数据流示意图

```
┌────────────┐    ┌─────┐    ┌───────┐    ┌────────┐    ┌───────┐
│   Camera   │───>│ INP │───>│ HW2x2 │───>│  CDM   │───>│ WDMA1 │
│  (Sensor)  │    │     │    │       │    │ Motion │    │       │
└────────────┘    │     │    └───────┘    └────────┘    └───────┘
                  │     │
                  │     │    ┌───────┐    ┌────────┐    ┌───────┐
                  │     │───>│ HW5x5 │───>│  JPEG  │───>│ WDMA2 │
                  │     │    │Demosaic│   │Encoder │    │       │
                  └─────┘    │       │    └────────┘    └───────┘
                             │       │
                             │       │───────────────>  ┌───────┐
                             │       │                  │ WDMA3 │
                             └───────┘                  │YUV/RGB│
                                                        └───────┘
                                                            │
                                                            v
                                                    ┌───────────────┐
                                                    │ TFLite Micro  │
                                                    │  Inference    │
                                                    │ (Tensor Arena)│
                                                    └───────────────┘
                                                            │
                                                            v
                                                    ┌───────────────┐
                                                    │ Post-Process  │
                                                    │ NMS/Decode    │
                                                    └───────────────┘
                                                            │
                                                            v
                                                    ┌───────────────┐
                                                    │  SPI/UART     │
                                                    │   Output      │
                                                    └───────────────┘
```

---

*文档生成日期: 2024-12*
*基于现有 scenario_app 分析整理*
