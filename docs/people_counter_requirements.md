# 人流计数传感器 - 需求文档 v2

## 1. 产品概述

基于 SSCMA 官方固件扩展的人流计数传感器，在保持原有功能兼容性的同时，添加目标追踪和越线计数后处理模块。

**系统架构**：
```
云端/手机 ←──WiFi──→ ESP32-S3 ←──UART──→ Grove Vision AI V2
                    (透传)              (SSCMA + 追踪 + 计数)
```

**设计原则**：
- 基于 `sscma-example-we2` 官方固件升级
- 保持原有 AT 命令和模型切换功能完全兼容
- 追踪和计数作为可选后处理模块
- 新增 AT 命令控制追踪/计数功能

---

## 2. 功能需求

### 2.1 核心功能（新增）

| 功能 | 描述 |
|------|------|
| 目标追踪 | 对检测结果进行跨帧关联，分配唯一 Track ID |
| 越线计数 | 检测目标穿越计数线，统计进/出人数 |

### 2.2 兼容性要求

| 原有功能 | 兼容性 |
|----------|--------|
| `AT+MODEL=n` | ✅ 完全兼容，可切换模型 |
| `AT+INVOKE=n,d,r` | ✅ 兼容，追踪作为后处理自动启用 |
| `AT+ALGO=n` | ✅ 兼容，检测类算法自动启用追踪 |
| `AT+TSCORE` / `AT+TIOU` | ✅ 兼容 |
| 其他 AT 命令 | ✅ 完全兼容 |

### 2.3 追踪模式

| 模式 | 描述 |
|------|------|
| 关闭 | 不进行追踪，与原固件行为一致 |
| 追踪 | 启用 ByteTrack，输出带 track_id 的检测结果 |
| 计数 | 启用追踪 + 越线计数 |

---

## 3. 接口规格

### 3.1 新增 AT 命令

**追踪控制**：
```
AT+TRACK=mode           设置追踪模式 (0=关闭, 1=追踪, 2=计数)
AT+TRACK?               查询追踪模式
响应: +TRACK:mode

AT+TRACKRST             重置追踪器状态（清空所有轨迹）
```

**计数线配置**（mode=2 时有效）：
```
AT+LINE=x1,y1,x2,y2     设置计数线坐标（像素）
AT+LINE?                查询计数线
响应: +LINE:x1,y1,x2,y2
```

**计数操作**：
```
AT+COUNT?               查询计数值
响应: +COUNT:in,out

AT+COUNTRST             重置计数值
```

**参数配置**：
```
AT+TRACKAGE=n           设置轨迹最大丢失帧数 (默认: 30)
AT+TRACKAGE?            查询轨迹最大丢失帧数
```

### 3.2 输出格式变更

**原有检测输出** (`AT+INVOKE`)：
```json
{
  "type": 1,
  "name": "INVOKE",
  "code": 0,
  "data": {
    "count": 5,
    "boxes": [
      {"x": 100, "y": 50, "w": 80, "h": 120, "score": 85, "target": 0}
    ]
  }
}
```

**追踪模式输出** (`AT+TRACK=1`)：
```json
{
  "type": 1,
  "name": "INVOKE",
  "code": 0,
  "data": {
    "count": 5,
    "tracks": [
      {"x": 100, "y": 50, "w": 80, "h": 120, "score": 85, "target": 0, "track_id": 3}
    ]
  }
}
```

**计数模式输出** (`AT+TRACK=2`)：
```json
{
  "type": 1,
  "name": "INVOKE",
  "code": 0,
  "data": {
    "count": 5,
    "tracks": [...],
    "counter": {"in": 12, "out": 8}
  }
}
```

**越线事件上报**（可选，通过配置启用）：
```json
{"type": 1, "name": "CROSS", "data": {"track_id": 3, "direction": 1}}
```

### 3.3 配置持久化

所有配置使用 SSCMA 原有的 FlashDB 存储机制：
- 追踪模式
- 计数线坐标
- 追踪参数

---

## 4. 技术架构

### 4.1 模块集成位置

```
sscma/callback/invoke.hpp
    │
    ├── event_loop_cam()
    │       │
    │       ├── algorithm->run()      // 检测推理
    │       │
    │       └── [NEW] post_process()  // 追踪 + 计数后处理
    │               │
    │               ├── ByteTracker::update()
    │               │
    │               └── Counter::update()
```

### 4.2 代码结构

```
sscma_micro/
├── sscma/
│   ├── callback/
│   │   ├── invoke.hpp          // 修改: 添加追踪后处理调用
│   │   └── track.hpp           // 新增: 追踪/计数 AT 命令
│   │
│   └── extension/
│       ├── bytetrack/          // 新增: ByteTrack 实现
│       │   ├── byte_tracker.hpp
│       │   ├── strack.hpp
│       │   ├── kalman_filter.hpp
│       │   └── lapjv.hpp
│       │
│       └── counter/            // 新增: 越线计数器
│           └── line_counter.hpp
```

### 4.3 追踪器接口

```cpp
// 追踪器单例
class TrackingModule {
public:
    static TrackingModule* get_instance();

    // 模式控制
    void set_mode(TrackMode mode);  // OFF=0, TRACK=1, COUNT=2
    TrackMode get_mode() const;

    // 追踪处理
    void update(const std::forward_list<el_box_t>& detections,
                std::forward_list<el_track_t>& tracks);
    void reset();

    // 计数线
    void set_line(int16_t x1, int16_t y1, int16_t x2, int16_t y2);
    void get_line(int16_t& x1, int16_t& y1, int16_t& x2, int16_t& y2) const;

    // 计数
    void get_count(int32_t& in_count, int32_t& out_count) const;
    void reset_count();

    // 参数
    void set_max_age(uint8_t age);
    uint8_t get_max_age() const;

private:
    ByteTracker _tracker;
    LineCounter _counter;
    TrackMode _mode;
};

// 追踪结果类型
struct el_track_t {
    int16_t x;
    int16_t y;
    int16_t w;
    int16_t h;
    uint8_t score;
    uint8_t target;
    uint16_t track_id;  // 新增字段
};
```

### 4.4 invoke.hpp 修改点

```cpp
// 在 event_loop_cam() 中，algorithm->run() 之后添加:

// 追踪后处理
auto tracking = TrackingModule::get_instance();
if (tracking->get_mode() != TrackMode::OFF) {
    std::forward_list<el_track_t> tracks;
    tracking->update(algorithm->get_results(), tracks);

    // 输出追踪结果而非检测结果
    if (tracking->get_mode() == TrackMode::COUNT) {
        int32_t in_count, out_count;
        tracking->get_count(in_count, out_count);
        event_reply(track_results_with_counter_2_json_str(tracks, in_count, out_count));
    } else {
        event_reply(track_results_2_json_str(tracks));
    }
} else {
    // 原有逻辑
    event_reply(algorithm_results_2_json_str(algorithm));
}
```

---

## 5. 算法规格

### 5.1 ByteTrack 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| track_thresh | 0.5 | 高置信度追踪阈值 |
| high_thresh | 0.6 | 第一次匹配阈值 |
| match_thresh | 0.8 | 第二次匹配阈值 |
| max_age | 30 | 最大丢失帧数 |
| min_hits | 3 | 最小确认帧数 |

### 5.2 计数线

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 起点 (x1, y1) | (0, 240) | 画面左侧中点 |
| 终点 (x2, y2) | (640, 240) | 画面右侧中点 |
| 方向定义 | 向量叉积 | 正=进入, 负=离开 |

### 5.3 性能指标

| 指标 | 目标值 |
|------|--------|
| 追踪延迟 | < 3ms/帧 |
| 最大追踪数 | 20 人 |
| 内存占用 | < 25KB |
| 计数准确率 | > 90% |

---

## 6. 内存规划

### 6.1 静态内存分配

```
追踪模块内存:
├── ByteTracker      ~15 KB
│   ├── STrack pool (20 tracks × 600B)  ~12 KB
│   ├── Kalman filter states             ~2 KB
│   └── Matching buffers                 ~1 KB
│
├── LineCounter      ~0.5 KB
│   ├── Track history (20 × 24B)        ~0.5 KB
│
└── 总计             ~15.5 KB
```

### 6.2 Tensor Arena 不变

- 保持 `CONFIG_SSCMA_TENSOR_ARENA_SIZE = 1110 KB`
- 追踪模块使用独立的静态内存

---

## 7. 实现计划

### 7.1 阶段一：ByteTrack 移植

| 任务 | 说明 |
|------|------|
| 复制 ByteTrack 源码 | 从 sscma-micro/extension/bytetrack/ |
| 适配 Eigen 库 | 使用 sscma-micro/3rdparty/eigen/ |
| 创建 TrackingModule | 单例封装 |
| 集成到 invoke.hpp | 后处理调用点 |

### 7.2 阶段二：计数器实现

| 任务 | 说明 |
|------|------|
| 复制 Counter 源码 | 从 sscma-micro/extension/counter/ |
| 集成到 TrackingModule | 越线检测 |
| 添加计数线配置 | 运行时可调 |

### 7.3 阶段三：AT 命令

| 任务 | 说明 |
|------|------|
| 创建 track.hpp | AT 命令处理 |
| 注册命令 | 在 main_task.hpp 中 |
| FlashDB 持久化 | 使用原有 storage 机制 |

### 7.4 阶段四：输出格式

| 任务 | 说明 |
|------|------|
| 定义 el_track_t | 带 track_id 的检测结果 |
| 实现 JSON 序列化 | track_results_2_json_str() |
| 修改 event_reply | 根据模式选择输出 |

---

## 8. 测试用例

### 8.1 功能测试

```bash
# 1. 基础兼容性
AT+MODEL=1          # 加载模型
AT+INVOKE=-1,0,0    # 持续推理 (应与原固件一致)

# 2. 追踪模式
AT+TRACK=1          # 启用追踪
AT+INVOKE=-1,0,0    # 输出应包含 track_id

# 3. 计数模式
AT+TRACK=2          # 启用计数
AT+LINE=0,120,320,120  # 设置计数线
AT+INVOKE=-1,0,0    # 输出应包含 counter

# 4. 计数查询
AT+COUNT?           # 查询当前计数
AT+COUNTRST         # 重置计数

# 5. 模式切换
AT+TRACK=0          # 关闭追踪
AT+INVOKE=-1,0,0    # 应恢复原有输出格式
```

### 8.2 边界测试

```bash
# 模型切换后追踪器重置
AT+TRACK=1
AT+INVOKE=-1,0,0
AT+BREAK
AT+MODEL=2          # 切换模型应自动 reset 追踪器
AT+INVOKE=-1,0,0

# 参数持久化
AT+TRACK=2
AT+LINE=100,100,200,200
AT+RST              # 重启
AT+TRACK?           # 应返回 2
AT+LINE?            # 应返回 100,100,200,200
```

---

## 9. 待复用代码

| 源码位置 | 用途 |
|----------|------|
| `sscma-micro/sscma/extension/bytetrack/` | ByteTrack 追踪算法 |
| `sscma-micro/sscma/extension/counter/` | 越线计数器 |
| `sscma-micro/3rdparty/eigen/` | 矩阵运算库 |
| `sscma-micro/core/utils/el_hash.h` | 配置哈希 |
| `sscma-micro/sscma/static_resource.hpp` | 全局资源访问 |

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Eigen 库过大 | 只包含必要头文件 |
| 追踪内存不足 | 限制最大追踪数为 20 |
| 帧率下降 | 优化匈牙利算法，使用静态分配 |
| 与原固件不兼容 | 默认关闭追踪，保持原有行为 |

---

## 11. 文档更新

需要更新的文档：
- AT 命令协议文档 (新增 TRACK/LINE/COUNT 命令)
- 固件发布说明 (新增功能说明)
- 用户指南 (计数线配置方法)
