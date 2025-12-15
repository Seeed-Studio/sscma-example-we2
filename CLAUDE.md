# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains firmware examples for the Seeed Grove Vision AI Module V2, built on the Himax HX6538 (WiseEye2) ARM Cortex-M55 processor. It provides 22+ scenario applications demonstrating AI inference, computer vision, audio processing, and sensor integration on embedded hardware.

**Target Hardware:** Himax HX6538 (Cortex-M55), Grove Vision AI Module V2
**Primary Language:** C, C++ (TensorFlow Lite), Assembly (ARM)
**Build System:** GNU Make with hierarchical configuration

## Common Development Commands

### Building Firmware

**Standard Build Flow:**
```bash
cd EPII_CM55M_APP_S
make clean
make
```

**Selecting a Scenario Application:**
Edit `EPII_CM55M_APP_S/makefile` and change the `APP_TYPE` variable:
```makefile
APP_TYPE = tflm_yolov8_od  # Change to desired scenario app
```

**Common Scenario Apps:**
- `tflm_yolov8_od` - YOLOv8 object detection
- `tflm_yolo11_od` - YOLOv11 object detection (newer)
- `tflm_yolov8_pose` - Pose estimation
- `tflm_fd_fm` - Face detection and mesh
- `kws_pdm_record` - Keyword spotting with audio
- `allon_sensor_tflm` - Comprehensive sensor + inference example
- `allon_sensor_tflm_freertos` - With FreeRTOS RTOS support

**Output:** `./obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf`

### Generating Firmware Image

```bash
cd we2_image_gen_local/
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/

# Linux
./we2_local_image_gen project_case1_blp_wlcsp.json

# macOS
./we2_local_image_gen_macOS_arm64 project_case1_blp_wlcsp.json

# Windows
we2_local_image_gen.exe project_case1_blp_wlcsp.json
```

**Output:** `./output_case1_sec_wlcsp/output.img` (max 1MB)

### Flashing Firmware

**Linux/macOS:**
```bash
# Install dependencies first
pip install -r xmodem/requirements.txt

# Linux: Grant permissions
sudo setfacl -m u:$USER:rw /dev/ttyACM0

# Flash firmware only
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img

# Flash firmware with model
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="model_zoo/tflm_yolov8_od/yolov8n_od_192_delete_transpose_0xB7B000.tflite 0xB7B000 0x00000"
```

**Windows:**
```cmd
pip install -r xmodem\requirements.txt

python xmodem\xmodem_send.py ^
  --port=COM3 ^
  --baudrate=921600 ^
  --protocol=xmodem ^
  --file=we2_image_gen_local\output_case1_sec_wlcsp\output.img
```

**Alternative (All Platforms):**
```bash
himax-flash-tool -d WiseEye2 -f <path_to_firmware_img>
```

**Important:** After flashing, press the physical reset button on the Grove Vision AI Module V2 to start the new firmware.

### Debugging

**Serial Console (Monitor Output):**
- Baud rate: 921600
- Data: 8 bit, No parity, 1 stop bit, No flow control
- Linux: `minicom -s` or `screen /dev/ttyACM0 921600`
- Windows: TeraTerm
- macOS: `screen /dev/tty.usbmodem* 921600`

**Hardware Debugging (SWD):**
- Use tools in `swd_debugging/pyocd/` with CMSIS-DAP probe
- Supports standard ARM SWD (Serial Wire Debug) protocol

## Architecture Overview

### Directory Structure

```
EPII_CM55M_APP_S/              # Main firmware source tree
├── app/scenario_app/          # 22+ example applications (entry point: app_main())
├── library/                   # Core libraries (inference, DSP, CV, audio, sensors)
├── drivers/                   # Hardware drivers and device abstractions
├── board/                     # Board-level initialization
├── device/                    # Cortex-M55 device definitions (CMSIS Device)
├── CMSIS/                     # ARM CMSIS-Core library
├── os/rtos2_freertos/         # FreeRTOS 10.5.1 (optional RTOS)
├── middleware/                # FatFS (SD card support)
├── options/                   # Build configuration and toolchain settings
├── linker_script/             # Memory layout configurations
├── trustzone/                 # ARM TrustZone-M security partition
└── makefile                   # Master build configuration

we2_image_gen_local/           # Post-build firmware image generation and secure boot
model_zoo/                     # Pre-trained TensorFlow Lite models (.tflite)
xmodem/                        # Python scripts for XMODEM firmware flashing
swd_debugging/                 # SWD debugging support (PyOCD)
```

### Layered Architecture

The firmware follows a clear layered design:

1. **Scenario Applications** - Self-contained examples in `app/scenario_app/[app_name]/`
2. **Framework Layer** - Event handlers, utilities, common configurations
3. **Inference & Processing** - TensorFlow Lite Micro, CMSIS-NN acceleration, sensor data path
4. **Hardware Abstraction** - CMSIS-compliant drivers (I2C, SPI, GPIO, UART, Camera ISP)
5. **OS Layer** - Optional FreeRTOS or bare-metal
6. **ARM CMSIS-Core** - Cortex-M55 initialization and registers
7. **TrustZone** - Security/non-security partition (ARM TrustZone-M)

### Key Architectural Patterns

**Scenario App Pattern:**
- Each app is self-contained with its own `.mk` makefile and entry point (`app_main()`)
- Apps can independently enable/disable libraries and drivers via their makefile
- Each app has `common_config.h` defining model flash addresses and feature flags

**Driver Configuration:**
- Pre-defined platform use cases in `drivers/mk_cfg/`:
  - `drv_onecore_cm55m_s_only` - Single security domain (most common)
  - `drv_dualcore_*` - Multi-core configurations
- Apps can override via `drv_user_defined.mk`

**Camera Sensor Abstraction:**
- Supported sensors: OV5647, IMX219, IMX477, IMX708, HM0360
- Configure in app makefile: `CIS_SUPPORT_INAPP_MODEL = cis_ov5647` (or cis_imx219, etc.)
- Sensor-specific code in `app/scenario_app/[app_name]/cis_sensor/[sensor]/`

**Model Flash Layout:**
- Firmware reserved: 0x00000000 - 0x00200000 (2 MB)
- Models stored from: 0x00200000+ (4KB aligned addresses)
- Each model defines its flash address in `common_config.h` (e.g., `MODEL_ADDR_FLASH 0xB7B000`)
- Multiple models can be flashed simultaneously for dynamic selection

**Event-Driven Processing:**
- `event_handler/` middleware enables asynchronous operation
- Decouples data acquisition (camera, audio) from inference pipeline
- Used in apps like `allon_sensor_tflm`

### Library System

Key libraries (enabled via `LIB_SEL` in app makefile):

- **Inference Engines:**
  - `tflmtag2412_u55tag2411` - TensorFlow Lite Micro (current, recommended)
  - `tflmtag2209_u55tag2205` - Legacy version

- **Acceleration Libraries:**
  - `cmsis_nn` / `cmsis_nn_7_0_0` - ARM CMSIS-NN for neural network acceleration
  - `cmsis_dsp` - ARM CMSIS-DSP for signal processing
  - `cmsis_cv` - ARM CMSIS-CV for computer vision (Git submodule)

- **Sensor & Media:**
  - `sensordp` - Sensor data path and ISP (Image Signal Processing)
  - `JPEGENC` - JPEG image encoder
  - `audio` - Audio processing

- **Communication:**
  - `i2c_comm` - I2C communication
  - `spi_eeprom` - SPI EEPROM access
  - `spi_ptl` - SPI protocol

### Build System

**Key Configuration Variables in `makefile`:**

| Variable | Purpose | Common Values |
|----------|---------|---------------|
| `APP_TYPE` | Select scenario app | `tflm_yolov8_od`, `allon_sensor_tflm`, etc. |
| `TOOLCHAIN` | Compiler | `gnu` (default), `arm` |
| `OLEVEL` | Optimization | `O0`, `O1`, `O2`, `O3` |
| `DEBUG` | Debug symbols | `0` (off), `1` (on) |
| `OS_SEL` | RTOS support | ` ` (bare-metal), `freertos` |
| `LIB_CMSIS_NN_ENALBE` | CMSIS-NN acceleration | `0` (off), `1` (on) |
| `LIB_CMSIS_NN_VERSION` | CMSIS-NN version | `0` (legacy), `7_0_0` (current) |
| `IC_PACKAGE_SEL` | Package type | `WLCSP65`, `LQFP128`, `BGA64` |
| `TRUSTZONE` | TrustZone security | `y`, `n` |

**Makefile Hierarchy:**
```
makefile (master)
  ├─ options/options.mk (defaults)
  ├─ options/rules.mk (build rules)
  ├─ options/toolchain.mk (compiler selection)
  ├─ app/scenario_app/[APP_TYPE]/[APP_TYPE].mk (app config)
  ├─ drivers/mk_cfg/[CONFIG].mk (driver config)
  └─ library/[lib]/[lib].mk (library configs)
```

Each component contributes via `.mk` files using variables like:
```makefile
LIB_*_CSRCDIR = [source directories]
LIB_*_INCDIR = [include directories]
LIB_*_DEFINES = [compilation defines]
```

## Creating a New Scenario App

1. **Create app directory:**
   ```bash
   mkdir EPII_CM55M_APP_S/app/scenario_app/my_custom_app
   cd EPII_CM55M_APP_S/app/scenario_app/my_custom_app
   ```

2. **Create main files:**
   - `my_custom_app.c` with entry point: `void app_main(void)`
   - `my_custom_app.h` with declarations
   - `common_config.h` with model addresses and feature flags
   - `memory_manage.c/.h` if custom memory allocation needed

3. **Create makefile (`my_custom_app.mk`):**
   ```makefile
   override SCENARIO_APP_SUPPORT_LIST := $(APP_TYPE)

   # Enable needed libraries
   LIB_SEL = tflmtag2412_u55tag2411 sensordp

   # Camera sensor selection
   CIS_SUPPORT_INAPP_MODEL = cis_ov5647

   # App-specific defines
   APPL_DEFINES += -DMY_CUSTOM_APP

   # Include common app config
   include $(SCENARIO_APP_ROOT)/[reference_app]/[reference_app].mk
   ```

4. **Add camera sensor support (if needed):**
   ```bash
   mkdir -p cis_sensor/cis_ov5647/
   # Add sensor-specific initialization code
   ```

5. **Create linker scripts:**
   - `*.sct` for ARM compiler
   - `*.ld` for GNU compiler

6. **Update main makefile:**
   ```makefile
   # In EPII_CM55M_APP_S/makefile
   APP_TYPE = my_custom_app
   ```

7. **Build and test:**
   ```bash
   cd EPII_CM55M_APP_S
   make clean && make
   ```

## Working with TensorFlow Lite Models

**Model Preparation:**
1. Quantize model to int8 for embedded deployment
2. Save as `.tflite` format
3. Place in `model_zoo/my_model/`

**Model Flash Configuration:**

In your app's `common_config.h`:
```c
// Choose address > 0x200000, 4KB aligned
#define MODEL_ADDR_FLASH 0xC00000
#define MODEL_SIZE_FLASH 0x200000  // 2MB max per model
```

**Model Integration Pattern:**
```c
// Load model from flash
const tflite::Model* model = tflite::GetModel((const void*)MODEL_ADDR_FLASH);

// Setup interpreter
static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
tflite::MicroInterpreter* interpreter = &static_interpreter;

// Allocate tensors
interpreter->AllocateTensors();

// Run inference
interpreter->Invoke();

// Get results
TfLiteTensor* output = interpreter->output(0);
```

**Flashing Models:**
```bash
# Model flash parameters: [tflite_file] [flash_address] [offset]
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --file=output.img \
  --model="model_zoo/my_model/model.tflite 0xC00000 0x00000"
```

## Important Development Notes

### Platform-Specific Build Requirements

**Linux:**
- Requires: `make`, ARM GNU Toolchain v13.2+
- Install: `sudo apt install make`
- Download toolchain: `arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz`

**macOS:**
- Must use GNU `make`, not BSD `make`
- Check version: `make --version` (should show "GNU Make")
- Install if needed: `brew install make` (accessible as `gmake`)
- Add alias: `alias make='gmake'` in `.zshrc` or `.bash_profile`
- Use macOS-specific image generator: `we2_local_image_gen_macOS_arm64`

**Windows:**
- Requires: xpack build tools, GNU Toolchain (mingw-w64)
- Toolchain: `arm-gnu-toolchain-13.2.rel1-mingw-w64-i686-arm-none-eabi.zip`
- May need CH343 UART driver for device connection

### Ethos-U55 NPU and Vela Compiler

**NPU Configuration:**
- Grove Vision AI Module V2 uses Ethos-U55 with 64 MAC units (ethos-u55-64)
- Models must be compiled with Vela to run on NPU

**Vela Compilation:**
```bash
# Standard compilation for Ethos-U55 NPU
vela --accelerator-config ethos-u55-64 \
     --optimise Performance \
     model.tflite

# With separate I/O regions (fixes tensor buffer overlap issues)
vela --accelerator-config ethos-u55-64 \
     --optimise Performance \
     --cop-format COP2 \
     --separate-io-regions \
     model.tflite
```

**Important Vela Notes:**
- `--separate-io-regions`: Use separate memory regions for input/output tensors (requires `--cop-format COP2`)
- Models with dynamic tensors will fallback to CPU; ensure all tensor shapes are static
- Check Vela output for "CPU operators = 0" to confirm full NPU acceleration
- Vela models cannot run on PC TensorFlow interpreter (ethos-u custom op)

### CMSIS-NN Acceleration

To enable CMSIS-NN hardware acceleration:

1. Edit `EPII_CM55M_APP_S/makefile`:
   ```makefile
   LIB_CMSIS_NN_ENALBE = 1
   LIB_CMSIS_NN_VERSION = 7_0_0  # Use current version
   ```

2. Reference example: `allon_sensor_tflm_cmsis_nn`

3. Models do NOT need Vela optimization when using CMSIS-NN (CPU-only)

### Git Submodules

This repository uses Git submodules for `CMSIS-CV`:

```bash
# Initial clone with submodules
git clone --recursive https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2.git

# Or initialize submodules after clone
git submodule update --init --recursive
```

### Security Features

- **TrustZone-M:** ARM security partitioning for privileged operations
- **Secure Boot:** Image encryption and signing in `we2_image_gen_local/`
- Configurable via `TRUSTZONE`, `TRUSTZONE_TYPE`, `TRUSTZONE_FW_TYPE` in makefile

### Memory Constraints

- Maximum firmware image size: **1 MB** (enforced by bootloader)
- Models stored in flash from 0x200000+ (separate from firmware)
- Careful RAM management required for embedded constraints
- Use `memory_manage.c/.h` for custom allocators if needed

### Hardware Memory Map (Quick Reference)

**SRAM Layout (HX6538/WiseEye2):**
| Region | Base Address | Alias Address | Size |
|--------|--------------|---------------|------|
| SRAM0 | 0x24000000 | 0x34000000 | 1 MB |
| SRAM1 | 0x24100000 | 0x34100000 | 1 MB |
| SRAM2 | 0x26000000 | 0x36000000 | 384 KB |
| **Total** | - | - | **2.5+ MB** |

**Tensor Arena (for ML inference):**
- Location: SRAM1 @ 0x340E0000
- Size: **1.125 MB** (0x120000)
- Linker script: `Algo_NoTrustZone.ld` (recommended for vision apps)

**WDMA Frame Buffers (from SRAM0_ALIAS 0x34000000):**
| Buffer | Offset | Size | Purpose |
|--------|--------|------|---------|
| WDMA1 | 0x30000 | 391 KB | HW2x2/CDM output |
| WDMA2 | 0x8F400 | 299 KB | JPEG compressed output |
| WDMA3 | 0xDA400 | varies | HW5x5 YUV/RGB output |

**Image Format Sizes (640x480):**
| Format | Bytes/Pixel | Buffer Size |
|--------|-------------|-------------|
| YUV400 (grayscale) | 1 | 307 KB |
| YUV420 | 1.5 | 461 KB |
| YUV422 | 2 | 614 KB |
| RGB888 | 3 | 922 KB |

**JPEG Hardware Constraints:**
- Resolution: 16×16 to 1023×1023
- Width/Height: must be multiples of 16
- Quantization: `JPEG_ENC_QTABLE_4X` (standard) or `JPEG_ENC_QTABLE_10X` (high quality)

**Model Flash Addresses:**
```
0x00000000 - 0x00200000: Firmware (2 MB reserved, 1 MB limit)
0x00200000+: Model storage (4KB aligned)
  ├── 0x200000 - First model slot
  ├── 0x400000 - Second model slot
  └── 0xB7B000 - YOLOv8 default
```

**Key Files for Memory Configuration:**
- Device memory map: `EPII_CM55M_APP_S/device/inc/WE2_device_addr.h`
- Linker scripts: `EPII_CM55M_APP_S/linker_script/gcc/`
- SensorDP config: `EPII_CM55M_APP_S/library/sensordp/inc/sensor_dp_lib.h`

### Camera Sensor Support

Supported sensors (configure via `CIS_SUPPORT_INAPP_MODEL` in app makefile):
- `cis_ov5647` - Raspberry Pi Camera Module v1 (default)
- `cis_imx219` - Raspberry Pi Camera Module v2
- `cis_imx477` - Raspberry Pi HQ Camera
- `cis_imx708` - Raspberry Pi Camera Module 3
- `cis_hm0360` - Legacy Himax sensor

Reference apps with multi-sensor support:
- `allon_sensor_tflm`
- `allon_sensor_tflm_freertos`
- `tflm_fd_fm`

### Output Conventions

The firmware uses `xprintf()` (custom printf) for serial output to minimize binary size. Output is visible on UART at 921600 baud.

### Restoring Factory Firmware

To restore SenseCraft AI factory firmware:
```bash
# Flash the factory image
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --file=Seeed_SenseCraft_AI*.img

# Press reset button
# Access via: https://seeed-studio.github.io/SenseCraft-Web-Toolkit
```

## Common Development Workflows

### One-Click Build and Flash (Recommended)

**Use `build_and_flash.sh` for quick iteration:**
```bash
# Build, generate image, and flash in one command
./build_and_flash.sh

# Build and generate image only (no flash)
./build_and_flash.sh --no-flash

# Flash firmware only (no models)
./build_and_flash.sh --no-model
```

The script automatically:
1. Runs `gmake clean && gmake -j$(nproc)` with parallel compilation
2. Copies ELF to we2_image_gen_local and generates output.img
3. Flashes firmware + models via xmodem
4. Auto-detects USB serial port on macOS

**Model Configuration (in build_and_flash.sh):**
- SCRFD model: `model_zoo/tflm_face_recognition/scrfd_500m_kps_int8_vela.tflite` @ 0x200000
- GhostFaceNet model: `model_zoo/tflm_face_recognition/ghostfacenet_0.5_112_int8_vela.tflite` @ 0x400000

### Workflow: Quick Test of Existing Example (Manual)
```bash
# 1. Select app
cd EPII_CM55M_APP_S
# Edit makefile: APP_TYPE = tflm_yolov8_od

# 2. Build
make clean && make

# 3. Generate image
cd ../we2_image_gen_local
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json

# 4. Flash
cd ..
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="model_zoo/tflm_yolov8_od/yolov8n_od_192_delete_transpose_0xB7B000.tflite 0xB7B000 0x00000"

# 5. Press reset button and monitor via serial console
```

### Workflow: Switch Camera Sensor
```bash
# Edit app makefile (e.g., EPII_CM55M_APP_S/app/scenario_app/tflm_yolov8_od/tflm_yolov8_od.mk)
# Change: CIS_SUPPORT_INAPP_MODEL = cis_ov5647
# To:     CIS_SUPPORT_INAPP_MODEL = cis_imx219

# Rebuild
cd EPII_CM55M_APP_S
make clean && make
# Then generate image and flash as usual
```

### Workflow: Enable FreeRTOS
```bash
# Edit EPII_CM55M_APP_S/makefile
# Set: OS_SEL = freertos

# Or use FreeRTOS example app
# Set: APP_TYPE = allon_sensor_tflm_freertos

# Rebuild
make clean && make
```

### Workflow: Flash Multiple Models
```bash
# Define multiple model addresses in common_config.h
# Then flash all at once
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --file=output.img \
  --model="model_zoo/yolov8_od.tflite 0xB7B000 0x0" \
  --model="model_zoo/face_detect.tflite 0xD00000 0x0"
```

## Project-Specific Conventions

- Firmware entry point: `app_main()` in each scenario app
- Serial output uses `xprintf()`, not standard `printf()`
- Model addresses must be 4KB aligned (0x1000 byte boundaries)
- App makefiles use `override SCENARIO_APP_SUPPORT_LIST := $(APP_TYPE)` pattern
- Linker scripts: `.sct` for ARM compiler, `.ld` for GNU compiler
- Build artifacts in `obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/`
- Final bootable image always: `we2_image_gen_local/output_case1_sec_wlcsp/output.img`
- to memorize