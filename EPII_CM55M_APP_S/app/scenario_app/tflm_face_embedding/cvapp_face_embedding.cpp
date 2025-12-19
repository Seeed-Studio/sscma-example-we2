/*
 * cvapp_face_embedding.cpp
 *
 * Face Embedding Implementation using SCRFD + MobileFaceNet
 *
 * Pipeline:
 *   1. Face Detection: SCRFD_500M_KPS (160x160) with 5-point landmarks
 *   2. Face Alignment: Similarity transform using eye positions
 *   3. Face Embedding: MobileFaceNet (112x112 -> 128D)
 *
 * Features:
 *   - CMSIS-NN hardware acceleration
 *   - INT8 quantization for both models
 *   - Face alignment for improved recognition accuracy
 *
 * Created for Grove Vision AI Module V2
 */

#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp_face_embedding.h"
#include "cisdp_sensor.h"
#include "WE2_core.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "img_proc_helium.h"

#include "xprintf.h"
#include "spi_master_protocol.h"
#include "cisdp_cfg.h"
#include "memory_manage.h"
#include "common_config.h"
#include "face_embedding_protocol.h"
#include "scrfd_postprocessing.h"
#include "face_alignment.h"
#include "send_result.h"

#ifdef TRUSTZONE_SEC
#define U55_BASE    BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE    BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE    BASE_ADDR_APB_U55_CTRL
#endif
#endif

#define TOTAL_STEP_TICK 1
#define CPU_CLK (0xffffff + 1)

/* DEBUG: Output all detected faces without running embedding
 * Set to 1 to debug face detection, 0 for normal operation */
#define DEBUG_DETECTION_ONLY 0

/* Debug verbosity levels:
 * 0 = Minimal (only timing summary)
 * 1 = Normal (step timing + key info)
 * 2 = Verbose (all debug info)
 */
#define DEBUG_VERBOSE 0

#if DEBUG_VERBOSE >= 2
#define DBG_VERBOSE(fmt, ...) xprintf(fmt, ##__VA_ARGS__)
#else
#define DBG_VERBOSE(fmt, ...) ((void)0)
#endif

#if DEBUG_VERBOSE >= 1
#define DBG_INFO(fmt, ...) xprintf(fmt, ##__VA_ARGS__)
#else
#define DBG_INFO(fmt, ...) ((void)0)
#endif

/* Helper macro for step timing */
#define TICK_TO_MS(ticks) ((ticks) / 24000)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

using namespace std;

namespace {

/*
 * Memory allocation for dual-model inference
 *
 * Strategy: SEPARATE tensor arenas for each model
 *
 * Memory usage (Vela 3.9.0):
 *   SCRFD arena:        220 KB (Vela: 201 KB)
 *   MobileFaceNet arena: 700 KB (Vela: 600 KB)
 *   Total:              920 KB
 */
constexpr int scrfd_arena_size = SCRFD_ARENA_SIZE;
constexpr int mobilefacenet_arena_size = MOBILEFACENET_ARENA_SIZE;
constexpr int mobilefacenet_arena_head_offset = 128;  /* Safety offset to avoid arena head conflicts */
constexpr int fd_resize_image_size = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL;
constexpr int aligned_face_buffer_size = ALIGNED_FACE_BUFFER_SIZE;

static uint32_t scrfd_tensor_arena = 0;
static uint32_t mobilefacenet_tensor_arena = 0;
static uint32_t fd_resized_img = 0;
static uint32_t aligned_face_img = 0;

/* Scale factors for coordinate mapping (model space -> original image space) */
static float scale_w = 1.0f;
static float scale_h = 1.0f;

/* Letterbox parameters for SCRFD preprocessing */
static float letterbox_scale = 1.0f;  /* Uniform scale factor */
static int letterbox_pad_x = 0;       /* X padding (left side) */
static int letterbox_pad_y = 0;       /* Y padding (top side) */

/* Ethos-U NPU driver */
struct ethosu_driver ethosu_drv;

/* Face Detection (SCRFD) interpreter */
tflite::MicroInterpreter *fd_int_ptr = nullptr;
TfLiteTensor *fd_input = nullptr;

/* SCRFD outputs: 9 tensors (3 strides x (score + bbox + kps)) */
TfLiteTensor *fd_score_tensors[SCRFD_NUM_STRIDES];
TfLiteTensor *fd_bbox_tensors[SCRFD_NUM_STRIDES];
TfLiteTensor *fd_kps_tensors[SCRFD_NUM_STRIDES];

/* Face Embedding (MobileFaceNet) interpreter */
tflite::MicroInterpreter *emb_int_ptr = nullptr;
TfLiteTensor *emb_input = nullptr;
TfLiteTensor *emb_output = nullptr;

/* SCRFD network configuration */
scrfd_network scrfd_net;

static uint32_t g_face_emb_init = 0;

/*
 * Operator resolver
 *
 * Vela compiles most ops to NPU:
 *   - SCRFD: 100% NPU
 *   - MobileFaceNet: NPU + L2_NORMALIZATION (CPU fallback)
 *
 * L2_NORMALIZATION is not supported by Ethos-U, requires CPU fallback.
 */
static tflite::MicroMutableOpResolver<2> op_resolver;  /* EthosU + L2Norm */

}  // namespace

/* NPU IRQ handler */
static void _arm_npu_irq_handler(void)
{
    ethosu_irq_handler(&ethosu_drv);
}

/* Initialize NPU IRQ */
static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
    NVIC_EnableIRQ(ethosu_irqnum);
}

/* Initialize NPU */
static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;
    _arm_npu_irq_init();

#if TFLM2209_U55TAG2205
    const void *ethosu_base_address = (void *)(U55_BASE);
#else
    void *const ethosu_base_address = (void *)(U55_BASE);
#endif

    if (0 != (err = ethosu_init(&ethosu_drv, ethosu_base_address, NULL, 0,
                                security_enable, privilege_enable))) {
        xprintf("ERROR: Failed to initialize Ethos-U device\n");
        return err;
    }

    xprintf("Ethos-U55 NPU initialized\n");
    return 0;
}

/* ========== PUBLIC API ========== */

int cv_face_embedding_init(bool security_enable, bool privilege_enable,
                           uint32_t fd_model_addr, uint32_t embedding_model_addr)
{
    xprintf("\n===== Face Embedding Initialization =====\n");
    xprintf("Models: SCRFD_500M_KPS + MobileFaceNet-128D\n\n");

    /*
     * Memory allocation - SEPARATE arenas for each model
     */
    scrfd_tensor_arena = mm_reserve_align(scrfd_arena_size, 0x20);
    /* Reserve 4KB safety gap to prevent Cache/MPU conflicts at arena boundary */
    mm_reserve_align(4096, 0x1000);  /* 4KB gap, 4KB aligned */

    /*
     * ARENA HEAD OFFSET FIX:
     * Allocate extra 128 bytes at the start to avoid placing tensors at offset 0.
     */
    uint32_t mobilefacenet_arena_raw = mm_reserve_align(mobilefacenet_arena_size + mobilefacenet_arena_head_offset, 0x20);
    mobilefacenet_tensor_arena = mobilefacenet_arena_raw + mobilefacenet_arena_head_offset;  /* Skip first 128 bytes */

    fd_resized_img = mm_reserve_align(fd_resize_image_size, 0x20);
    aligned_face_img = mm_reserve_align(aligned_face_buffer_size, 0x20);

    xprintf("Memory: SCRFD %dKB, MFN %dKB\n",
            scrfd_arena_size / 1024, mobilefacenet_arena_size / 1024);

    /* Initialize NPU */
    if (_arm_npu_init(security_enable, privilege_enable) != 0) {
        xprintf("ERROR: NPU initialization failed\n");
        return -1;
    }

    /* Load models from flash */
    xprintf("\nLoading models...\n");
    static const tflite::Model *fd_model = tflite::GetModel((const void *)fd_model_addr);
    static const tflite::Model *emb_model = tflite::GetModel((const void *)embedding_model_addr);

    /* Model Integrity Check - silent on success */
    uint8_t* scrfd_bytes = (uint8_t*)fd_model_addr;
    if (scrfd_bytes[4] != 'T' || scrfd_bytes[5] != 'F' ||
        scrfd_bytes[6] != 'L' || scrfd_bytes[7] != '3') {
        xprintf("ERROR: SCRFD model invalid @ 0x%08X\n", fd_model_addr);
        return -10;
    }

    uint8_t* emb_bytes = (uint8_t*)embedding_model_addr;
    if (emb_bytes[4] != 'T' || emb_bytes[5] != 'F' ||
        emb_bytes[6] != 'L' || emb_bytes[7] != '3') {
        xprintf("ERROR: MobileFaceNet model invalid @ 0x%08X\n", embedding_model_addr);
        return -11;
    }

    /* Verify model schema versions */
    if (fd_model->version() != TFLITE_SCHEMA_VERSION) {
        xprintf("ERROR: SCRFD schema mismatch\n");
        return -2;
    }
    if (emb_model->version() != TFLITE_SCHEMA_VERSION) {
        xprintf("ERROR: MobileFaceNet schema mismatch\n");
        return -3;
    }
    xprintf("  SCRFD @ 0x%08X, MFN @ 0x%08X\n", fd_model_addr, embedding_model_addr);

    /* Register operators */
    if (g_face_emb_init == 0) {
        if (kTfLiteOk != op_resolver.AddEthosU()) {
            xprintf("ERROR: Failed to add Ethos-U\n");
            return -4;
        }
        if (kTfLiteOk != op_resolver.AddL2Normalization()) {
            xprintf("ERROR: Failed to add L2Normalization\n");
            return -5;
        }
    }

    /* Create interpreters */

#if TFLM2209_U55TAG2205
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::MicroInterpreter fd_static_interpreter(
        fd_model, op_resolver,
        (uint8_t *)scrfd_tensor_arena, scrfd_arena_size,
        &micro_error_reporter);
    static tflite::MicroInterpreter emb_static_interpreter(
        emb_model, op_resolver,
        (uint8_t *)mobilefacenet_tensor_arena, mobilefacenet_arena_size,
        &micro_error_reporter);
#else
    static tflite::MicroInterpreter fd_static_interpreter(
        fd_model, op_resolver,
        (uint8_t *)scrfd_tensor_arena, scrfd_arena_size);
    static tflite::MicroInterpreter emb_static_interpreter(
        emb_model, op_resolver,
        (uint8_t *)mobilefacenet_tensor_arena, mobilefacenet_arena_size);
#endif

    /* Allocate tensors */
    if (fd_static_interpreter.AllocateTensors() != kTfLiteOk) {
        xprintf("ERROR: SCRFD tensor allocation failed\n");
        return -26;
    }
    if (emb_static_interpreter.AllocateTensors() != kTfLiteOk) {
        xprintf("ERROR: MobileFaceNet tensor allocation failed\n");
        return -27;
    }
    xprintf("Arena: SCRFD %dKB, MFN %dKB\n",
            fd_static_interpreter.arena_used_bytes() / 1024,
            emb_static_interpreter.arena_used_bytes() / 1024);

    /* Setup SCRFD interpreter */
    fd_int_ptr = &fd_static_interpreter;
    fd_input = fd_static_interpreter.input(0);

    xprintf("SCRFD: %dx%d %s, %d outputs\n",
            fd_input->dims->data[1], fd_input->dims->data[2],
            fd_input->type == kTfLiteInt8 ? "INT8" : "UINT8",
            fd_static_interpreter.outputs_size());

    /* Get SCRFD output tensors */
    int num_outputs = fd_static_interpreter.outputs_size();

    /* Initialize all to nullptr first */
    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        fd_score_tensors[i] = nullptr;
        fd_bbox_tensors[i] = nullptr;
        fd_kps_tensors[i] = nullptr;
    }

    /* Map outputs by shape analysis - handle both 2D (Vela) and 4D (original) formats */
    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* t = fd_static_interpreter.output(i);

        int stride_idx = -1;
        int tensor_type = -1;  /* 0=score, 1=bbox, 2=kps */

        if (t->dims->size == 2) {
            /* Vela 2D format: [num_elements, channels] */
            int num_elements = t->dims->data[0];
            int channels = t->dims->data[1];

            /* Determine stride from number of elements */
            if (num_elements == 800) stride_idx = 0;       /* 20*20*2 = stride 8 */
            else if (num_elements == 200) stride_idx = 1;  /* 10*10*2 = stride 16 */
            else if (num_elements == 50) stride_idx = 2;   /* 5*5*2 = stride 32 */

            /* Determine tensor type from channels */
            if (channels == 1) tensor_type = 0;       /* score */
            else if (channels == 4) tensor_type = 1;  /* bbox */
            else if (channels == 10) tensor_type = 2; /* kps */

        } else if (t->dims->size == 4) {
            /* Original 4D format: [1, H, W, C] */
            int h = t->dims->data[1];
            int w = t->dims->data[2];
            int c = t->dims->data[3];

            if (h == 20 && w == 20) stride_idx = 0;
            else if (h == 10 && w == 10) stride_idx = 1;
            else if (h == 5 && w == 5) stride_idx = 2;

            if (c == 2) tensor_type = 0;
            else if (c == 8) tensor_type = 1;
            else if (c == 20) tensor_type = 2;
        }

        if (stride_idx < 0 || tensor_type < 0) continue;

        /* Assign tensor to appropriate slot */
        if (tensor_type == 0) {
            fd_score_tensors[stride_idx] = t;
        } else if (tensor_type == 1) {
            fd_bbox_tensors[stride_idx] = t;
        } else if (tensor_type == 2) {
            fd_kps_tensors[stride_idx] = t;
        }
    }

    /* Verify we have at least one complete branch */
    bool has_valid_branch = false;
    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        if (fd_score_tensors[i] && fd_bbox_tensors[i] && fd_kps_tensors[i]) {
            has_valid_branch = true;
        }
    }

    if (!has_valid_branch) {
        xprintf("ERROR: No complete SCRFD branch found\n");
        return -28;
    }

    /* Initialize SCRFD post-processing */
    scrfd_net = scrfd_init(
        fd_score_tensors, fd_bbox_tensors, fd_kps_tensors,
        FD_INPUT_TENSOR_WIDTH, FD_INPUT_TENSOR_HEIGHT,
        FACE_CONF_THRESHOLD, FACE_NMS_THRESHOLD);

    /* Setup MobileFaceNet interpreter */
    emb_int_ptr = &emb_static_interpreter;
    emb_input = emb_static_interpreter.input(0);
    emb_output = emb_static_interpreter.output(0);

    xprintf("MFN: %dx%d -> %dD embedding\n",
            emb_input->dims->data[1], emb_input->dims->data[2],
            emb_output->dims->data[1]);

    g_face_emb_init = 1;
    xprintf("Init complete\n");

    return 0;
}

int cv_face_embedding_run(struct_algoResult *alg_result, face_embedding_msg_t *embedding_msg)
{
    static int frame_count = 0;
    frame_count++;
    DBG_VERBOSE("\n[Frame %d] cv_face_embedding_run started\n", frame_count);

#if TOTAL_STEP_TICK
    uint32_t systick_1, systick_2;
    uint32_t loop_cnt_1, loop_cnt_2;
    SystemGetTick(&systick_1, &loop_cnt_1);

    /* Step timing variables */
    uint32_t tick_preprocess, tick_scrfd, tick_postproc, tick_align, tick_mobilefacenet;
    uint32_t systick_step, loop_cnt_step;
#endif

    TfLiteStatus invoke_status = kTfLiteOk;

    uint32_t img_w = app_get_raw_width();
    uint32_t img_h = app_get_raw_height();
    uint32_t raw_addr = app_get_raw_addr();
    DBG_VERBOSE("  Image: %dx%d @ 0x%08X\n", img_w, img_h, raw_addr);


    /* CRITICAL: Invalidate D-Cache before reading DMA-written raw buffer
     * The sensor DMA writes directly to memory, bypassing CPU cache.
     * Without invalidation, CPU may read stale cached data instead of
     * actual image data, causing incorrect inference results.
     */
    uint32_t raw_size = img_w * img_h * 3;  // RGB format
    SCB_InvalidateDCache_by_Addr((uint32_t*)raw_addr, raw_size);

    /* Initialize results */
    alg_result->num_tracked_human_targets = 0;
    memset(embedding_msg, 0, sizeof(face_embedding_msg_t));

    /* ===== STEP 1: Preprocess for SCRFD with Direct Resize ===== */
    /* NOTE: SCRFD was trained with direct resize (no letterbox padding).
     * Using letterbox causes wrong detection coordinates because the model
     * never saw black padding bands during training.
     * Direct resize: 320x240 -> 160x160 (with ~33% vertical stretch)
     */
    DBG_VERBOSE("  Step 1: Preprocessing for SCRFD (direct resize)...\n");

    /* Direct resize to full tensor size (no letterbox) */
    int new_w = FD_INPUT_TENSOR_WIDTH;   // 160
    int new_h = FD_INPUT_TENSOR_HEIGHT;  // 160

    /* No padding needed for direct resize */
    letterbox_pad_x = 0;
    letterbox_pad_y = 0;
    letterbox_scale = 1.0f;  // Not used, but keep for compatibility

    DBG_VERBOSE("    Direct Resize: %dx%d -> %dx%d (no letterbox)\n",
                img_w, img_h, new_w, new_h);


    /*
     * Direct resize scales: convert from model space (160x160) to original image space
     * For 320x240 -> 160x160:
     *   scale_w = 320/160 = 2.0 (model x * 2.0 = original x)
     *   scale_h = 240/160 = 1.5 (model y * 1.5 = original y)
     */
    float w_scale = (float)img_w / new_w;
    float h_scale = (float)img_h / new_h;


    /* Store for coordinate mapping in post-processing */
    scale_w = w_scale;
    scale_h = h_scale;

    /* First, resize image maintaining aspect ratio */
    hx_lib_image_resize_BGR8U3C_to_RGB24_helium(
        (uint8_t *)raw_addr,
        (uint8_t *)fd_resized_img,
        img_w, img_h, FD_INPUT_TENSOR_CHANNEL,
        new_w, new_h,
        w_scale, h_scale);
        

    /* Fill input tensor with resized image (direct copy, no letterbox padding) */
    if (fd_input->type == kTfLiteInt8) {
        int8_t *dst = fd_input->data.int8;
        uint8_t *src = (uint8_t *)fd_resized_img;
        int32_t zp = fd_input->params.zero_point;
        int total = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL;

        /* Direct copy with quantization (uint8 -> int8 with zero point) */
        for (int i = 0; i < total; i++) {
            int32_t val = (int32_t)src[i] + zp;
            if (val < -128) val = -128;
            if (val > 127) val = 127;
            dst[i] = (int8_t)val;
        }
    } else {
        uint8_t *dst = fd_input->data.uint8;
        uint8_t *src = (uint8_t *)fd_resized_img;
        int total = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL;

        /* Direct copy (uint8 -> uint8) */
        memcpy(dst, src, total);
    }


#if TOTAL_STEP_TICK
    SystemGetTick(&systick_step, &loop_cnt_step);
    tick_preprocess = (loop_cnt_step - loop_cnt_1) * CPU_CLK + (systick_1 - systick_step);
#endif

    /* ===== STEP 2: Run SCRFD Face Detection ===== */
    DBG_VERBOSE("  Step 2: Running SCRFD inference...\n");

    /* D-Cache Coherency Fix - CRITICAL for NPU */
    if (fd_input->data.data) {
        SCB_CleanDCache_by_Addr((uint32_t*)fd_input->data.data, fd_input->bytes);
    }
    
    invoke_status = fd_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("ERROR: SCRFD invoke failed\n");
        return -1;
    }

    /* CRITICAL: Invalidate D-Cache for output tensors after NPU inference
     * NPU writes directly to memory, bypassing CPU cache. Without invalidation,
     * CPU may read stale cached data causing false detections (ghost faces).
     */
    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        if (fd_score_tensors[i] && fd_score_tensors[i]->data.data) {
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_score_tensors[i]->data.data,
                                         fd_score_tensors[i]->bytes);
        }
        if (fd_bbox_tensors[i] && fd_bbox_tensors[i]->data.data) {
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_bbox_tensors[i]->data.data,
                                         fd_bbox_tensors[i]->bytes);
        }
        if (fd_kps_tensors[i] && fd_kps_tensors[i]->data.data) {
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_kps_tensors[i]->data.data,
                                         fd_kps_tensors[i]->bytes);
        }
    }

#if TOTAL_STEP_TICK
    uint32_t systick_scrfd, loop_cnt_scrfd;
    SystemGetTick(&systick_scrfd, &loop_cnt_scrfd);
    tick_scrfd = (loop_cnt_scrfd - loop_cnt_step) * CPU_CLK + (systick_step - systick_scrfd);
#endif

    /* ===== STEP 3: SCRFD Post-processing ===== */
    /* Set coordinate mapping parameters
     * Formula: orig = (model - pad) * scale
     * We use the explicit geometric scales calculated during resize.
     * scale_x = img_w / new_w (stored in scale_w)
     * scale_y = img_h / new_h (stored in scale_h)
     */
    scrfd_net.scale_x = scale_w;
    scrfd_net.scale_y = scale_h;
    scrfd_net.pad_x = letterbox_pad_x;
    scrfd_net.pad_y = letterbox_pad_y;

    int num_faces = 0;
    std::forward_list<scrfd_face> faces = scrfd_detect(&scrfd_net, img_w, img_h, &num_faces);

    if (num_faces == 0) {
        /* No face detected - still send video frame */
        uint32_t jpeg_size = 0;
        uint32_t jpeg_addr = 0;
        cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);

        el_img_t jpeg_img;
        jpeg_img.data = (uint8_t*)jpeg_addr;
        jpeg_img.size = jpeg_size;
        jpeg_img.width = img_w;
        jpeg_img.height = img_h;
        jpeg_img.format = EL_PIXEL_FORMAT_JPEG;
        jpeg_img.rotate = EL_PIXEL_ROTATE_0;


        send_face_recognition_json(&jpeg_img, nullptr, nullptr, 0, nullptr, 0.0f, nullptr, 0, 0);
        return 0;
    }

    DBG_INFO("[Frame] Detected %d face(s)\n", num_faces);


#if DEBUG_DETECTION_ONLY
    /* DEBUG MODE: Send all detected faces without running embedding */
    {
        uint32_t jpeg_size = 0;
        uint32_t jpeg_addr = 0;
        cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);

        /* Build JSON with multiple faces */
        std::string json_str = "{\"type\":1,\"name\":\"FACE_RESULT\",\"code\":0,\"data\":{";

        /* Add image */
        if (jpeg_addr && jpeg_size > 0) {
            el_img_t jpeg_img;
            jpeg_img.data = (uint8_t*)jpeg_addr;
            jpeg_img.size = jpeg_size;
            jpeg_img.width = img_w;
            jpeg_img.height = img_h;
            jpeg_img.format = EL_PIXEL_FORMAT_JPEG;
            json_str += img_2_json_str(&jpeg_img);
            json_str += ", ";
            json_str += img_res_2_json_str(&jpeg_img);
        } else {
            json_str += "\"image\": \"\", \"resolution\": [0, 0]";
        }

        /* Add all faces */
        json_str += ", \"faces\": [";
        bool first_face = true;
        for (auto& f : faces) {
            if (f.score <= 0) continue;

            if (!first_face) json_str += ", ";
            first_face = false;

            char face_buf[256];
            snprintf(face_buf, sizeof(face_buf),
                     "{\"bbox\": [%d, %d, %d, %d], \"confidence\": %d.%02d, \"landmarks\": [",
                     (int)f.bbox.x, (int)f.bbox.y, (int)f.bbox.w, (int)f.bbox.h,
                     (int)(f.score * 100) / 100, (int)(f.score * 100) % 100);
            json_str += face_buf;

            /* Add 5 landmarks */
            for (int k = 0; k < 5; k++) {
                if (k > 0) json_str += ", ";
                char lm_buf[32];
                snprintf(lm_buf, sizeof(lm_buf), "[%d, %d]",
                         (int)f.landmarks[k].x, (int)f.landmarks[k].y);
                json_str += lm_buf;
            }
            json_str += "]}";
        }
        json_str += "]}}";

        /* Send JSON */
        json_str += "\r\n";
        send_bytes(json_str.c_str(), json_str.size());

        xprintf("[DEBUG] Sent %d faces (detection only mode)\n", num_faces);
        scrfd_free_dets(faces);
        return 0;
    }
#endif

    /* Get best (highest score) face with valid size */
    scrfd_face *best_face = scrfd_get_best_face(faces, MIN_FACE_SIZE);
    if (best_face == nullptr || best_face->score <= 0) {
        uint32_t jpeg_size = 0;
        uint32_t jpeg_addr = 0;
        cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);

        el_img_t jpeg_img;
        jpeg_img.data = (uint8_t*)jpeg_addr;
        jpeg_img.size = jpeg_size;
        jpeg_img.width = img_w;
        jpeg_img.height = img_h;
        jpeg_img.format = EL_PIXEL_FORMAT_JPEG;
        jpeg_img.rotate = EL_PIXEL_ROTATE_0;

        send_face_recognition_json(&jpeg_img, nullptr, nullptr, 0, nullptr, 0.0f, nullptr, 0, 0);
        scrfd_free_dets(faces);
        return 0;
    }

    /* Store detection result */
    alg_result->num_tracked_human_targets = 1;
    alg_result->ht[0].upper_body_score = (uint32_t)(best_face->score * 100);
    alg_result->ht[0].upper_body_bbox.x = (uint32_t)best_face->bbox.x;
    alg_result->ht[0].upper_body_bbox.y = (uint32_t)best_face->bbox.y;
    alg_result->ht[0].upper_body_bbox.width = (uint32_t)best_face->bbox.w;
    alg_result->ht[0].upper_body_bbox.height = (uint32_t)best_face->bbox.h;

    /* Quality check: minimum face size */
    if (best_face->bbox.w < MIN_FACE_SIZE || best_face->bbox.h < MIN_FACE_SIZE) {
        DBG_INFO("  Face too small: %.0fx%.0f\n", best_face->bbox.w, best_face->bbox.h);
        alg_result->num_tracked_human_targets = 0;

        uint32_t jpeg_size = 0;
        uint32_t jpeg_addr = 0;
        cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);

        el_img_t jpeg_img;
        jpeg_img.data = (uint8_t*)jpeg_addr;
        jpeg_img.size = jpeg_size;
        jpeg_img.width = img_w;
        jpeg_img.height = img_h;
        jpeg_img.format = EL_PIXEL_FORMAT_JPEG;
        jpeg_img.rotate = EL_PIXEL_ROTATE_0;

        el_box_t face_box;
        face_box.x = (uint16_t)best_face->bbox.x;
        face_box.y = (uint16_t)best_face->bbox.y;
        face_box.w = (uint16_t)best_face->bbox.w;
        face_box.h = (uint16_t)best_face->bbox.h;
        face_box.score = (uint8_t)(best_face->score * 100);
        face_box.target = 0;

        send_face_recognition_json(&jpeg_img, &face_box, nullptr, 0, nullptr, best_face->score, nullptr, 0, 0);
        scrfd_free_dets(faces);
        return 0;
    }

    /* Validate landmarks */
    if (!scrfd_validate_face(best_face, MIN_FACE_SIZE, img_w, img_h)) {
        DBG_INFO("  Face validation failed\n");
        alg_result->num_tracked_human_targets = 0;

        uint32_t jpeg_size = 0;
        uint32_t jpeg_addr = 0;
        cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);

        el_img_t jpeg_img;
        jpeg_img.data = (uint8_t*)jpeg_addr;
        jpeg_img.size = jpeg_size;
        jpeg_img.width = img_w;
        jpeg_img.height = img_h;
        jpeg_img.format = EL_PIXEL_FORMAT_JPEG;
        jpeg_img.rotate = EL_PIXEL_ROTATE_0;

        el_box_t face_box;
        face_box.x = (uint16_t)best_face->bbox.x;
        face_box.y = (uint16_t)best_face->bbox.y;
        face_box.w = (uint16_t)best_face->bbox.w;
        face_box.h = (uint16_t)best_face->bbox.h;
        face_box.score = (uint8_t)(best_face->score * 100);
        face_box.target = 0;

        send_face_recognition_json(&jpeg_img, &face_box, nullptr, 0, nullptr, best_face->score, nullptr, 0, 0);
        scrfd_free_dets(faces);
        return 0;
    }

#if TOTAL_STEP_TICK
    uint32_t systick_post, loop_cnt_post;
    SystemGetTick(&systick_post, &loop_cnt_post);
    tick_postproc = (loop_cnt_post - loop_cnt_scrfd) * CPU_CLK + (systick_scrfd - systick_post);
#endif

    DBG_INFO("  Face: %dx%d score=%d%%\n",
            (int)best_face->bbox.w, (int)best_face->bbox.h, (int)(best_face->score * 100));

#if ENABLE_FACE_ALIGNMENT
    /* ===== STEP 4: Face Alignment ===== */
    affine_transform_t align_transform;
    compute_face_alignment(best_face->landmarks, &align_transform);

    apply_face_alignment(
        (uint8_t *)raw_addr, img_w, img_h,
        (uint8_t *)aligned_face_img,
        &align_transform);

    /* Copy to embedding input tensor - handle INT8 vs UINT8 */
    if (emb_input->type == kTfLiteInt8) {
        uint8_t *src = (uint8_t *)aligned_face_img;
        int8_t *dst = emb_input->data.int8;
        for (int i = 0; i < aligned_face_buffer_size; i++) {
            dst[i] = (int8_t)((int)src[i] - 128);
        }
        DBG_VERBOSE("  Embedding input ready (INT8, aligned)\n");
    } else {
        memcpy(emb_input->data.uint8, (uint8_t *)aligned_face_img, aligned_face_buffer_size);
        DBG_VERBOSE("  Embedding input ready (UINT8, aligned)\n");
    }
#else
    /* Fallback: simple resize of full frame to 112x112 */
    DBG_VERBOSE("  Step 4: Resizing for embedding (no alignment)...\n");

    float emb_w_scale = (float)(img_w - 1) / (EMBEDDING_INPUT_WIDTH - 1);
    float emb_h_scale = (float)(img_h - 1) / (EMBEDDING_INPUT_HEIGHT - 1);

    hx_lib_image_resize_BGR8U3C_to_RGB24_helium(
        (uint8_t *)raw_addr, (uint8_t *)aligned_face_img,
        img_w, img_h, EMBEDDING_INPUT_CHANNEL,
        EMBEDDING_INPUT_WIDTH, EMBEDDING_INPUT_HEIGHT,
        emb_w_scale, emb_h_scale);

    /* Copy to embedding input tensor - handle INT8 vs UINT8 */
    if (emb_input->type == kTfLiteInt8) {
        uint8_t *src = (uint8_t *)aligned_face_img;
        int8_t *dst = emb_input->data.int8;
        for (int i = 0; i < aligned_face_buffer_size; i++) {
            dst[i] = (int8_t)((int)src[i] - 128);
        }
        DBG_VERBOSE("  Embedding input ready (INT8)\n");
    } else {
        memcpy(emb_input->data.uint8, (uint8_t *)aligned_face_img, aligned_face_buffer_size);
        DBG_VERBOSE("  Embedding input ready (UINT8)\n");
    }
#endif

#if TOTAL_STEP_TICK
    uint32_t systick_align, loop_cnt_align;
    SystemGetTick(&systick_align, &loop_cnt_align);
    tick_align = (loop_cnt_align - loop_cnt_post) * CPU_CLK + (systick_post - systick_align);
#endif

    /* ===== STEP 5: Run MobileFaceNet Embedding ===== */
    DBG_VERBOSE("  Step 5: Running MobileFaceNet...\n");

    /* D-Cache Coherency Fix */
    SCB_CleanDCache_by_Addr((uint32_t*)emb_input->data.data, emb_input->bytes);
    SCB_CleanDCache_by_Addr((uint32_t*)mobilefacenet_tensor_arena, mobilefacenet_arena_size);
    __DSB();
    __ISB();

    /* Run embedding inference */
    invoke_status = emb_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("ERROR: MobileFaceNet invoke failed\n");
        scrfd_free_dets(faces);
        return -2;
    }

#if TOTAL_STEP_TICK
    uint32_t systick_mfn, loop_cnt_mfn;
    SystemGetTick(&systick_mfn, &loop_cnt_mfn);
    tick_mobilefacenet = (loop_cnt_mfn - loop_cnt_align) * CPU_CLK + (systick_align - systick_mfn);
#endif

    /* ===== STEP 6: Extract Embedding ===== */
    int emb_dim = MIN(emb_output->dims->data[1], EMBEDDING_OUTPUT_DIM);
    DBG_VERBOSE("  Extracting embedding: type=%d, dim=%d\n", emb_output->type, emb_dim);

    if (emb_output->type == kTfLiteFloat32) {
        memcpy(embedding_msg->embedding, emb_output->data.f, emb_dim * sizeof(float));
    } else if (emb_output->type == kTfLiteInt8) {
        float scale = emb_output->params.scale;
        int32_t zero_point = emb_output->params.zero_point;
        int8_t *quant_data = emb_output->data.int8;
        for (int i = 0; i < emb_dim; i++) {
            embedding_msg->embedding[i] = (quant_data[i] - zero_point) * scale;
        }
    } else if (emb_output->type == kTfLiteUInt8) {
        float scale = emb_output->params.scale;
        int32_t zero_point = emb_output->params.zero_point;
        uint8_t *quant_data = emb_output->data.uint8;
        for (int i = 0; i < emb_dim; i++) {
            embedding_msg->embedding[i] = (quant_data[i] - zero_point) * scale;
        }
    }

    /* L2 Normalization */
    normalize_embedding(embedding_msg->embedding, emb_dim);

    /* ===== STEP 7: Fill Message Metadata ===== */
    embedding_msg->face_id = 0;

    uint32_t systick, loop_cnt;
    SystemGetTick(&systick, &loop_cnt);
    embedding_msg->timestamp = loop_cnt * CPU_CLK + systick;

    embedding_msg->confidence = best_face->score;
    embedding_msg->quality = estimate_face_quality(best_face->landmarks);

    embedding_msg->bbox.x = (uint16_t)best_face->bbox.x;
    embedding_msg->bbox.y = (uint16_t)best_face->bbox.y;
    embedding_msg->bbox.width = (uint16_t)best_face->bbox.w;
    embedding_msg->bbox.height = (uint16_t)best_face->bbox.h;

    estimate_face_pose(
        best_face->landmarks,
        &embedding_msg->pose.yaw,
        &embedding_msg->pose.pitch,
        &embedding_msg->pose.roll);

    /* Copy landmarks */
    for (int i = 0; i < 5; i++) {
        embedding_msg->landmarks[i].x = best_face->landmarks[i].x;
        embedding_msg->landmarks[i].y = best_face->landmarks[i].y;
    }

    /* Pack message and calculate CRC */
    pack_face_embedding_msg(
        embedding_msg,
        embedding_msg->embedding,
        &embedding_msg->bbox,
        &embedding_msg->pose,
        embedding_msg->confidence,
        embedding_msg->timestamp,
        embedding_msg->face_id);
    DBG_VERBOSE("  Embedding extracted (%dD), quality=%.2f\n", emb_dim, embedding_msg->quality);

    /* Send JSON result for Web Debug Tool */
    {
        uint32_t jpeg_size = 0;
        uint32_t jpeg_addr = 0;
        cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);

        el_img_t jpeg_img;
        jpeg_img.data = (uint8_t*)jpeg_addr;
        jpeg_img.size = jpeg_size;
        jpeg_img.width = img_w;
        jpeg_img.height = img_h;
        jpeg_img.format = EL_PIXEL_FORMAT_JPEG;
        jpeg_img.rotate = EL_PIXEL_ROTATE_0;

        el_box_t face_box;
        face_box.x = (uint16_t)best_face->bbox.x;
        face_box.y = (uint16_t)best_face->bbox.y;
        face_box.w = (uint16_t)best_face->bbox.w;
        face_box.h = (uint16_t)best_face->bbox.h;
        face_box.score = (uint8_t)(best_face->score * 100);
        face_box.target = 0;

        float landmarks_flat[10];
        for (int i = 0; i < 5; i++) {
            landmarks_flat[i*2]     = best_face->landmarks[i].x;
            landmarks_flat[i*2+1]   = best_face->landmarks[i].y;
        }

        send_face_recognition_json(
            &jpeg_img,
            &face_box,
            embedding_msg->embedding,
            EMBEDDING_OUTPUT_DIM,
            landmarks_flat,
            best_face->score,
            (uint8_t *)aligned_face_img,
            EMBEDDING_INPUT_WIDTH,
            EMBEDDING_INPUT_HEIGHT
        );
    }

    scrfd_free_dets(faces);

#if TOTAL_STEP_TICK
    SystemGetTick(&systick_2, &loop_cnt_2);
    uint32_t algo_tick = (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2);
    uint32_t total_ms = algo_tick / 24000;

    xprintf("[Perf] SCRFD=%lums MFN=%lums Total=%lums (%.1f FPS)\n",
            tick_scrfd / 24000,
            tick_mobilefacenet / 24000,
            total_ms,
            total_ms > 0 ? 1000.0f / total_ms : 0.0f);
#endif

    return 0;
}

int cv_face_embedding_deinit()
{
    xprintf("Face embedding deinitialized\n");
    return 0;
}
