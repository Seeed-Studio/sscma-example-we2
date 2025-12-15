/*
 * common_config.h
 *
 * Configuration for Face Embedding using SCRFD + MobileFaceNet.
 *
 * Models:
 *   - SCRFD_500M_KPS: Face detection with 5-point landmarks
 *   - MobileFaceNet: Lightweight face embedding (128D output)
 *
 *  Created on: Dec 11, 2025
 *      Author: Face Embedding App
 */


#ifndef SCENARIO_TFLM_FACE_EMBEDDING_COMMON_CONFIG_H_
#define SCENARIO_TFLM_FACE_EMBEDDING_COMMON_CONFIG_H_

/* Debug and system configuration */
#define FRAME_CHECK_DEBUG               1
#define EN_ALGO                         1
#define SPI_SEN_PIC_CLK                 (12000000)
#define WATCHDOG_VERSION
#define DBG_APP_LOG                     0

/*
 * Model flash addresses (4KB aligned)
 *
 * Memory layout:
 *   0x00000000 - 0x00200000: Firmware (2 MB)
 *   0x00200000 - 0x00400000: SCRFD model (~600 KB)
 *   0x00400000 - 0x00600000: MobileFaceNet model (~1.2 MB)
 */
#define SCRFD_MODEL_FLASH_ADDR          (BASE_ADDR_FLASH1_R_ALIAS + 0x200000)
#define MOBILEFACENET_MODEL_FLASH_ADDR  (BASE_ADDR_FLASH1_R_ALIAS + 0x400000)

/* Legacy defines for backward compatibility */
#define FACE_DETECT_FLASH_ADDR          SCRFD_MODEL_FLASH_ADDR
#define FACE_EMBEDDING_FLASH_ADDR       MOBILEFACENET_MODEL_FLASH_ADDR

/*
 * SCRFD Face Detection Model Configuration
 *
 * SCRFD_500M_KPS specifications:
 *   - Input: 160x160 RGB
 *   - Output: Multi-scale detection (strides 8, 16, 32)
 *   - Features: Bounding boxes + 5-point landmarks
 */
#define FD_INPUT_TENSOR_WIDTH           160
#define FD_INPUT_TENSOR_HEIGHT          160
#define FD_INPUT_TENSOR_CHANNEL         3

/* SCRFD-specific configuration */
#define SCRFD_NUM_STRIDES               3       /* Detection at 3 scales */
#define SCRFD_NUM_ANCHORS_PER_LOC       2       /* Anchors per grid location */
#define SCRFD_NUM_LANDMARKS             5       /* 5-point landmarks */

/*
 * MobileFaceNet Embedding Model Configuration
 *
 * sirius-ai MobileFaceNet specifications:
 *   - Source: https://github.com/sirius-ai/MobileFaceNet_TF
 *   - Input: 112x112 RGB (aligned face, normalized to [-1,1])
 *   - Output: 128-dimensional embedding (L2 normalized)
 *   - Accuracy: 99.25% LFW
 *   - Inference: ~105ms on Ethos-U55 (99.6% NPU, 0.4% CPU)
 */
#define EMBEDDING_INPUT_WIDTH           112
#define EMBEDDING_INPUT_HEIGHT          112
#define EMBEDDING_INPUT_CHANNEL         3
#define EMBEDDING_OUTPUT_DIM            128     /* MobileFaceNet outputs 128D embeddings */

/*
 * Memory Configuration
 *
 * Tensor arena allocation for dual-model inference.
 *
 * Memory usage (from Vela 3.12.0 compilation):
 *   - SCRFD: 220 KB (same as face_recognition)
 *   - MobileFaceNet: 1200 KB (Vela reports 1176 KB SRAM)
 *   - Total: 1420 KB
 *
 * Note: sirius-ai MobileFaceNet (99.25% LFW) requires significantly more
 * SRAM than GhostFaceNet due to different architecture (more activations).
 */
#define SCRFD_ARENA_SIZE                (220 * 1024)    /* 220 KB for face detection */
#define MOBILEFACENET_ARENA_SIZE        (1200 * 1024)   /* 1200 KB for face embedding (Vela: 1176 KB) */

/* Legacy define for total reference */
#define TENSOR_ARENA_SIZE               (SCRFD_ARENA_SIZE + MOBILEFACENET_ARENA_SIZE)

/* Aligned face buffer size (112x112 RGB) */
#define ALIGNED_FACE_BUFFER_SIZE        (112 * 112 * 3)

/* Face detection thresholds */
#define FACE_CONF_THRESHOLD             0.70f   /* Confidence threshold for face detection */
#define FACE_NMS_THRESHOLD              0.4f    /* NMS IoU threshold */
#define MIN_FACE_SIZE                   40      /* Minimum face size in pixels */

/* Face pose limits for quality filtering (more lenient with alignment) */
#define MAX_YAW_ANGLE                   45.0f   /* Left-right rotation limit */
#define MAX_PITCH_ANGLE                 45.0f   /* Up-down rotation limit */
#define MAX_ROLL_ANGLE                  45.0f   /* In-plane rotation limit */

/* Face quality threshold */
#define MIN_FACE_QUALITY                0.3f    /* Minimum quality score for recognition */

/* UART communication */
#define DATA_TYPE_FACE_EMBEDDING        0xA0

/* Enable face alignment (recommended for best accuracy) */
/* TODO: Fix alignment for RGB planar input - disabled for now to avoid crash */
#define ENABLE_FACE_ALIGNMENT           0

#endif /* SCENARIO_TFLM_FACE_EMBEDDING_COMMON_CONFIG_H_ */
