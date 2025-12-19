/*
 * face_embedding_protocol.h
 *
 *  Created on: Dec 11, 2025
 *      Author: Face Embedding App
 *
 *  Description: Protocol structures for face embedding transmission via UART
 *
 *  Updated for MobileFaceNet with 128-dimensional embeddings
 */

#ifndef FACE_EMBEDDING_PROTOCOL_H_
#define FACE_EMBEDDING_PROTOCOL_H_

#include <stdint.h>
#include <stddef.h>
#include "common_config.h"

/* Face bounding box structure */
typedef struct {
    uint16_t x;
    uint16_t y;
    uint16_t width;
    uint16_t height;
} face_bbox_t;

/* Face quality metrics */
typedef struct {
    float yaw;      // Head rotation left/right (-180 to 180 degrees)
    float pitch;    // Head tilt up/down (-90 to 90 degrees)
    float roll;     // Head tilt side-to-side (-180 to 180 degrees)
} face_pose_t;

/* 2D landmark point */
typedef struct {
    float x;
    float y;
} face_landmark_t;

/* Face embedding message structure for UART transmission */
typedef struct __attribute__((packed)) {
    uint8_t  header[5];                         // Standard UART protocol header
    uint8_t  reserved;                          // Padding for 4-byte alignment
    uint16_t face_id;                           // Tracking ID (0 if not tracked)
    float    embedding[EMBEDDING_OUTPUT_DIM] __attribute__((aligned(4)));   // 128-dimensional embedding (512 bytes), must be 4-byte aligned!
    uint32_t timestamp;                         // Frame timestamp (milliseconds)
    float    confidence;                        // Face detection confidence (0.0-1.0)
    float    quality;                           // Face quality score (0.0-1.0)
    face_bbox_t bbox;                           // Face bounding box
    face_pose_t pose;                           // Face pose angles
    face_landmark_t landmarks[5];               // 5-point landmarks for debugging
    uint16_t checksum;                          // CRC16 checksum
} face_embedding_msg_t;

/*
 * Message size calculation (128D version):
 *   header:     5 bytes
 *   reserved:   1 byte (padding for 4-byte alignment)
 *   face_id:    2 bytes
 *   embedding:  128 * 4 = 512 bytes (must be 4-byte aligned for Cortex-M55 MVE!)
 *   timestamp:  4 bytes
 *   confidence: 4 bytes
 *   quality:    4 bytes
 *   bbox:       8 bytes
 *   pose:       12 bytes
 *   landmarks:  5 * 8 = 40 bytes
 *   checksum:   2 bytes
 *   Total:      594 bytes (was 2130 bytes for 512D)
 */
#define FACE_EMBEDDING_MSG_SIZE    sizeof(face_embedding_msg_t)

/* Function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize face embedding protocol
 */
void face_embedding_protocol_init(void);

/**
 * @brief Pack face embedding data into message structure
 *
 * @param msg Pointer to message structure to fill
 * @param embedding Pointer to 128-dimensional embedding array
 * @param bbox Pointer to face bounding box
 * @param pose Pointer to face pose angles
 * @param confidence Face detection confidence score
 * @param timestamp Frame timestamp
 * @param face_id Optional tracking ID (0 if not used)
 */
void pack_face_embedding_msg(
    face_embedding_msg_t *msg,
    const float *embedding,
    const face_bbox_t *bbox,
    const face_pose_t *pose,
    float confidence,
    uint32_t timestamp,
    uint16_t face_id
);

/**
 * @brief Send face embedding message via UART
 *
 * @param msg Pointer to face embedding message
 * @return int 0 on success, -1 on error
 */
int send_face_embedding_uart(const face_embedding_msg_t *msg);

/**
 * @brief Normalize embedding vector (L2 normalization)
 *
 * @param embedding Pointer to embedding array (modified in-place)
 * @param dim Dimension of embedding (should be EMBEDDING_OUTPUT_DIM)
 */
void normalize_embedding(float *embedding, int dim);

/**
 * @brief Check if face quality is sufficient for recognition
 *
 * @param pose Face pose angles
 * @param bbox Face bounding box
 * @param confidence Detection confidence
 * @return int 1 if quality is good, 0 otherwise
 */
int check_face_quality(const face_pose_t *pose, const face_bbox_t *bbox, float confidence);

#ifdef __cplusplus
}
#endif

#endif /* FACE_EMBEDDING_PROTOCOL_H_ */
