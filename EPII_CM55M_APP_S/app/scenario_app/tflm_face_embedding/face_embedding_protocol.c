/*
 * face_embedding_protocol.c
 *
 *  Created on: Oct 23, 2025
 *      Author: Face Recognition App
 *
 *  Description: Implementation of face embedding transmission protocol
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "face_embedding_protocol.h"
#include "hx_drv_uart.h"
#include "xprintf.h"

/* Debug verbosity (0=off, 1=on) - must match cvapp_face_recognition.cpp */
#define DEBUG_VERBOSE 0

#if DEBUG_VERBOSE >= 1
#define DBG_INFO(fmt, ...) xprintf(fmt, ##__VA_ARGS__)
#else
#define DBG_INFO(fmt, ...) ((void)0)
#endif

// Forward declaration
static uint16_t calculate_crc16(const uint8_t *data, size_t length);

/* UART protocol header configuration */
#define UART_PROTOCOL_FLAG      0xAA
#define UART_FEATURE_FACE       0x02
#define UART_CMD_EMBEDDING      0xA0

void face_embedding_protocol_init(void) {
    // Initialize UART if not already done
    // UART initialization is typically done in main app
    DBG_INFO("Face embedding protocol initialized\n");
}

void pack_face_embedding_msg(
    face_embedding_msg_t *msg,
    const float *embedding,
    const face_bbox_t *bbox,
    const face_pose_t *pose,
    float confidence,
    uint32_t timestamp,
    uint16_t face_id
) {
    // Save embedding to temp buffer BEFORE clearing msg
    // (in case embedding points to msg->embedding)
    float temp_embedding[EMBEDDING_OUTPUT_DIM];
    memcpy(temp_embedding, embedding, EMBEDDING_OUTPUT_DIM * sizeof(float));

    // Clear message structure
    memset(msg, 0, sizeof(face_embedding_msg_t));

    // Pack header: [Flag, Feature, Command, PayloadLen_L, PayloadLen_H]
    msg->header[0] = UART_PROTOCOL_FLAG;
    msg->header[1] = UART_FEATURE_FACE;
    msg->header[2] = UART_CMD_EMBEDDING;

    uint16_t payload_len = sizeof(face_embedding_msg_t) - 5 - 2; // Exclude header and checksum
    msg->header[3] = payload_len & 0xFF;
    msg->header[4] = (payload_len >> 8) & 0xFF;

    // Pack face data
    msg->face_id = face_id;
    memcpy(msg->embedding, temp_embedding, EMBEDDING_OUTPUT_DIM * sizeof(float));
    msg->timestamp = timestamp;
    msg->confidence = confidence;
    msg->quality = 0.0f;  // Set by caller if needed

    // Pack bounding box
    msg->bbox.x = bbox->x;
    msg->bbox.y = bbox->y;
    msg->bbox.width = bbox->width;
    msg->bbox.height = bbox->height;

    // Pack pose
    msg->pose.yaw = pose->yaw;
    msg->pose.pitch = pose->pitch;
    msg->pose.roll = pose->roll;

    // Landmarks are zeroed by memset, set by caller if needed

    // Calculate and pack checksum
    msg->checksum = calculate_crc16((const uint8_t*)msg, sizeof(face_embedding_msg_t) - 2);
}

static uint16_t calculate_crc16(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFF;

    for (size_t i = 0; i < length; i++) {
        crc ^= (uint16_t)data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }

    return crc;
}

int send_face_embedding_uart(const face_embedding_msg_t *msg) {
    // Send message via UART
    // Using UART write function
    DEV_UART_PTR uart_obj;
    uart_obj = hx_drv_uart_get_dev(USE_DW_UART_0);

    if (uart_obj == NULL) {
        xprintf("ERROR: Failed to get UART device\n");
        return -1;
    }

    int ret = uart_obj->uart_write((void*)msg, sizeof(face_embedding_msg_t));

    if (ret == sizeof(face_embedding_msg_t)) {
        DBG_INFO("Face embedding sent: ID=%u, conf=%.2f, ts=%lu\n",
                msg->face_id, msg->confidence, msg->timestamp);
        return 0;
    } else {
        xprintf("ERROR: Failed to send face embedding (ret=%d)\n", ret);
        return -1;
    }
}

void normalize_embedding(float *embedding, int dim) {
    // Calculate L2 norm
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);

    // Normalize (avoid division by zero)
    if (norm > 1e-6f) {
        for (int i = 0; i < dim; i++) {
            embedding[i] /= norm;
        }
    }
}

int check_face_quality(const face_pose_t *pose, const face_bbox_t *bbox, float confidence) {
    // Check confidence threshold
    if (confidence < FACE_CONF_THRESHOLD) {
        return 0;
    }

    // Check minimum face size
    if (bbox->width < MIN_FACE_SIZE || bbox->height < MIN_FACE_SIZE) {
        return 0;
    }

    // Check pose angles
    if (fabsf(pose->yaw) > MAX_YAW_ANGLE ||
        fabsf(pose->pitch) > MAX_PITCH_ANGLE ||
        fabsf(pose->roll) > MAX_ROLL_ANGLE) {
        return 0;
    }

    return 1; // Face quality is good
}
