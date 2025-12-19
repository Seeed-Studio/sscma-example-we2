/*
 * face_alignment.h
 *
 * Face alignment module using 5-point landmarks.
 *
 * This module provides functionality to:
 * - Compute similarity transform from detected landmarks
 * - Apply affine transformation to align faces to canonical position
 * - Prepare aligned face images for embedding extraction
 *
 * The alignment uses the ArcFace standard reference positions,
 * ensuring consistent face orientation for recognition.
 *
 * Created for Grove Vision AI Module V2 face recognition project.
 */

#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include <stdint.h>
#include "scrfd_types.h"  /* Pure C header with type definitions */

#ifdef __cplusplus
extern "C" {
#endif

/* Aligned face output dimensions (GhostFaceNet input) */
#define ALIGNED_FACE_WIDTH      112
#define ALIGNED_FACE_HEIGHT     112
#define ALIGNED_FACE_CHANNELS   3

/* Total bytes for aligned face image (RGB interleaved) */
#define ALIGNED_FACE_SIZE       (ALIGNED_FACE_WIDTH * ALIGNED_FACE_HEIGHT * ALIGNED_FACE_CHANNELS)

/**
 * @brief Affine transformation matrix (2x3)
 *
 * Transformation equations:
 *   x' = m[0]*x + m[1]*y + m[2]
 *   y' = m[3]*x + m[4]*y + m[5]
 */
typedef struct {
    float m[6];
} affine_transform_t;

/**
 * @brief Compute face alignment transformation from 5-point landmarks
 *
 * Calculates a similarity transform (rotation + scale + translation)
 * that aligns the detected face to the ArcFace canonical position.
 *
 * Reference landmark positions (112x112 output):
 *   Left eye:     (38.29, 51.70)
 *   Right eye:    (73.53, 51.50)
 *   Nose:         (56.03, 71.74)
 *   Left mouth:   (41.55, 92.37)
 *   Right mouth:  (70.73, 92.20)
 *
 * @param landmarks     Input 5-point landmarks from SCRFD detection
 * @param transform     Output affine transformation matrix
 */
void compute_face_alignment(
    const scrfd_point2f* landmarks,
    affine_transform_t* transform
);

/**
 * @brief Apply affine transformation to align face image
 *
 * Warps the source image region containing the face to produce
 * an aligned 112x112 RGB image suitable for embedding extraction.
 *
 * Input format:  BGR planar (camera output format)
 * Output format: RGB interleaved (model input format)
 *
 * Uses bilinear interpolation for high-quality resampling.
 *
 * @param src_image     Source image in BGR planar format
 * @param src_w         Source image width
 * @param src_h         Source image height
 * @param dst_image     Output aligned face (112x112 RGB interleaved)
 * @param transform     Affine transformation matrix
 */
void apply_face_alignment(
    const uint8_t* src_image,
    int src_w,
    int src_h,
    uint8_t* dst_image,
    const affine_transform_t* transform
);

/**
 * @brief Compute inverse affine transformation
 *
 * @param forward   Forward transformation matrix
 * @param inverse   Output inverse transformation matrix
 */
void invert_affine_transform(
    const affine_transform_t* forward,
    affine_transform_t* inverse
);

/**
 * @brief Estimate face quality from landmark positions
 *
 * Returns a quality score based on:
 * - Face frontality (eye horizontal alignment)
 * - Nose position relative to eyes
 * - Symmetry of mouth corners
 *
 * @param landmarks     5-point landmarks
 *
 * @return Quality score in range [0, 1], higher is better
 */
float estimate_face_quality(const scrfd_point2f* landmarks);

/**
 * @brief Estimate face pose angles from landmarks
 *
 * Provides rough estimates of yaw, pitch, and roll angles
 * based on landmark geometry.
 *
 * @param landmarks     5-point landmarks
 * @param yaw           Output yaw angle (left-right rotation)
 * @param pitch         Output pitch angle (up-down rotation)
 * @param roll          Output roll angle (in-plane rotation)
 */
void estimate_face_pose(
    const scrfd_point2f* landmarks,
    float* yaw,
    float* pitch,
    float* roll
);

#ifdef __cplusplus
}
#endif

#endif /* FACE_ALIGNMENT_H */
