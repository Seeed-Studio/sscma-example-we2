/*
 * face_alignment.c
 *
 * Face alignment implementation using 5-point landmarks.
 *
 * Uses similarity transform (rotation + uniform scale + translation)
 * to align detected faces to the ArcFace canonical position.
 *
 * Created for Grove Vision AI Module V2 face recognition project.
 */

#include "face_alignment.h"
#include <math.h>
#include <string.h>

/*
 * ArcFace canonical landmark positions for 112x112 aligned face.
 * These reference positions are the standard used for training
 * face recognition models like ArcFace, CosFace, etc.
 */
static const scrfd_point2f REFERENCE_LANDMARKS[SCRFD_NUM_LANDMARKS] = {
    {38.2946f, 51.6963f},   /* Left eye */
    {73.5318f, 51.5014f},   /* Right eye */
    {56.0252f, 71.7366f},   /* Nose tip */
    {41.5493f, 92.3655f},   /* Left mouth corner */
    {70.7299f, 92.2041f}    /* Right mouth corner */
};

void compute_face_alignment(
    const scrfd_point2f* landmarks,
    affine_transform_t* transform
) {
    if (landmarks == NULL || transform == NULL) return;

    /*
     * Compute similarity transform using eye positions.
     *
     * The similarity transform has 4 DOF:
     * - 1 rotation angle
     * - 1 uniform scale
     * - 2 translation (x, y)
     *
     * We use the eye positions as the primary alignment reference
     * since they provide the most stable features.
     */

    /* Source eye positions (from detection) */
    float src_left_eye_x = landmarks[SCRFD_LM_LEFT_EYE].x;
    float src_left_eye_y = landmarks[SCRFD_LM_LEFT_EYE].y;
    float src_right_eye_x = landmarks[SCRFD_LM_RIGHT_EYE].x;
    float src_right_eye_y = landmarks[SCRFD_LM_RIGHT_EYE].y;

    /* Source eye center */
    float src_eye_center_x = (src_left_eye_x + src_right_eye_x) / 2.0f;
    float src_eye_center_y = (src_left_eye_y + src_right_eye_y) / 2.0f;

    /* Source inter-ocular vector */
    float src_dx = src_right_eye_x - src_left_eye_x;
    float src_dy = src_right_eye_y - src_left_eye_y;
    float src_dist = sqrtf(src_dx * src_dx + src_dy * src_dy);

    /* Destination (reference) eye positions */
    float dst_left_eye_x = REFERENCE_LANDMARKS[SCRFD_LM_LEFT_EYE].x;
    float dst_left_eye_y = REFERENCE_LANDMARKS[SCRFD_LM_LEFT_EYE].y;
    float dst_right_eye_x = REFERENCE_LANDMARKS[SCRFD_LM_RIGHT_EYE].x;
    float dst_right_eye_y = REFERENCE_LANDMARKS[SCRFD_LM_RIGHT_EYE].y;

    /* Destination eye center */
    float dst_eye_center_x = (dst_left_eye_x + dst_right_eye_x) / 2.0f;
    float dst_eye_center_y = (dst_left_eye_y + dst_right_eye_y) / 2.0f;

    /* Destination inter-ocular vector */
    float dst_dx = dst_right_eye_x - dst_left_eye_x;
    float dst_dy = dst_right_eye_y - dst_left_eye_y;
    float dst_dist = sqrtf(dst_dx * dst_dx + dst_dy * dst_dy);

    /* Compute scale factor */
    float scale = dst_dist / (src_dist + 1e-6f);

    /* Compute rotation angle */
    float src_angle = atan2f(src_dy, src_dx);
    float dst_angle = atan2f(dst_dy, dst_dx);
    float angle = dst_angle - src_angle;

    /* Rotation matrix components (with scale) */
    float cos_a = cosf(angle) * scale;
    float sin_a = sinf(angle) * scale;

    /*
     * Build affine transformation matrix.
     *
     * The transform maps source coordinates to destination:
     *   x' = cos_a * (x - cx_src) - sin_a * (y - cy_src) + cx_dst
     *   y' = sin_a * (x - cx_src) + cos_a * (y - cy_src) + cy_dst
     *
     * Expanding:
     *   x' = cos_a * x - sin_a * y + (cx_dst - cos_a * cx_src + sin_a * cy_src)
     *   y' = sin_a * x + cos_a * y + (cy_dst - sin_a * cx_src - cos_a * cy_src)
     */

    transform->m[0] = cos_a;
    transform->m[1] = -sin_a;
    transform->m[2] = dst_eye_center_x - cos_a * src_eye_center_x + sin_a * src_eye_center_y;

    transform->m[3] = sin_a;
    transform->m[4] = cos_a;
    transform->m[5] = dst_eye_center_y - sin_a * src_eye_center_x - cos_a * src_eye_center_y;
}

void invert_affine_transform(
    const affine_transform_t* forward,
    affine_transform_t* inverse
) {
    if (forward == NULL || inverse == NULL) return;

    /*
     * For affine transform [a, b, c; d, e, f]:
     *
     * The 2x2 rotation/scale part has inverse:
     *   [a, b]^-1 = 1/det * [e, -b; -d, a]
     *
     * where det = a*e - b*d
     *
     * Full inverse transform:
     *   x = inv_a * (x' - c) + inv_b * (y' - f)
     *   y = inv_d * (x' - c) + inv_e * (y' - f)
     */

    float a = forward->m[0];
    float b = forward->m[1];
    float c = forward->m[2];
    float d = forward->m[3];
    float e = forward->m[4];
    float f = forward->m[5];

    float det = a * e - b * d;
    float inv_det = 1.0f / (det + 1e-8f);

    inverse->m[0] = e * inv_det;
    inverse->m[1] = -b * inv_det;
    inverse->m[2] = (b * f - e * c) * inv_det;

    inverse->m[3] = -d * inv_det;
    inverse->m[4] = a * inv_det;
    inverse->m[5] = (d * c - a * f) * inv_det;
}

void apply_face_alignment(
    const uint8_t* src_image,
    int src_w,
    int src_h,
    uint8_t* dst_image,
    const affine_transform_t* transform
) {
    if (src_image == NULL || dst_image == NULL || transform == NULL) return;

    /*
     * Source image format: BGR planar
     * [B plane (src_w * src_h)] [G plane (src_w * src_h)] [R plane (src_w * src_h)]
     *
     * Destination image format: RGB interleaved
     * [R, G, B, R, G, B, ...]
     */

    int src_plane_size = src_w * src_h;
    /* BGR planar: plane order is B, G, R */
    const uint8_t* b_plane = src_image;
    const uint8_t* g_plane = src_image + src_plane_size;
    const uint8_t* r_plane = g_plane + src_plane_size;

    /* Compute inverse transform for backward mapping */
    affine_transform_t inv_transform;
    invert_affine_transform(transform, &inv_transform);

    float inv_a = inv_transform.m[0];
    float inv_b = inv_transform.m[1];
    float inv_c = inv_transform.m[2];
    float inv_d = inv_transform.m[3];
    float inv_e = inv_transform.m[4];
    float inv_f = inv_transform.m[5];

    /* Iterate over destination pixels */
    for (int dy = 0; dy < ALIGNED_FACE_HEIGHT; dy++) {
        for (int dx = 0; dx < ALIGNED_FACE_WIDTH; dx++) {
            /* Backward map: find source coordinates for this destination pixel */
            float sx = inv_a * dx + inv_b * dy + inv_c;
            float sy = inv_d * dx + inv_e * dy + inv_f;

            /* Destination pixel index (RGB interleaved) */
            int dst_idx = (dy * ALIGNED_FACE_WIDTH + dx) * 3;

            /* Integer source coordinates for bilinear interpolation */
            int ix = (int)floorf(sx);
            int iy = (int)floorf(sy);

            /* Check bounds with margin for interpolation */
            if (ix >= 0 && ix < src_w - 1 && iy >= 0 && iy < src_h - 1) {
                /* Bilinear interpolation weights */
                float fx = sx - ix;
                float fy = sy - iy;
                float w00 = (1.0f - fx) * (1.0f - fy);
                float w01 = fx * (1.0f - fy);
                float w10 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                /* Source pixel indices */
                int src_idx00 = iy * src_w + ix;
                int src_idx01 = src_idx00 + 1;
                int src_idx10 = src_idx00 + src_w;
                int src_idx11 = src_idx10 + 1;

                /* Interpolate R channel */
                float r_val = w00 * r_plane[src_idx00] +
                              w01 * r_plane[src_idx01] +
                              w10 * r_plane[src_idx10] +
                              w11 * r_plane[src_idx11];
                dst_image[dst_idx] = (uint8_t)(r_val + 0.5f);

                /* Interpolate G channel */
                float g_val = w00 * g_plane[src_idx00] +
                              w01 * g_plane[src_idx01] +
                              w10 * g_plane[src_idx10] +
                              w11 * g_plane[src_idx11];
                dst_image[dst_idx + 1] = (uint8_t)(g_val + 0.5f);

                /* Interpolate B channel */
                float b_val = w00 * b_plane[src_idx00] +
                              w01 * b_plane[src_idx01] +
                              w10 * b_plane[src_idx10] +
                              w11 * b_plane[src_idx11];
                dst_image[dst_idx + 2] = (uint8_t)(b_val + 0.5f);
            } else {
                /* Out of bounds - use black padding */
                dst_image[dst_idx] = 0;
                dst_image[dst_idx + 1] = 0;
                dst_image[dst_idx + 2] = 0;
            }
        }
    }
}

float estimate_face_quality(const scrfd_point2f* landmarks) {
    if (landmarks == NULL) return 0.0f;

    float quality = 1.0f;

    /* 1. Eye horizontal alignment (roll check) */
    float eye_dy = fabsf(landmarks[SCRFD_LM_LEFT_EYE].y - landmarks[SCRFD_LM_RIGHT_EYE].y);
    float eye_dx = fabsf(landmarks[SCRFD_LM_LEFT_EYE].x - landmarks[SCRFD_LM_RIGHT_EYE].x);
    float roll_ratio = eye_dy / (eye_dx + 1e-6f);
    float roll_score = fmaxf(0.0f, 1.0f - roll_ratio * 3.0f);
    quality *= roll_score;

    /* 2. Nose position symmetry (yaw check) */
    float eye_center_x = (landmarks[SCRFD_LM_LEFT_EYE].x + landmarks[SCRFD_LM_RIGHT_EYE].x) / 2.0f;
    float nose_offset = fabsf(landmarks[SCRFD_LM_NOSE].x - eye_center_x);
    float nose_range = eye_dx / 2.0f;
    float yaw_score = fmaxf(0.0f, 1.0f - nose_offset / (nose_range + 1e-6f));
    quality *= yaw_score;

    /* 3. Mouth symmetry */
    float mouth_center_x = (landmarks[SCRFD_LM_LEFT_MOUTH].x + landmarks[SCRFD_LM_RIGHT_MOUTH].x) / 2.0f;
    float mouth_offset = fabsf(mouth_center_x - eye_center_x);
    float mouth_score = fmaxf(0.0f, 1.0f - mouth_offset / (nose_range + 1e-6f));
    quality *= mouth_score;

    return quality;
}

void estimate_face_pose(
    const scrfd_point2f* landmarks,
    float* yaw,
    float* pitch,
    float* roll
) {
    if (landmarks == NULL) return;

    /* Eye vector for roll estimation */
    float eye_dx = landmarks[SCRFD_LM_RIGHT_EYE].x - landmarks[SCRFD_LM_LEFT_EYE].x;
    float eye_dy = landmarks[SCRFD_LM_RIGHT_EYE].y - landmarks[SCRFD_LM_LEFT_EYE].y;

    if (roll != NULL) {
        /* Roll: rotation in image plane (degrees) */
        *roll = atan2f(eye_dy, eye_dx) * 180.0f / 3.14159265f;
    }

    /* Eye center for yaw/pitch estimation */
    float eye_center_x = (landmarks[SCRFD_LM_LEFT_EYE].x + landmarks[SCRFD_LM_RIGHT_EYE].x) / 2.0f;
    float eye_center_y = (landmarks[SCRFD_LM_LEFT_EYE].y + landmarks[SCRFD_LM_RIGHT_EYE].y) / 2.0f;
    float eye_width = sqrtf(eye_dx * eye_dx + eye_dy * eye_dy);

    if (yaw != NULL) {
        /* Yaw: left-right rotation (rough estimate from nose position) */
        float nose_offset = landmarks[SCRFD_LM_NOSE].x - eye_center_x;
        float norm_offset = nose_offset / (eye_width / 2.0f + 1e-6f);
        *yaw = norm_offset * 45.0f;  /* Scale to approximate degrees */
    }

    if (pitch != NULL) {
        /* Pitch: up-down rotation (rough estimate from nose-eye distance) */
        float nose_eye_dist = landmarks[SCRFD_LM_NOSE].y - eye_center_y;
        float expected_dist = eye_width * 0.6f;  /* Typical ratio */
        float ratio = nose_eye_dist / (expected_dist + 1e-6f);
        *pitch = (ratio - 1.0f) * 30.0f;  /* Scale deviation to degrees */
    }
}
