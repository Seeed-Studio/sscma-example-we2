/*
 * scrfd_types.h
 *
 * Pure C header with SCRFD type definitions.
 * Separated from scrfd_postprocessing.h for C compatibility.
 *
 * Created for Grove Vision AI Module V2 face recognition project.
 */

#ifndef SCRFD_TYPES_H
#define SCRFD_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SCRFD Configuration */
#define SCRFD_INPUT_WIDTH       160
#define SCRFD_INPUT_HEIGHT      160
#define SCRFD_NUM_LANDMARKS     5       /* left_eye, right_eye, nose, left_mouth, right_mouth */
#define SCRFD_NUM_STRIDES       3       /* Detection at strides: 8, 16, 32 */
#define SCRFD_NUM_ANCHORS       2       /* Anchors per location for 500M model */

/* Landmark indices for face alignment */
#define SCRFD_LM_LEFT_EYE       0
#define SCRFD_LM_RIGHT_EYE      1
#define SCRFD_LM_NOSE           2
#define SCRFD_LM_LEFT_MOUTH     3
#define SCRFD_LM_RIGHT_MOUTH    4

/* 2D point structure */
typedef struct {
    float x;
    float y;
} scrfd_point2f;

/* Face bounding box structure */
typedef struct {
    float x;        /* Top-left x coordinate */
    float y;        /* Top-left y coordinate */
    float w;        /* Width */
    float h;        /* Height */
} scrfd_bbox;

/* Face detection result with landmarks */
typedef struct {
    scrfd_bbox bbox;                            /* Face bounding box */
    float score;                                /* Detection confidence [0, 1] */
    scrfd_point2f landmarks[SCRFD_NUM_LANDMARKS]; /* 5-point facial landmarks */
    int stride_idx;                             /* Source stride index (0=s8, 1=s16, 2=s32) */
} scrfd_face;

/* Single detection branch (one stride level) */
typedef struct {
    int stride;                 /* Stride value: 8, 16, or 32 */
    int grid_h;                 /* Feature map height */
    int grid_w;                 /* Feature map width */

    /* Score tensor data */
    int8_t* score_data;
    float score_scale;
    int score_zp;

    /* Bounding box tensor data */
    int8_t* bbox_data;
    float bbox_scale;
    int bbox_zp;

    /* Keypoint/landmark tensor data */
    int8_t* kps_data;
    float kps_scale;
    int kps_zp;
} scrfd_branch;

/* SCRFD network configuration */
typedef struct {
    int input_w;                            /* Model input width */
    int input_h;                            /* Model input height */
    int num_branches;                       /* Number of detection scales */
    scrfd_branch branches[SCRFD_NUM_STRIDES]; /* Detection branches */
    float score_thresh;                     /* Score threshold for detection */
    float nms_thresh;                       /* NMS IoU threshold */

    /* Preprocessing parameters for coordinate mapping */
    float scale_x;                          /* X scale factor (Input -> Original) or (1/Scale) */
    float scale_y;                          /* Y scale factor (Input -> Original) or (1/Scale) */
    int pad_x;                              /* Padding offset in x direction */
    int pad_y;                              /* Padding offset in y direction */
} scrfd_network;

#ifdef __cplusplus
}
#endif

#endif /* SCRFD_TYPES_H */
