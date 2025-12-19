/*
 * scrfd_postprocessing.h
 *
 * SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
 * Post-processing module for face detection with 5-point landmarks.
 *
 * SCRFD outputs multi-scale detection results with:
 * - Bounding boxes (distance-based format)
 * - Face confidence scores
 * - 5-point facial landmarks (eyes, nose, mouth corners)
 *
 * Created for Grove Vision AI Module V2 face recognition project.
 */

#ifndef SCRFD_POSTPROCESSING_H
#define SCRFD_POSTPROCESSING_H

/* Include pure C type definitions */
#include "scrfd_types.h"

/* C++ specific includes */
#ifdef __cplusplus
#include <forward_list>
#include "tensorflow/lite/c/common.h"
#endif

#ifdef __cplusplus

/**
 * @brief Initialize SCRFD post-processing network
 *
 * @param score_tensors Array of score output tensors (3 strides)
 * @param bbox_tensors  Array of bbox output tensors (3 strides)
 * @param kps_tensors   Array of keypoint output tensors (3 strides)
 * @param input_w       Model input width (default: 160)
 * @param input_h       Model input height (default: 160)
 * @param score_thresh  Score threshold (default: 0.5)
 * @param nms_thresh    NMS threshold (default: 0.4)
 *
 * @return Initialized scrfd_network structure
 */
scrfd_network scrfd_init(
    TfLiteTensor** score_tensors,
    TfLiteTensor** bbox_tensors,
    TfLiteTensor** kps_tensors,
    int input_w,
    int input_h,
    float score_thresh,
    float nms_thresh
);

/**
 * @brief Run face detection on SCRFD outputs
 *
 * Decodes multi-scale outputs, applies NMS, and returns detected faces.
 *
 * @param net       Pointer to initialized SCRFD network
 * @param image_w   Original image width (for coordinate scaling)
 * @param image_h   Original image height (for coordinate scaling)
 * @param num_faces Output: number of detected faces
 *
 * @return Forward list of detected faces with landmarks
 */
std::forward_list<scrfd_face> scrfd_detect(
    scrfd_network* net,
    int image_w,
    int image_h,
    int* num_faces
);

/**
 * @brief Apply Non-Maximum Suppression to face detections
 *
 * @param dets   List of face detections (modified in place)
 * @param thresh IoU threshold for suppression
 */
void scrfd_nms(std::forward_list<scrfd_face>& dets, float thresh);

/**
 * @brief Get the best (highest score) valid face from detection list
 *
 * @param dets     List of face detections
 * @param min_size Minimum bbox size to consider valid (skip smaller faces)
 *
 * @return Pointer to best face, or nullptr if no valid face found
 */
scrfd_face* scrfd_get_best_face(std::forward_list<scrfd_face>& dets, int min_size = 10);

/**
 * @brief Free detection list resources
 *
 * @param dets List of face detections
 */
void scrfd_free_dets(std::forward_list<scrfd_face>& dets);

/**
 * @brief Check if face is suitable for recognition
 *
 * Validates face size, position, and landmark positions.
 *
 * @param face      Pointer to face detection
 * @param min_size  Minimum face size in pixels
 * @param img_w     Image width
 * @param img_h     Image height
 *
 * @return true if face is valid for recognition
 */
bool scrfd_validate_face(
    const scrfd_face* face,
    int min_size,
    int img_w,
    int img_h
);

#endif /* __cplusplus */

#endif /* SCRFD_POSTPROCESSING_H */
