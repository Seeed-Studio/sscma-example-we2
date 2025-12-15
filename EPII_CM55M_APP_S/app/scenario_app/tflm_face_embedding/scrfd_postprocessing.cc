/*
 * scrfd_postprocessing.cc
 *
 * SCRFD post-processing implementation for face detection with landmarks.
 *
 * This module handles:
 * - Multi-scale output decoding (strides 8, 16, 32)
 * - INT8 dequantization
 * - Distance-based bounding box decoding
 * - 5-point landmark extraction
 * - Non-Maximum Suppression (NMS)
 *
 * Created for Grove Vision AI Module V2 face recognition project.
 */

#include "scrfd_postprocessing.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

extern "C" {
#include "xprintf.h"
}

/* Detection stride values for SCRFD_500M */
static const int STRIDES[SCRFD_NUM_STRIDES] = {8, 16, 32};

/* Temporal stability: store previous frame's best face */
static scrfd_face g_prev_best_face = {0};
static bool g_has_prev_face = false;

/* Stability parameters */
static constexpr float CENTER_DIST_THRESH_RATIO = 0.3f;  /* Max center distance as ratio of bbox size */
static constexpr float AREA_CHANGE_THRESH = 0.5f;        /* Max area change ratio (50%) */
static constexpr float SMOOTHING_ALPHA = 0.7f;           /* EMA smoothing factor (higher = more responsive) */
static constexpr float MAX_FACE_RATIO = 0.6f;            /* Max face size as ratio of image dimension */

/**
 * @brief Sigmoid activation function
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Dequantize INT8 value to float
 */
static inline float dequantize(int8_t value, float scale, int zero_point) {
    return (float)(value - zero_point) * scale;
}

/**
 * @brief Compute IoU between two bounding boxes
 */
static float compute_iou(const scrfd_bbox& a, const scrfd_bbox& b) {
    float x1 = fmaxf(a.x, b.x);
    float y1 = fmaxf(a.y, b.y);
    float x2 = fminf(a.x + a.w, b.x + b.w);
    float y2 = fminf(a.y + a.h, b.y + b.h);

    float inter_w = fmaxf(0.0f, x2 - x1);
    float inter_h = fmaxf(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float union_area = a.w * a.h + b.w * b.h - inter_area;

    if (union_area <= 0.0f) return 0.0f;

    return inter_area / union_area;
}

/**
 * @brief Compute center distance between two bounding boxes
 */
static float compute_center_distance(const scrfd_bbox& a, const scrfd_bbox& b) {
    float cx_a = a.x + a.w / 2.0f;
    float cy_a = a.y + a.h / 2.0f;
    float cx_b = b.x + b.w / 2.0f;
    float cy_b = b.y + b.h / 2.0f;

    float dx = cx_a - cx_b;
    float dy = cy_a - cy_b;
    return sqrtf(dx * dx + dy * dy);
}

/**
 * @brief Check if two detections are likely the same face (center-based)
 *
 * For cross-stride suppression: if centers are close relative to the
 * smaller bbox size, they're likely the same face.
 */
static bool is_same_face_by_center(const scrfd_face& a, const scrfd_face& b, float thresh_ratio) {
    float dist = compute_center_distance(a.bbox, b.bbox);
    /* Use the smaller bbox as reference size */
    float ref_size = fminf(fminf(a.bbox.w, a.bbox.h), fminf(b.bbox.w, b.bbox.h));
    return dist < ref_size * thresh_ratio;
}

/**
 * @brief Cross-stride suppression using center distance
 *
 * When detections from different strides have similar centers,
 * keep the one from finer stride (smaller bbox, more precise).
 */
static void cross_stride_suppress(std::forward_list<scrfd_face>& dets, float center_thresh_ratio) {
    /* Sort by bbox area (ascending) - finer stride detections first */
    dets.sort([](const scrfd_face& a, const scrfd_face& b) {
        return (a.bbox.w * a.bbox.h) < (b.bbox.w * b.bbox.h);
    });

    /* Suppress larger detections that have similar center to smaller ones */
    for (auto it = dets.begin(); it != dets.end(); ++it) {
        if (it->score <= 0) continue;

        for (auto jt = std::next(it); jt != dets.end(); ++jt) {
            if (jt->score <= 0) continue;

            /* Check if centers are close */
            if (is_same_face_by_center(*it, *jt, center_thresh_ratio)) {
                /* Suppress the larger one (jt, since list is sorted by area ascending) */
                jt->score = 0;
            }
        }
    }
}

scrfd_network scrfd_init(
    TfLiteTensor** score_tensors,
    TfLiteTensor** bbox_tensors,
    TfLiteTensor** kps_tensors,
    int input_w,
    int input_h,
    float score_thresh,
    float nms_thresh
) {
    scrfd_network net;

    net.input_w = input_w;
    net.input_h = input_h;
    net.num_branches = SCRFD_NUM_STRIDES;
    net.score_thresh = score_thresh;
    net.nms_thresh = nms_thresh;

    /* Default params: no scaling (1:1) */
    net.scale_x = 1.0f;
    net.scale_y = 1.0f;
    net.pad_x = 0;
    net.pad_y = 0;

    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        scrfd_branch* branch = &net.branches[i];

        branch->stride = STRIDES[i];
        branch->grid_h = input_h / STRIDES[i];
        branch->grid_w = input_w / STRIDES[i];

        /* Score tensor configuration */
        if (score_tensors && score_tensors[i]) {
            branch->score_data = score_tensors[i]->data.int8;
            branch->score_scale = score_tensors[i]->params.scale;
            branch->score_zp = score_tensors[i]->params.zero_point;
        } else {
            branch->score_data = nullptr;
            branch->score_scale = 1.0f;
            branch->score_zp = 0;
        }

        /* Bounding box tensor configuration */
        if (bbox_tensors && bbox_tensors[i]) {
            branch->bbox_data = bbox_tensors[i]->data.int8;
            branch->bbox_scale = bbox_tensors[i]->params.scale;
            branch->bbox_zp = bbox_tensors[i]->params.zero_point;
        } else {
            branch->bbox_data = nullptr;
            branch->bbox_scale = 1.0f;
            branch->bbox_zp = 0;
        }

        /* Keypoint tensor configuration */
        if (kps_tensors && kps_tensors[i]) {
            branch->kps_data = kps_tensors[i]->data.int8;
            branch->kps_scale = kps_tensors[i]->params.scale;
            branch->kps_zp = kps_tensors[i]->params.zero_point;
        } else {
            branch->kps_data = nullptr;
            branch->kps_scale = 1.0f;
            branch->kps_zp = 0;
        }
    }

    return net;
}

std::forward_list<scrfd_face> scrfd_detect(
    scrfd_network* net,
    int image_w,
    int image_h,
    int* num_faces
) {
    std::forward_list<scrfd_face> dets;
    *num_faces = 0;

    if (net == nullptr) {
        xprintf("ERROR: scrfd_detect net is NULL!\n");
        return dets;
    }

    /*
     * Coordinate mapping (Model Space -> Original Image Space)
     *
     * Supports both Letterbox (uniform scale + pad) and Stretch (non-uniform scale).
     *
     * Formula:
     *   orig = (model - pad) * scale
     *
     * Where scale = Original_Size / Model_Input_Size
     */
    float sc_x = net->scale_x;
    float sc_y = net->scale_y;
    int p_x = net->pad_x;
    int p_y = net->pad_y;

    /* Debug: print mapping info once */
    static int printed_cnt = 0;
    if (printed_cnt < 3) {
        xprintf("[SCRFD] img=%dx%d input=%dx%d scale=(%d,%d)/1000 pad=(%d,%d)\n",
                image_w, image_h, net->input_w, net->input_h,
                (int)(sc_x * 1000), (int)(sc_y * 1000), p_x, p_y);
        xprintf("[SCRFD] branches=%d: ", net->num_branches);
        for (int i = 0; i < net->num_branches; i++) {
            xprintf("s%d(%dx%d) ", net->branches[i].stride,
                    net->branches[i].grid_w, net->branches[i].grid_h);
        }
        xprintf("\n");
        printed_cnt++;
    }

    /* Track max score for debugging */
    float max_score = 0.0f;
    int max_score_branch = -1;
    float max_score_per_stride[3] = {0.0f, 0.0f, 0.0f};  /* stride 8, 16, 32 */

    /* Per-stride detection lists for hierarchical NMS */
    std::forward_list<scrfd_face> stride_dets[SCRFD_NUM_STRIDES];
    int stride_counts[SCRFD_NUM_STRIDES] = {0, 0, 0};

    /* Process each detection stride */
    for (int s = 0; s < net->num_branches; s++) {
        scrfd_branch* branch = &net->branches[s];

        if (branch->score_data == nullptr ||
            branch->bbox_data == nullptr ||
            branch->kps_data == nullptr) {
            continue;
        }

        int stride = branch->stride;
        int grid_h = branch->grid_h;
        int grid_w = branch->grid_w;

        /*
         * Vela 2D output format: [num_elements, channels]
         * - score: [H*W*anchors, 1]  -> access: data[row_idx]
         * - bbox:  [H*W*anchors, 4]  -> access: data[row_idx * 4 + col]
         * - kps:   [H*W*anchors, 10] -> access: data[row_idx * 10 + col]
         *
         * row_idx = (h * grid_w + w) * num_anchors + anchor_idx
         */

        /* Iterate over feature map */
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                for (int a = 0; a < SCRFD_NUM_ANCHORS; a++) {
                    /* Row index in the flattened 2D tensor */
                    int row_idx = (h * grid_w + w) * SCRFD_NUM_ANCHORS + a;

                    /* Decode face score - score tensor is [N, 1]
                     * Note: TFLite model outputs probabilities directly (0-1 range),
                     * NOT logits. Do NOT apply sigmoid!
                     * Clamp to [0,1] to handle minor quantization overflow. */
                    float score = dequantize(
                        branch->score_data[row_idx],
                        branch->score_scale,
                        branch->score_zp
                    );
                    score = fminf(1.0f, fmaxf(0.0f, score));

                    /* Track max score for debugging */
                    if (score > max_score) {
                        max_score = score;
                        max_score_branch = s;
                    }
                    if (score > max_score_per_stride[s]) {
                        max_score_per_stride[s] = score;
                    }

                    if (score < net->score_thresh) continue;

                    /* Decode bounding box - bbox tensor is [N, 4] */
                    int bbox_base = row_idx * 4;

                    float d_left = dequantize(branch->bbox_data[bbox_base + 0], branch->bbox_scale, branch->bbox_zp);
                    float d_top = dequantize(branch->bbox_data[bbox_base + 1], branch->bbox_scale, branch->bbox_zp);
                    float d_right = dequantize(branch->bbox_data[bbox_base + 2], branch->bbox_scale, branch->bbox_zp);
                    float d_bottom = dequantize(branch->bbox_data[bbox_base + 3], branch->bbox_scale, branch->bbox_zp);

                    /* Anchor center in input coordinates */
                    float cx = (w + 0.5f) * stride;
                    float cy = (h + 0.5f) * stride;

                    /*
                     * Distance-based bbox decoding
                     * SCRFD outputs distance values that need to be scaled by stride.
                     * Following InsightFace reference: bbox_preds = bbox_preds * stride
                     */
                    float x1 = cx - d_left * stride;
                    float y1 = cy - d_top * stride;
                    float x2 = cx + d_right * stride;
                    float y2 = cy + d_bottom * stride;

                    /* Clamp to input image bounds */
                    x1 = fmaxf(0.0f, x1);
                    y1 = fmaxf(0.0f, y1);
                    x2 = fminf((float)net->input_w, x2);
                    y2 = fminf((float)net->input_h, y2);

                    /*
                     * Map coordinates from model input space to original image space.
                     * Formula: orig = (model - pad) * scale
                     */
                    scrfd_face det;

                    float orig_x1 = (x1 - p_x) * sc_x;
                    float orig_x2 = (x2 - p_x) * sc_x;
                    float orig_y1 = (y1 - p_y) * sc_y;
                    float orig_y2 = (y2 - p_y) * sc_y;

                    /* Clamp to original image bounds */
                    orig_x1 = fmaxf(0.0f, orig_x1);
                    orig_y1 = fmaxf(0.0f, orig_y1);
                    orig_x2 = fminf((float)image_w, orig_x2);
                    orig_y2 = fminf((float)image_h, orig_y2);

                    det.bbox.x = orig_x1;
                    det.bbox.y = orig_y1;
                    det.bbox.w = orig_x2 - orig_x1;
                    det.bbox.h = orig_y2 - orig_y1;
                    det.score = score;

                    /* Decode 5-point landmarks - kps tensor is [N, 10]
                     * Note: Landmarks use anchor CORNER (w*stride),
                     * NOT anchor center like bbox. This is different from box decoding! */
                    int kps_base = row_idx * 10;
                    for (int k = 0; k < SCRFD_NUM_LANDMARKS; k++) {
                        float kp_dx = dequantize(
                            branch->kps_data[kps_base + k * 2],
                            branch->kps_scale,
                            branch->kps_zp
                        );
                        float kp_dy = dequantize(
                            branch->kps_data[kps_base + k * 2 + 1],
                            branch->kps_scale,
                            branch->kps_zp
                        );

                        /* Landmark position in model space
                         * Use anchor corner (w*stride), NOT center ((w+0.5)*stride) */
                        float lm_x = w * stride + kp_dx * stride;
                        float lm_y = h * stride + kp_dy * stride;

                        /* Map to original image space */
                        det.landmarks[k].x = (lm_x - p_x) * sc_x;
                        det.landmarks[k].y = (lm_y - p_y) * sc_y;
                    }

                    /* Store stride index for debugging */
                    det.stride_idx = s;

                    /* Add to per-stride detection list */
                    stride_dets[s].push_front(det);
                    stride_counts[s]++;
                }
            }
        }
    }

    /* Debug: print max score per stride to diagnose detection issues */
    xprintf("max_score: s8=%d s16=%d s32=%d /1000, best=%d(b%d), thresh=%d\n",
            (int)(max_score_per_stride[0] * 1000),
            (int)(max_score_per_stride[1] * 1000),
            (int)(max_score_per_stride[2] * 1000),
            (int)(max_score * 1000), max_score_branch,
            (int)(net->score_thresh * 1000));

    /*
     * Hierarchical NMS Strategy:
     * 1. Intra-stride NMS: Apply traditional IoU NMS within each stride
     * 2. Cross-stride suppression: Use center distance to merge detections from different strides
     */

    /* Step 1: Intra-stride NMS */
    for (int s = 0; s < SCRFD_NUM_STRIDES; s++) {
        if (stride_counts[s] > 1) {
            scrfd_nms(stride_dets[s], net->nms_thresh);
            /* Recount after NMS */
            stride_counts[s] = 0;
            for (auto& face : stride_dets[s]) {
                if (face.score > 0) stride_counts[s]++;
            }
        }
    }

    xprintf("[NMS] After intra-stride: s8=%d s16=%d s32=%d\n",
            stride_counts[0], stride_counts[1], stride_counts[2]);

    /* Step 2: Merge all stride detections */
    for (int s = 0; s < SCRFD_NUM_STRIDES; s++) {
        for (auto& face : stride_dets[s]) {
            if (face.score > 0) {
                dets.push_front(face);
                (*num_faces)++;
            }
        }
    }

    /* Step 3: Cross-stride suppression using center distance
     * Prefer finer stride (smaller bbox) when centers are close */
    if (*num_faces > 1) {
        cross_stride_suppress(dets, CENTER_DIST_THRESH_RATIO);

        /* Recount valid detections after cross-stride suppression */
        *num_faces = 0;
        for (auto& face : dets) {
            if (face.score > 0) {
                (*num_faces)++;
            }
        }
    }

    xprintf("[NMS] After cross-stride: %d faces\n", *num_faces);

    /* Step 4: Filter out oversized detections
     * Faces larger than MAX_FACE_RATIO of image dimension are likely false positives */
    float max_face_w = image_w * MAX_FACE_RATIO;
    float max_face_h = image_h * MAX_FACE_RATIO;
    int oversized_count = 0;

    for (auto& face : dets) {
        if (face.score > 0 && (face.bbox.w > max_face_w || face.bbox.h > max_face_h)) {
            xprintf("[SCRFD] Suppressed oversized: %.0fx%.0f (max=%.0fx%.0f)\n",
                    face.bbox.w, face.bbox.h, max_face_w, max_face_h);
            face.score = 0;
            oversized_count++;
        }
    }

    if (oversized_count > 0) {
        *num_faces -= oversized_count;
        xprintf("[NMS] After size filter: %d faces\n", *num_faces);
    }

    return dets;
}

void scrfd_nms(std::forward_list<scrfd_face>& dets, float thresh) {
    /* Sort detections by score in descending order */
    dets.sort([](const scrfd_face& a, const scrfd_face& b) {
        return a.score > b.score;
    });

    /* Apply NMS */
    for (auto it = dets.begin(); it != dets.end(); ++it) {
        if (it->score <= 0) continue;

        for (auto jt = std::next(it); jt != dets.end(); ++jt) {
            if (jt->score <= 0) continue;

            float iou = compute_iou(it->bbox, jt->bbox);
            if (iou > thresh) {
                jt->score = 0;  /* Suppress lower-score detection */
            }
        }
    }
}

/**
 * @brief Compute stability score for a detection relative to previous frame
 *
 * Higher score = more stable (consistent with previous frame)
 * Returns 0.0 if no previous frame or too different
 */
static float compute_stability_score(const scrfd_face& face, const scrfd_face& prev) {
    /* Center distance penalty */
    float dist = compute_center_distance(face.bbox, prev.bbox);
    float ref_size = (prev.bbox.w + prev.bbox.h) / 2.0f;
    float dist_ratio = dist / ref_size;

    if (dist_ratio > CENTER_DIST_THRESH_RATIO * 2) {
        return 0.0f;  /* Too far, not the same face */
    }

    /* Area change penalty */
    float area_curr = face.bbox.w * face.bbox.h;
    float area_prev = prev.bbox.w * prev.bbox.h;
    float area_ratio = (area_curr > area_prev) ?
                       (area_curr / area_prev) : (area_prev / area_curr);

    if (area_ratio > (1.0f + AREA_CHANGE_THRESH)) {
        return 0.0f;  /* Area changed too much */
    }

    /* Stability score: higher = more stable */
    float dist_score = 1.0f - (dist_ratio / (CENTER_DIST_THRESH_RATIO * 2));
    float area_score = 1.0f - ((area_ratio - 1.0f) / AREA_CHANGE_THRESH);

    return dist_score * 0.6f + area_score * 0.4f;
}

/**
 * @brief Apply EMA smoothing to bbox coordinates
 */
static void smooth_bbox(scrfd_face* face, const scrfd_face& prev, float alpha) {
    face->bbox.x = alpha * face->bbox.x + (1.0f - alpha) * prev.bbox.x;
    face->bbox.y = alpha * face->bbox.y + (1.0f - alpha) * prev.bbox.y;
    face->bbox.w = alpha * face->bbox.w + (1.0f - alpha) * prev.bbox.w;
    face->bbox.h = alpha * face->bbox.h + (1.0f - alpha) * prev.bbox.h;

    /* Also smooth landmarks */
    for (int i = 0; i < SCRFD_NUM_LANDMARKS; i++) {
        face->landmarks[i].x = alpha * face->landmarks[i].x + (1.0f - alpha) * prev.landmarks[i].x;
        face->landmarks[i].y = alpha * face->landmarks[i].y + (1.0f - alpha) * prev.landmarks[i].y;
    }
}

scrfd_face* scrfd_get_best_face(std::forward_list<scrfd_face>& dets, int min_size) {
    scrfd_face* best = nullptr;
    float best_combined_score = -1.0f;

    /* Count valid candidates */
    int valid_count = 0;
    for (auto& face : dets) {
        if (face.score > 0 && face.bbox.w >= min_size && face.bbox.h >= min_size) {
            valid_count++;
        }
    }

    for (auto& face : dets) {
        /* Skip invalid detections with tiny bbox */
        if (face.bbox.w < min_size || face.bbox.h < min_size) {
            continue;
        }
        if (face.score <= 0) {
            continue;
        }

        float combined_score;

        if (g_has_prev_face && valid_count > 1) {
            /* Multiple candidates: use stability-weighted score */
            float stability = compute_stability_score(face, g_prev_best_face);
            /* Combined score: 60% detection confidence + 40% temporal stability */
            combined_score = face.score * 0.6f + stability * 0.4f;
        } else {
            /* Single candidate or no history: use detection score only */
            combined_score = face.score;
        }

        if (combined_score > best_combined_score) {
            best_combined_score = combined_score;
            best = &face;
        }
    }

    /* Apply temporal smoothing and update history */
    if (best != nullptr) {
        if (g_has_prev_face) {
            float stability = compute_stability_score(*best, g_prev_best_face);
            if (stability > 0.3f) {
                /* Smooth bbox if reasonably stable */
                smooth_bbox(best, g_prev_best_face, SMOOTHING_ALPHA);
            }
        }

        /* Update previous frame state */
        g_prev_best_face = *best;
        g_has_prev_face = true;
    } else {
        /* No face detected, reset history after a few frames */
        static int no_face_count = 0;
        no_face_count++;
        if (no_face_count > 5) {
            g_has_prev_face = false;
            no_face_count = 0;
        }
    }

    return best;
}

void scrfd_free_dets(std::forward_list<scrfd_face>& dets) {
    dets.clear();
}

bool scrfd_validate_face(
    const scrfd_face* face,
    int min_size,
    int img_w,
    int img_h
) {
    if (face == nullptr) return false;

    /* Check minimum face size */
    if (face->bbox.w < min_size || face->bbox.h < min_size) {
        return false;
    }

    /* Check if face is within image bounds */
    if (face->bbox.x < 0 || face->bbox.y < 0 ||
        face->bbox.x + face->bbox.w > img_w ||
        face->bbox.y + face->bbox.h > img_h) {
        return false;
    }

    /* Check landmark positions (should be within bbox with margin) */
    float margin = 0.1f;  /* 10% margin */
    float x_min = face->bbox.x - face->bbox.w * margin;
    float x_max = face->bbox.x + face->bbox.w * (1.0f + margin);
    float y_min = face->bbox.y - face->bbox.h * margin;
    float y_max = face->bbox.y + face->bbox.h * (1.0f + margin);

    for (int i = 0; i < SCRFD_NUM_LANDMARKS; i++) {
        if (face->landmarks[i].x < x_min || face->landmarks[i].x > x_max ||
            face->landmarks[i].y < y_min || face->landmarks[i].y > y_max) {
            return false;
        }
    }

    /* Check eye positions (left eye should be to the left of right eye) */
    if (face->landmarks[SCRFD_LM_LEFT_EYE].x >= face->landmarks[SCRFD_LM_RIGHT_EYE].x) {
        return false;
    }

    /* Check nose position (should be between eyes vertically) */
    float eye_center_y = (face->landmarks[SCRFD_LM_LEFT_EYE].y +
                          face->landmarks[SCRFD_LM_RIGHT_EYE].y) / 2.0f;
    if (face->landmarks[SCRFD_LM_NOSE].y < eye_center_y) {
        return false;  /* Nose should be below eyes */
    }

    return true;
}
