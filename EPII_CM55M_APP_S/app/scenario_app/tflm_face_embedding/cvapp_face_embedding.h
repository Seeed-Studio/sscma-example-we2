/*
 * cvapp_face_embedding.h
 *
 *  Created on: Dec 11, 2024
 *      Author: Face Embedding App (MobileFaceNet 128D)
 */

#ifndef APP_SCENARIO_APP_TFLM_FACE_EMBEDDING_CVAPP_FACE_EMBEDDING_H_
#define APP_SCENARIO_APP_TFLM_FACE_EMBEDDING_CVAPP_FACE_EMBEDDING_H_

#include "spi_protocol.h"
#include "face_embedding_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize face embedding models (detection + embedding)
 *
 * @param security_enable Enable security features
 * @param privilege_enable Enable privilege features
 * @param fd_model_addr Flash address of face detection model (SCRFD)
 * @param embedding_model_addr Flash address of face embedding model (MobileFaceNet)
 * @return int 0 on success, -1 on error
 */
int cv_face_embedding_init(bool security_enable, bool privilege_enable,
                            uint32_t fd_model_addr, uint32_t embedding_model_addr);

/**
 * @brief Run face embedding pipeline
 *
 * Performs face detection using SCRFD, aligns detected faces using 5-point
 * landmarks, and extracts 128-dimensional embeddings using MobileFaceNet.
 *
 * @param alg_result Basic detection result structure
 * @param embedding_result Face embedding result structure (128D)
 * @return int 0 on success, -1 on error
 */
int cv_face_embedding_run(struct_algoResult *alg_result,
                           face_embedding_msg_t *embedding_result);

/**
 * @brief Deinitialize face embedding
 *
 * @return int 0 on success, -1 on error
 */
int cv_face_embedding_deinit();

#ifdef __cplusplus
}
#endif

#endif /* APP_SCENARIO_APP_TFLM_FACE_EMBEDDING_CVAPP_FACE_EMBEDDING_H_ */
