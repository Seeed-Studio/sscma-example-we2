/*
 * tflm_face_embedding.h
 *
 *  Created on: Dec 11, 2024
 *      Author: Face Embedding App (MobileFaceNet 128D)
 */

#ifndef SCENARIO_TFLM_FACE_EMBEDDING_
#define SCENARIO_TFLM_FACE_EMBEDDING_

#define APP_BLOCK_FUNC() do{ \
	__asm volatile("b    .");\
	}while(0)

typedef enum
{
	APP_STATE_ALLON_FD_FL,
	APP_STATE_ALLON_PL,
	APP_STATE_ALLON_FD_FM,
	APP_STATE_ALLON_FD_FL_EL_9_POINT,
}APP_STATE_E;

int app_main(void);
void SetPSPDNoVid();
#endif /* SCENARIO_TFLM_FACE_EMBEDDING_ */
