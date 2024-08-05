
main.c
``` c
#ifdef __cplusplus
extern "C" {
#endif

// 표준 C 라이브러리와 STM32 관련 헤더 파일 포함
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

// AI 관련 헤더 파일
#include "ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "sin_model.h"
#include "sin_model_data.h"

// 모델 입력 및 출력을 위한 버퍼 정의
#if !defined(AI_SIN_MODEL_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_SIN_MODEL_IN_1_SIZE_BYTES]; // 입력 버퍼 정의
ai_i8* data_ins[AI_SIN_MODEL_IN_NUM] = {data_in_1}; // 입력 버퍼 배열
#else
ai_i8* data_ins[AI_SIN_MODEL_IN_NUM] = {NULL};
#endif

#if !defined(AI_SIN_MODEL_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_SIN_MODEL_OUT_1_SIZE_BYTES]; // 출력 버퍼 정의
ai_i8* data_outs[AI_SIN_MODEL_OUT_NUM] = {data_out_1}; // 출력 버퍼 배열
#else
ai_i8* data_outs[AI_SIN_MODEL_OUT_NUM] = {NULL};
#endif

AI_ALIGNED(32) static uint8_t pool0[AI_SIN_MODEL_DATA_ACTIVATION_1_SIZE]; // 활성화 버퍼

ai_handle data_activations0[] = {pool0}; // 활성화 데이터 핸들

// AI 모델 객체 및 입력/출력 버퍼
static ai_handle sin_model = AI_HANDLE_NULL;
static ai_buffer* ai_input;
static ai_buffer* ai_output;

// 오류 로깅 함수
static void ai_log_err(const ai_error err, const char *fct)
{
  // 오류 발생 시 콘솔에 메시지 출력
  if (fct)
    printf("Error in %s - type=0x%02x code=0x%02x\n", fct, err.type, err.code);
  else
    printf("Error - type=0x%02x code=0x%02x\n", err.type, err.code);

  do {} while (1); // 무한 루프로 오류 상태 유지
}

// 모델 초기화 및 부트스트랩
static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  // 모델 인스턴스 생성 및 초기화
  err = ai_sin_model_create_and_init(&sin_model, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_sin_model_create_and_init");
    return -1;
  }

  // 입력 및 출력 버퍼 할당
  ai_input = ai_sin_model_inputs_get(sin_model, NULL);
  ai_output = ai_sin_model_outputs_get(sin_model, NULL);

  // 입력 및 출력 버퍼 설정
#if defined(AI_SIN_MODEL_INPUTS_IN_ACTIVATIONS)
  for (int idx = 0; idx < AI_SIN_MODEL_IN_NUM; idx++) {
    data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx = 0; idx < AI_SIN_MODEL_IN_NUM; idx++) {
    ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_SIN_MODEL_OUTPUTS_IN_ACTIVATIONS)
  for (int idx = 0; idx < AI_SIN_MODEL_OUT_NUM; idx++) {
    data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx = 0; idx < AI_SIN_MODEL_OUT_NUM; idx++) {
    ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0; // 성공적인 초기화
}

// 주요 초기화 함수
void MX_X_CUBE_AI_Init(void)
{
  printf("\nAI model initialization\n");
  ai_boostrap(data_activations0); // 모델 부트스트랩
}

int main(void)
{
  // 시스템 초기화
  HAL_Init();
  SystemClock_Config();

  // 주변장치 초기화
  MX_GPIO_Init();
  MX_CRC_Init();
  MX_USART3_UART_Init();

  // AI 모델 초기화
  MX_X_CUBE_AI_Init();

  // 무한 루프 내에서 모델 실행
  while (1)
  {
    float input_value = 1.0f; // 모델 입력값으로 1 설정
    *((float*)data_ins[0]) = input_value; // 입력 버퍼에 값 할당

    // 모델 실행
    if (ai_sin_model_run(sin_model, ai_input, ai_output) != 1) {
      // 실행 실패 처리
      ai_log_err(ai_sin_model_get_error(sin_model), "ai_sin_model_run");
      return -1;
    }

    // 출력 결과 추출
    float output_value = *((float*)data_outs[0]);
    // 결과 출력 (여기서는 추가적인 출력 코드가 필요)
  }
}

```