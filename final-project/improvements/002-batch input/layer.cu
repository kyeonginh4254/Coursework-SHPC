#include "layer.h"
#include <cuda_runtime.h>
#include <algorithm>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCK_OC 16
#define BLOCK_OS 8
#define BATCH_SIZE 4

/* Embedding
 * @param [in1]  in: [B, s]
 * @param [in2]   w: [NUM_VOCAB, H]
 * @param [out] out: [B, s, H]
 * 's' is the sequence length
 * 'H' is the embedding dimension
 */
void Embedding(int *in, Tensor* w, Tensor *out) {
  size_t B = out->shape[0];
  size_t s = out->shape[1];
  size_t H = out->shape[2];

  for (size_t b = 0; b < B; b++) {
    for (size_t i = 0; i < s; i++) {
      int word_id = in[b * s + i];
      for (size_t j = 0; j < H; j++) {
        out->buf[b * (s * H) + i * H + j] = w->buf[word_id * H + j];
      }
    }
  }
}

/* Permute
 * @param [in]   in: [B, M, N]
 * @param [out] out: [B, N, M]
 */
void Permute(Tensor *in, Tensor *out) {
  size_t B = in->shape[0];
  size_t M = in->shape[1];
  size_t N = in->shape[2];

  for (size_t b = 0; b < B; b++) {
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        out->buf[b * (N * M) + j * M + i] = in->buf[b * (M * N) + i * N + j];
      }
    }
  }
}

/* GetMax
 * @param [in]   in: [B, C, s]
 * @param [out] out: [B, C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(Tensor *in, Tensor *out) {
  size_t B = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];

  for (size_t b = 0; b < B; b++) {
    for (size_t i = 0; i < C; i++) {
      float max_val = in->buf[b * (C * s) + i * s];
      for (size_t j = 1; j < s; j++) {
        float val = in->buf[b * (C * s) + i * s + j];
        if (val > max_val) max_val = val;
      }
      out->buf[b * C + i] = max_val;
    }
  }
}

/* Concat
 * @param [in1] in1: [B, N1]
 * @param [in2] in2: [B, N2]
 * @param [in3] in3: [B, N3]
 * @param [in4] in4: [B, N4]
 * @param [out] out: [B, N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {
  size_t B = in1->shape[0];
  size_t N1 = in1->shape[1];
  size_t N2 = in2->shape[1];
  size_t N3 = in3->shape[1];
  size_t N4 = in4->shape[1];

  for (size_t b = 0; b < B; b++) {
    for (size_t i = 0; i < N1; i++) {
      out->buf[b * (N1 + N2 + N3 + N4) + i] = in1->buf[b * N1 + i];
    }
    for (size_t i = 0; i < N2; i++) {
      out->buf[b * (N1 + N2 + N3 + N4) + N1 + i] = in2->buf[b * N2 + i];
    }
    for (size_t i = 0; i < N3; i++) {
      out->buf[b * (N1 + N2 + N3 + N4) + N1 + N2 + i] = in3->buf[b * N3 + i];
    }
    for (size_t i = 0; i < N4; i++) {
      out->buf[b * (N1 + N2 + N3 + N4) + N1 + N2 + N3 + i] = in4->buf[b * N4 + i];
    }
  }
}

/* Linear 
 * @param [in1]  in: [B, N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [B, M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t B = in->shape[0]; // 배치 크기
  size_t N = in->shape[1]; // 입력 특성 수
  size_t M = w->shape[0];  // 출력 특성 수

  for (size_t b_idx = 0; b_idx < B; b_idx++) {
    for (size_t i = 0; i < M; i++) {
      float val = 0.f;
      for (size_t j = 0; j < N; j++) {
        val += in->buf[b_idx * N + j] * w->buf[i * N + j];
      }
      out->buf[b_idx * M + i] = val + b->buf[i];
    }
  }
}


/* Conv1D + ReLU
 * @param [in1]  in: [B, C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [B, OC, os]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *          = (s - K + 2 * 0) / 1 + 1
 *          = s - K + 1
 *
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
__global__ void Conv1d_ReLU_Kernel(const float * __restrict__ global_in, 
                                   const float * __restrict__ global_w,
                                   float * __restrict__ global_out, 
                                   const float * __restrict__ global_b,
                                   size_t B, size_t s, size_t C, size_t OC, size_t K) {

    // 블록 인덱스로부터 현재 타일의 OC, OS 시작점 계산
    size_t oc_start = blockIdx.x * BLOCK_OC;
    size_t os_start = blockIdx.y * BLOCK_OS;

    // threadIdx.x 로부터 현재 스레드가 담당하는 oc, os 인덱스 계산
    size_t linear_t = threadIdx.x; 
    size_t index_OC = linear_t / BLOCK_OS;
    size_t index_os = linear_t % BLOCK_OS;

    size_t oc = oc_start + index_OC;
    size_t pos = os_start + index_os;

    // 출력 sequence 길이
    size_t os = s - K + 1;

    // 범위 밖 스레드는 연산 불필요
    if (oc >= OC || pos >= os) {
        return;
    }

    // 결과를 배치별로 저장할 배열
    float val[64]; // B가 최대 64 이하라 가정 (원하는 B에 맞게 조정)
    for (size_t b = 0; b < B; b++) {
        val[b] = 0.0f;
    }

    // convolution 연산
    for (size_t c = 0; c < C; c++) {
        for (size_t k = 0; k < K; k++) {
            // weight index
            float w_val = global_w[oc * C * K + c * K + k];

            for (size_t b = 0; b < B; b++) {
                float in_val = global_in[b * C * s + c * s + (pos + k)];
                val[b] += in_val * w_val;
            }
        }
    }

    // bias & ReLU
    float bias_val = global_b[oc];
    for (size_t b = 0; b < B; b++) {
        float o = val[b] + bias_val;
        val[b] = o > 0.f ? o : 0.f;
    }

    // 결과 쓰기
    for (size_t b = 0; b < B; b++) {
        global_out[b * OC * os + oc * os + pos] = val[b];
    }
}

/* Conv1D + ReLU using CUDA
 * in:  [B, C, s]
 * w:   [OC, C, K]
 * b:   [OC]
 * out: [B, OC, os]
 */
void Conv1D_ReLU_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
    size_t B = in->shape[0];
    size_t C = in->shape[1];
    size_t s = in->shape[2];

    size_t OC = w->shape[0];
    size_t K = w->shape[2];
    size_t os = s - K + 1;

    float *d_in, *d_w, *d_b, *d_out;

    CHECK_CUDA(cudaMalloc(&d_in, B * C * s * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, OC * C * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, OC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, B * OC * os * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, in->buf, B * C * s * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w->buf, OC * C * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b->buf, OC * sizeof(float), cudaMemcpyHostToDevice));

    // grid, block 설정
    dim3 grid(CEIL_DIV(OC, BLOCK_OC), CEIL_DIV(os, BLOCK_OS));
    dim3 block(BLOCK_OC * BLOCK_OS);

    Conv1d_ReLU_Kernel<<<grid, block>>>(d_in, d_w, d_out, d_b, B, s, C, OC, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out->buf, d_out, B * OC * os * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}

/* ReLU CUDA kernel */
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  float *d_inout;
  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float), 
                        cudaMemcpyHostToDevice));

  ReLU_Kernel<<<(N + 255) / 256, 256>>>(d_inout, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_inout));
}