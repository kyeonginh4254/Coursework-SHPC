#include "layer.h"
#include <cuda_runtime.h>
#include <algorithm>

#define BATCH_SIZE 64
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

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

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8
#define TK 8

__global__ void Linear_ReLU_CUDA_Kernel(const float *__restrict__ A, 
                                        const float *__restrict__ B,
                                        float *__restrict__ C, 
                                        const float *__restrict__ bias,
                                        size_t M, size_t N, size_t K, int relu) {
    // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    tmp += bias[y];
    if (relu) {
      C[x * N + y] = tmp > 0 ? tmp : 0;
    } else {
      C[x * N + y] = tmp;
    }
  }
}

void Linear_ReLU_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, int relu) {
  size_t B = in->shape[0]; // batch size
  size_t N = in->shape[1]; // input feature size
  size_t M = w->shape[0];  // output feature size

  // w: [M, N]를 w^T: [N, M]로 전치
  float *WT = new float[N * M];
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      WT[j * M + i] = w->buf[i * N + j];
    }
  }

  float *d_in, *d_wt, *d_b, *d_out;

  CHECK_CUDA(cudaMalloc(&d_in, B * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_wt, N * M * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, M * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, B * M * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_in, in->buf, B * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_wt, WT, N * M * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b->buf, M * sizeof(float), cudaMemcpyHostToDevice));

  delete[] WT;

  // matmul 형태: A: [B,N], B: [N,M], C: [B,M]
  // 커널 파라미터: M=B, N=M, K=N
  dim3 grid(CEIL_DIV(B, 16), CEIL_DIV(M, 16), 1);
  dim3 block(16, 16, 1);

  Linear_ReLU_CUDA_Kernel<<<grid, block>>>(d_in, d_wt, d_out, d_b, B, M, N, relu);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, d_out, B * M * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_wt));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
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


#define BOC 8
#define TC 8

#define K3 3
#define K5 5
#define K7 7
#define K9 9

#define BS 16

#define BOS3 BS - K3 + 1
#define BOS5 BS - K5 + 1
#define BOS7 BS - K7 + 1
#define BOS9 BS - K9 + 1



__global__ void Conv1d_3_ReLU_Kernel(const float * __restrict__ global_in, 
                                   const float * __restrict__ global_w,
                                   float * __restrict__ global_out, 
                                   const float * __restrict__ global_b,
                                   size_t global_B, size_t S, size_t C, size_t OC, size_t global_K) {
  
  /* PARAMETERS */
  const size_t B = BATCH_SIZE;
  const size_t K = K3;
  const size_t BOS = BOS3;
  const size_t OS = S - K + 1;
  
  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[BATCH_SIZE][TC][BS];
  __shared__ float shared_w[BOC][TC][K];

  /* REGISTER */
  float regW[TC], regIN[TC];

  /* ACCUMULATOR */
  float val[B] = {0.0f};
  float o;
  
  for (size_t C_OFFSET = 0; C_OFFSET < C; C_OFFSET += TC) {

    /* LOAD INPUT */
    for (size_t b = 0; b < B; b++) {
      if (OS_OFFSET + threadIdx.y < S) {
        shared_in[b][threadIdx.x][threadIdx.y] = global_in[b * C * S + (C_OFFSET + threadIdx.x) * S + OS_OFFSET + threadIdx.y];
      } else {
        shared_in[b][threadIdx.x][threadIdx.y] = 0.0f;
      }
    }

    /* LOAD WEIGHT */
    if (threadIdx.y < K) {
      for (size_t boc = 0; boc < BOC; boc++) {
        shared_w[boc][threadIdx.x][threadIdx.y] = global_w[(OC_OFFSET + boc) * C * K + (C_OFFSET + threadIdx.x) * K + threadIdx.y];
      }
    }

    __syncthreads();

    /* CALCULATE */
    if (threadIdx.y < BOS) {
      for (size_t k = 0; k < K; k++) {
        /* LOAD WEIGHT TO REGISTER */
        for (size_t tc = 0; tc < TC; tc++) {
          regW[tc] = shared_w[threadIdx.x][tc][k];
        }
        for (size_t b = 0; b < B; b++) {
          /* LOAD INPUT TO REGISTER */
          for (size_t tc = 0; tc < TC; tc++) {
            regIN[tc] = shared_in[b][tc][threadIdx.y + k];
          }
          /* CALCULATE */
          for (size_t tc = 0; tc < TC; tc++) {
            val[b] += regW[tc] * regIN[tc];
          }
        }
      }
    }

    __syncthreads();

  }

  if (OS_OFFSET + threadIdx.y < OS) {
    for (size_t b = 0; b < B; b++) {
      o = val[b] + global_b[OC_OFFSET + threadIdx.x];
      o = o > 0 ? o : 0;
      global_out[b * OC * OS + (OC_OFFSET + threadIdx.x) * OS + OS_OFFSET + threadIdx.y] = o;
    }
  }
}

__global__ void Conv1d_5_ReLU_Kernel(const float * __restrict__ global_in, 
                                   const float * __restrict__ global_w,
                                   float * __restrict__ global_out, 
                                   const float * __restrict__ global_b,
                                   size_t global_B, size_t S, size_t C, size_t OC, size_t global_K) {

  /* PARAMETERS */
  const size_t B = BATCH_SIZE;
  const size_t K = K5;
  const size_t BOS = BOS5;
  const size_t OS = S - K + 1;

  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[BATCH_SIZE][TC][BS];
  __shared__ float shared_w[BOC][TC][K];

  /* REGISTER */
  float regW[TC], regIN[TC];

  /* ACCUMULATOR */
  float val[B] = {0.0f};
  float o;
  
  for (size_t C_OFFSET = 0; C_OFFSET < C; C_OFFSET += TC) {

    /* LOAD INPUT */
    for (size_t b = 0; b < B; b++) {
      if (OS_OFFSET + threadIdx.y < S) {
        shared_in[b][threadIdx.x][threadIdx.y] = global_in[b * C * S + (C_OFFSET + threadIdx.x) * S + OS_OFFSET + threadIdx.y];
      } else {
        shared_in[b][threadIdx.x][threadIdx.y] = 0.0f;
      }
    }

    /* LOAD WEIGHT */
    if (threadIdx.y < K) {
      for (size_t boc = 0; boc < BOC; boc++) {
        shared_w[boc][threadIdx.x][threadIdx.y] = global_w[(OC_OFFSET + boc) * C * K + (C_OFFSET + threadIdx.x) * K + threadIdx.y];
      }
    }

    __syncthreads();

    /* CALCULATE */
    if (threadIdx.y < BOS) {
      for (size_t k = 0; k < K; k++) {
        /* LOAD WEIGHT TO REGISTER */
        for (size_t tc = 0; tc < TC; tc++) {
          regW[tc] = shared_w[threadIdx.x][tc][k];
        }
        for (size_t b = 0; b < B; b++) {
          /* LOAD INPUT TO REGISTER */
          for (size_t tc = 0; tc < TC; tc++) {
            regIN[tc] = shared_in[b][tc][threadIdx.y + k];
          }
          /* CALCULATE */
          for (size_t tc = 0; tc < TC; tc++) {
            val[b] += regW[tc] * regIN[tc];
          }
        }
      }
    }

    __syncthreads();

  }

  if (OS_OFFSET + threadIdx.y < OS) {
    for (size_t b = 0; b < B; b++) {
      o = val[b] + global_b[OC_OFFSET + threadIdx.x];
      o = o > 0 ? o : 0;
      global_out[b * OC * OS + (OC_OFFSET + threadIdx.x) * OS + OS_OFFSET + threadIdx.y] = o;
    }
  }
}

__global__ void Conv1d_7_ReLU_Kernel(const float * __restrict__ global_in, 
                                   const float * __restrict__ global_w,
                                   float * __restrict__ global_out, 
                                   const float * __restrict__ global_b,
                                   size_t global_B, size_t S, size_t C, size_t OC, size_t global_K) {
  
  /* PARAMETERS */
  const size_t B = BATCH_SIZE;
  const size_t K = K7;
  const size_t BOS = BOS7;
  const size_t OS = S - K + 1;
  
  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[BATCH_SIZE][TC][BS];
  __shared__ float shared_w[BOC][TC][K];

  /* REGISTER */
  float regW[TC], regIN[TC];

  /* ACCUMULATOR */
  float val[B] = {0.0f};
  float o;
  
  for (size_t C_OFFSET = 0; C_OFFSET < C; C_OFFSET += TC) {

    /* LOAD INPUT */
    for (size_t b = 0; b < B; b++) {
      if (OS_OFFSET + threadIdx.y < S) {
        shared_in[b][threadIdx.x][threadIdx.y] = global_in[b * C * S + (C_OFFSET + threadIdx.x) * S + OS_OFFSET + threadIdx.y];
      } else {
        shared_in[b][threadIdx.x][threadIdx.y] = 0.0f;
      }
    }

    /* LOAD WEIGHT */
    if (threadIdx.y < K) {
      for (size_t boc = 0; boc < BOC; boc++) {
        shared_w[boc][threadIdx.x][threadIdx.y] = global_w[(OC_OFFSET + boc) * C * K + (C_OFFSET + threadIdx.x) * K + threadIdx.y];
      }
    }

    __syncthreads();

    /* CALCULATE */
    if (threadIdx.y < BOS) {
      for (size_t k = 0; k < K; k++) {
        /* LOAD WEIGHT TO REGISTER */
        for (size_t tc = 0; tc < TC; tc++) {
          regW[tc] = shared_w[threadIdx.x][tc][k];
        }
        for (size_t b = 0; b < B; b++) {
          /* LOAD INPUT TO REGISTER */
          for (size_t tc = 0; tc < TC; tc++) {
            regIN[tc] = shared_in[b][tc][threadIdx.y + k];
          }
          /* CALCULATE */
          for (size_t tc = 0; tc < TC; tc++) {
            val[b] += regW[tc] * regIN[tc];
          }
        }
      }
    }

    __syncthreads();
  }

  if (OS_OFFSET + threadIdx.y < OS) {
    for (size_t b = 0; b < B; b++) {
      o = val[b] + global_b[OC_OFFSET + threadIdx.x];
      o = o > 0 ? o : 0;
      global_out[b * OC * OS + (OC_OFFSET + threadIdx.x) * OS + OS_OFFSET + threadIdx.y] = o;
    }
  }
}

__global__ void Conv1d_9_ReLU_Kernel(const float * __restrict__ global_in, 
                                   const float * __restrict__ global_w,
                                   float * __restrict__ global_out, 
                                   const float * __restrict__ global_b,
                                   size_t global_B, size_t S, size_t C, size_t OC, size_t global_K) {

  /* PARAMETERS */
  const size_t B = BATCH_SIZE;
  const size_t K = K9;
  const size_t BOS = BOS9;
  const size_t OS = S - K + 1;
  
  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[BATCH_SIZE][TC][BS];
  __shared__ float shared_w[BOC][TC][K];

  /* REGISTER */
  float regW[TC], regIN[TC];

  /* ACCUMULATOR */
  float val[B] = {0.0f};
  float o;
  
  for (size_t C_OFFSET = 0; C_OFFSET < C; C_OFFSET += TC) {

    /* LOAD INPUT */
    for (size_t b = 0; b < B; b++) {
      if (OS_OFFSET + threadIdx.y < S) {
        shared_in[b][threadIdx.x][threadIdx.y] = global_in[b * C * S + (C_OFFSET + threadIdx.x) * S + OS_OFFSET + threadIdx.y];
      } else {
        shared_in[b][threadIdx.x][threadIdx.y] = 0.0f;
      }
    }

    /* LOAD WEIGHT */
    if (threadIdx.y < K) {
      for (size_t boc = 0; boc < BOC; boc++) {
        shared_w[boc][threadIdx.x][threadIdx.y] = global_w[(OC_OFFSET + boc) * C * K + (C_OFFSET + threadIdx.x) * K + threadIdx.y];
      }
    }

    __syncthreads();

    /* CALCULATE */
    if (threadIdx.y < BOS) {
      for (size_t k = 0; k < K; k++) {
        /* LOAD WEIGHT TO REGISTER */
        for (size_t tc = 0; tc < TC; tc++) {
          regW[tc] = shared_w[threadIdx.x][tc][k];
        }
        for (size_t b = 0; b < B; b++) {
          /* LOAD INPUT TO REGISTER */
          for (size_t tc = 0; tc < TC; tc++) {
            regIN[tc] = shared_in[b][tc][threadIdx.y + k];
          }
          /* CALCULATE */
          for (size_t tc = 0; tc < TC; tc++) {
            val[b] += regW[tc] * regIN[tc];
          }
        }
      }
    }

    __syncthreads();

  }

  if (OS_OFFSET + threadIdx.y < OS) {
    for (size_t b = 0; b < B; b++) {
      o = val[b] + global_b[OC_OFFSET + threadIdx.x];
      o = o > 0 ? o : 0;
      global_out[b * OC * OS + (OC_OFFSET + threadIdx.x) * OS + OS_OFFSET + threadIdx.y] = o;
    }
  }
}

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

    if (K == 3) {
      dim3 grid(CEIL_DIV(OC, BOC), CEIL_DIV(os, BOS3));
      dim3 block(BOC, BS);
      Conv1d_3_ReLU_Kernel<<<grid, block>>>(d_in, d_w, d_out, d_b, B, s, C, OC, K);
    } else if (K == 5) {
      dim3 grid(CEIL_DIV(OC, BOC), CEIL_DIV(os, BOS5));
      dim3 block(BOC, BS);
      Conv1d_5_ReLU_Kernel<<<grid, block>>>(d_in, d_w, d_out, d_b, B, s, C, OC, K);
    } else if (K == 7) {
      dim3 grid(CEIL_DIV(OC, BOC), CEIL_DIV(os, BOS7));
      dim3 block(BOC, BS);
      Conv1d_7_ReLU_Kernel<<<grid, block>>>(d_in, d_w, d_out, d_b, B, s, C, OC, K);
    } else if (K == 9) {
      dim3 grid(CEIL_DIV(OC, BOC), CEIL_DIV(os, BOS9));
      dim3 block(BOC, BS);
      Conv1d_9_ReLU_Kernel<<<grid, block>>>(d_in, d_w, d_out, d_b, B, s, C, OC, K);
    } else {
      printf("something gets wrong...\n");
    }
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaMemcpy(out->buf, d_out, B * OC * os * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}