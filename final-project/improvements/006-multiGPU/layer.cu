#include "layer.h"
#include <cuda_runtime.h>
#include <algorithm>

#define BATCH_SIZE 1024
#define MINI_BATCH_SIZE 256
#define NUM_CHUNKS (BATCH_SIZE / MINI_BATCH_SIZE)
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/* Embedding
 * @param [in1]  in: [B, s]
 * @param [in2]   w: [NUM_VOCAB, H]
 * @param [out] out: [B, s, H]
 * 's' is the sequence length
 * 'H' is the embedding dimension
 */
__global__ void Embedding_Permute_Kernel(const int *__restrict__ in, 
                                        const float *__restrict__ w, 
                                        float *__restrict__ out,
                                        size_t B, size_t s, size_t H) {
  // 배치별 오프셋
  size_t b = blockIdx.z; // 배치 인덱스

  // 행 인덱스 (sequence dimension)와 열 인덱스 (embedding dimension)
  size_t i = blockIdx.y * blockDim.y + threadIdx.y; // sequence index
  size_t j = blockIdx.x * blockDim.x + threadIdx.x; // embedding index

  if (i < s && j < H) {
    // word_id 로드
    int word_id = in[b * s + i];

    // Embedding 데이터 로드
    float val = w[word_id * H + j];

    // Transpose: 결과를 out_permuted에 저장 [B, H, s]
    size_t out_idx = b * (H * s) + j * s + i;
    out[out_idx] = val;
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
__global__ void GetMax_Kernel(const float *__restrict__ global_in,
                              float *__restrict__ global_out, 
                              size_t B, size_t C, size_t s) {
  
  /* BLOCK INDEX */

  size_t IN_OFFSET = blockIdx.x * C * s + threadIdx.x * s;
  size_t OUT_OFFSET = blockIdx.x * C + threadIdx.x;

  float max_val = global_in[IN_OFFSET];
  for (size_t seq_idx = 1; seq_idx < s; ++seq_idx) {
    float val = global_in[IN_OFFSET + seq_idx];
    max_val = max_val > val ? max_val : val;
  }

  global_out[OUT_OFFSET] = max_val;

  }


void GetMax(float *d_in, float *d_out, size_t s, cudaStream_t stream) {
  size_t B = MINI_BATCH_SIZE;
  size_t C = 1024;
  dim3 grid(B);
  dim3 block(C);
  GetMax_Kernel<<<grid, block, 0, stream>>>(d_in, d_out, B, C, s);
}

/* Concat
 * @param [in1] in1: [B, N1]
 * @param [in2] in2: [B, N2]
 * @param [in3] in3: [B, N3]
 * @param [in4] in4: [B, N4]
 * @param [out] out: [B, N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */

__global__ void Concat_Kernel(const float *in1, const float *in2, const float *in3, const float *in4,
                              float *out, size_t B, size_t N1, size_t N2, size_t N3, size_t N4) {
  size_t total_N = N1 + N2 + N3 + N4;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_size = B * total_N;

  if (idx < total_size) {
    size_t b = idx / total_N;  // 배치 인덱스
    size_t i = idx % total_N;  // 배치 내 인덱스

    float val;
    if (i < N1) {
      val = in1[b * N1 + i];
    } else if (i < N1 + N2) {
      val = in2[b * N2 + (i - N1)];
    } else if (i < N1 + N2 + N3) {
      val = in3[b * N3 + (i - (N1 + N2))];
    } else {
      val = in4[b * N4 + (i - (N1 + N2 + N3))];
    }

    out[idx] = val;
  }
}

void Concat(float *d_in1, float *d_in2, float *d_in3, float *d_in4, float *d_out, cudaStream_t stream) {
  size_t B = MINI_BATCH_SIZE;
  size_t N1 = 1024;
  size_t N2 = 1024;
  size_t N3 = 1024;
  size_t N4 = 1024;
  size_t total_N = N1 + N2 + N3 + N4;
  size_t total_size = B * total_N;
  dim3 block(256);
  dim3 grid((total_size + block.x - 1) / block.x);

  Concat_Kernel<<<grid, block, 0, stream>>>(d_in1, d_in2, d_in3, d_in4, d_out, B, N1, N2, N3, N4);
  CHECK_CUDA(cudaPeekAtLastError());
}

/* Linear 
 * @param [in1]  in: [B, N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [B, M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */


#define BB 64
#define BM 64
#define BN 8
#define TB 8
#define TM 8

#define BB2 64
#define BN2 64

__global__ void Linear_ReLU_CUDA_Kernel(const float *__restrict__ global_in, 
                                        const float *__restrict__ global_w,
                                        float *__restrict__ global_out, 
                                        const float *__restrict__ global_b,
                                        size_t B, size_t M, size_t N) {
  
  /* THREAD INDICES */
  size_t threadIdx_x = threadIdx.x % (BM / TM);
  size_t threadIdx_y = threadIdx.x / (BM / TM);

  /* OFFSETS */
  size_t B_OFFSET = blockIdx.x * BB;
  size_t M_OFFSET = blockIdx.y * BM;
  size_t B_GLOBAL_OFFSET = B_OFFSET + TB * threadIdx_x;
  size_t M_GLOBAL_OFFSET = M_OFFSET + TM * threadIdx_y;

  /* SHARED MEMORY */
  __shared__ float shared_in[BB][BN + 1];
  __shared__ float shared_w[BM][BN + 1];

  /* REGISTER */
  float regIN[TB], regW[TM];

  /* ACCUMULATOR */
  float val[TB][TM] = {0.0f};
  float o;
  float bias;

  for (size_t N_OFFSET = 0; N_OFFSET < N; N_OFFSET += BN) {

    /* LOAD INPUT */
    for (size_t BB_OFFSET = 0; BB_OFFSET < BB; BB_OFFSET += TB) {
      shared_in[BB_OFFSET + threadIdx_x][threadIdx_y] = global_in[(B_OFFSET + BB_OFFSET + threadIdx_x) * N + N_OFFSET + threadIdx_y];
    }

    /* LOAD WEIGHT */
    for (size_t BM_OFFSET = 0; BM_OFFSET < BM; BM_OFFSET += TM) {
      shared_w[BM_OFFSET + threadIdx_x][threadIdx_y] = global_w[(M_OFFSET + BM_OFFSET + threadIdx_x) * N + N_OFFSET + threadIdx_y];
    }

    __syncthreads();

    /* CALCULATE */

    for (size_t j = 0; j < BN; ++j) {
      for (size_t i = 0; i < TB; ++i) {
        regIN[i] = shared_in[threadIdx_x * TB + i][j];
      }
      for (size_t i = 0; i < TM; ++i) {
        regW[i] = shared_w[threadIdx_y * TM + i][j];
      }
      for (size_t ii = 0; ii < TB; ++ii) {
        for (size_t jj = 0; jj < TM; ++jj) {
          val[ii][jj] += regIN[ii] * regW[jj];
        }
      }
    }
    __syncthreads();
  }

  for (size_t j = 0; j < TM; ++j) {
    bias = global_b[M_GLOBAL_OFFSET + j];
    for (size_t i = 0; i < TB; ++i) {
      o = val[i][j] + bias;
      o = o > 0 ? o : 0;
      global_out[(B_GLOBAL_OFFSET + i) * M + M_GLOBAL_OFFSET + j] = o;
    }
  }
}

void Linear_ReLU_CUDA(float *d_in, float *d_w, float *d_b, float *d_out, size_t N, size_t M, cudaStream_t stream) {
  size_t B = MINI_BATCH_SIZE; // batch size
  dim3 grid(CEIL_DIV(B, BB), CEIL_DIV(M, BM));
  dim3 block(BB * BM / (TB * TM));
  Linear_ReLU_CUDA_Kernel<<<grid, block, 0, stream>>>(d_in, d_w, d_out, d_b, B, M, N);
  CHECK_CUDA(cudaPeekAtLastError());
}

// Transpose kernel
// d_w:  shape [M, N]
// d_wt: shape [N, M]
// 여기서 M은 row 개수, N은 col 개수라고 가정
__global__ void transpose_kernel(const float *__restrict__ d_w, float *__restrict__ d_wt, size_t M, size_t N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // output의 col (input의 row)
    int y = blockIdx.y * blockDim.y + threadIdx.y; // output의 row (input의 col)
    
    // d_w: [M,N]에서 w[i,j] = w[i*N + j]
    // d_wt: [N,M]에서 w^T[j,i] = w[i*N + j]
    // d_wt[x * M + y] = d_w[y * N + x]
    
    if (x < N && y < M) {
        d_wt[x * M + y] = d_w[y * N + x];
    }
}

__global__ void Linear_CUDA_Kernel(const float *__restrict__ A, 
                                   const float *__restrict__ B,
                                   float *__restrict__ C, 
                                   const float *__restrict__ bias,
                                   size_t BATCH, size_t M, size_t N) {
    // A: [BATCH, N]
    // B: [N, M]
    // C: [BATCH, M]
    // bias: [M]
    const uint x = blockIdx.x * blockDim.x + threadIdx.x; // BATCH dim index
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; // M dim index

    if (x < BATCH && y < M) {
        float tmp = 0.0f;
        for (int i = 0; i < N; ++i) {
            tmp += A[x * N + i] * B[i * M + y];
        }
        tmp += bias[y];
        C[x * M + y] = tmp;
    }
}

void Linear_CUDA(float *d_in, float *d_w, float *d_b, float *d_out, size_t N, size_t M, cudaStream_t stream) {
  size_t B = MINI_BATCH_SIZE; // batch size
  float *d_wt = nullptr;
  CHECK_CUDA(cudaMalloc(&d_wt, N * M * sizeof(float)));
  {
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(N, 16), CEIL_DIV(M, 16));
    transpose_kernel<<<grid, block, 0, stream>>>(d_w, d_wt, M, N);
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  {
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(B, 16), CEIL_DIV(M, 16));
    Linear_CUDA_Kernel<<<grid, block, 0, stream>>>(d_in, d_wt, d_out, d_b, B, M, N);
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  CHECK_CUDA(cudaFree(d_wt));
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

#define BOS3 (BS - K3 + 1)
#define BOS5 (BS - K5 + 1)
#define BOS7 (BS - K7 + 1)
#define BOS9 (BS - K9 + 1)

#define conv_B 32

__global__ void Conv1d_3_ReLU_Kernel(const float * __restrict__ global_in, 
                                     const float * __restrict__ global_w,
                                     float * __restrict__ global_out, 
                                     const float * __restrict__ global_b,
                                     size_t global_B, size_t S, size_t C, size_t OC) {
  
  /* PARAMETERS */
  const size_t B = conv_B;
  const size_t K = K3;
  const size_t BOS = BOS3;
  const size_t OS = S - K + 1;
  
  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[B][TC][BS];
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
                                   size_t global_B, size_t S, size_t C, size_t OC) {

  /* PARAMETERS */
  const size_t B = conv_B;
  const size_t K = K5;
  const size_t BOS = BOS5;
  const size_t OS = S - K + 1;

  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[B][TC][BS];
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
                                   size_t global_B, size_t S, size_t C, size_t OC) {
  
  /* PARAMETERS */
  const size_t B = conv_B;
  const size_t K = K7;
  const size_t BOS = BOS7;
  const size_t OS = S - K + 1;
  
  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[B][TC][BS];
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
                                   size_t global_B, size_t S, size_t C, size_t OC) {

  /* PARAMETERS */
  const size_t B = conv_B;
  const size_t K = K9;
  const size_t BOS = BOS9;
  const size_t OS = S - K + 1;
  
  /* OFFSETS */
  size_t OC_OFFSET = blockIdx.x * BOC;
  size_t OS_OFFSET = blockIdx.y * BOS;
  
  /* SHARED MEMORY */
  __shared__ float shared_in[B][TC][BS];
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

void Conv1d(float *d_permute_a, float **d_conv_w, float **d_conv_b, float **d_conv_a, float **d_out, cudaStream_t *streams) {

  size_t OC = 1024;   
  size_t B = MINI_BATCH_SIZE;
  size_t C = 4096;
  size_t s = 16;
  
  size_t K[4] = {K3, K5, K7, K9};
  size_t BOS[4] = {BOS3, BOS5, BOS7, BOS9};

  size_t os[4];
  for (size_t i = 0; i < 4; i++) {
    os[i] = s - K[i] + 1;
  }
  for (size_t B_OFFSET = 0; B_OFFSET < B; B_OFFSET += conv_B) {

    dim3 grid3(CEIL_DIV(OC, BOC), CEIL_DIV(os[0], BOS[0]));
    dim3 block3(BOC, BS);
    Conv1d_3_ReLU_Kernel<<<grid3, block3, 0, streams[0]>>>(d_permute_a + B_OFFSET * C * s, d_conv_w[0], d_out[0], d_conv_b[0], conv_B, s, C, OC);

    dim3 grid5(CEIL_DIV(OC, BOC), CEIL_DIV(os[1], BOS[1]));
    dim3 block5(BOC, BS);
    Conv1d_5_ReLU_Kernel<<<grid5, block5, 0, streams[1]>>>(d_permute_a + B_OFFSET * C * s, d_conv_w[1], d_out[1], d_conv_b[1], conv_B, s, C, OC);

    dim3 grid7(CEIL_DIV(OC, BOC), CEIL_DIV(os[2], BOS[2]));
    dim3 block7(BOC, BS);
    Conv1d_7_ReLU_Kernel<<<grid7, block7, 0, streams[2]>>>(d_permute_a + B_OFFSET * C * s, d_conv_w[2], d_out[2], d_conv_b[2], conv_B, s, C, OC);

    dim3 grid9(CEIL_DIV(OC, BOC), CEIL_DIV(os[3], BOS[3]));
    dim3 block9(BOC, BS);
    Conv1d_9_ReLU_Kernel<<<grid9, block9, 0, streams[3]>>>(d_permute_a + B_OFFSET * C * s, d_conv_w[3], d_out[3], d_conv_b[3], conv_B, s, C, OC);
    
    for (size_t i = 0; i < 4; i++) {
      CHECK_CUDA(cudaMemcpyAsync(d_conv_a[i] + B_OFFSET * OC * os[i], d_out[i], conv_B * OC * os[i] * sizeof(float), cudaMemcpyDeviceToDevice, streams[i]));
    }
  }
}