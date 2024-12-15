#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

/* GLOBALS */

int num_device;

#define NUM_DEVICE 4
#define NUM_CHUNK 4
int Mbegin[NUM_DEVICE][NUM_CHUNK], Mend[NUM_DEVICE][NUM_CHUNK], Msize[NUM_DEVICE][NUM_CHUNK];
int M_device, M_chunk;

static float *a_d[NUM_DEVICE][NUM_CHUNK], *b_d[NUM_DEVICE], *c_d[NUM_DEVICE][NUM_CHUNK];
static cudaStream_t streams[NUM_DEVICE][NUM_CHUNK];

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define BM 64
#define BN 64
#define BK 8

#define TM 8
#define TN 8
#define TK 8
/*
이 커널은 hw5의 제 커널과 거의 유사하나,
필요한 데이터를 shared memory에서 참조하는 것이 아니라,
각 스레드가 점유하는 레지스터에 직접 적재하여 참조합니다.
따라서 속도에 큰 이점이 있습니다.

https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh 
*/

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint numThreadsBlocktile = BM * BN / (TM * TN);

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM][BK+1];
  __shared__ float Bs[BK][BN+1];

  float Cvalue[TM][TN] = {0.0};

  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreadsBlocktile / BK;

  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreadsBlocktile / BN;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[innerRowA + loadOffset][innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[innerRowB + loadOffset][innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint j = 0; j < BK; ++j) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[threadRow * TM + i][j];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[j][threadCol * TN + i];
      }
      for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
          Cvalue[i][j] += regM[i] * regN[j];
        }
      }
    }
    __syncthreads();
  }

  for (uint i = 0; i < TM; ++i) {
    for (uint j = 0; j < TN; ++j) {
      C[(threadRow * TM + i) * N + threadCol * TN + j] = Cvalue[i][j];
    }
  }
}
  

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  #pragma omp parallel for num_threads(num_device)
  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(b_d[i],
                              B,
                              K * N * sizeof(float),
                              cudaMemcpyHostToDevice,
                              streams[i][0]));

    CUDA_CALL(cudaStreamSynchronize(streams[i][0]))

    for (int j = 0; j < NUM_CHUNK; j++) {

      CUDA_CALL(cudaMemcpyAsync(a_d[i][j],
                                A + Mbegin[i][j] * K,
                                Msize[i][j] * K * sizeof(float),
                                cudaMemcpyHostToDevice,
                                streams[i][j]));

      dim3 gridDim(CEIL_DIV(Msize[i][j], BM), CEIL_DIV(N, BN));
      dim3 blockDim(BM * BN / (TM * TN));
      matmul_kernel<<<gridDim, blockDim, 0, streams[i][j]>>>(
                   a_d[i][j], b_d[i], c_d[i][j], Msize[i][j], N, K);

      CUDA_CALL(cudaMemcpyAsync(C + Mbegin[i][j] * N,
                                c_d[i][j],
                                Msize[i][j] * N * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                streams[i][j]));
    }
  }

  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaSetDevice(i));
    for (int j = 0; j < NUM_CHUNK; j++) {
      CUDA_CALL(cudaStreamSynchronize(streams[i][j]));
    }
  }
}

void matmul_initialize(int M, int N, int K) {
  num_device = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_device));

  M_device = CEIL_DIV(M, num_device);  

  for (int i = 0; i < num_device; i++) {
    M_chunk = CEIL_DIV(M_device, NUM_CHUNK);
    for (int j = 0; j < NUM_CHUNK; j++) {
      Mbegin[i][j] = M_device * i + M_chunk * j;
      Mend[i][j] = M_device * i + M_chunk * (j+1);
      if (j == NUM_CHUNK - 1) Mend[i][j] = M_device * (i+1);
      Msize[i][j] = Mend[i][j] - Mbegin[i][j];
    }
  }

  printf("Number of devices: %d\n", num_device);

  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    for (int j = 0; j < NUM_CHUNK; j++) {
      CUDA_CALL(cudaMalloc(&a_d[i][j], (Mend[i][j] - Mbegin[i][j]) * K * sizeof(float)));
      CUDA_CALL(cudaMalloc(&c_d[i][j], (Mend[i][j] - Mbegin[i][j]) * N * sizeof(float)));
      CUDA_CALL(cudaStreamCreate(&streams[i][j]));
    }
  }
}

void matmul_finalize() {
  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaFree(b_d[i]));
    for (int j = 0; j < NUM_CHUNK; j++) {
      CUDA_CALL(cudaFree(a_d[i][j]));
      CUDA_CALL(cudaFree(c_d[i][j]));
      CUDA_CALL(cudaStreamDestroy(streams[i][j]));
    }
  }
}