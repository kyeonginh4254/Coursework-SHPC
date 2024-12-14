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
static cudaStream_t streams_knl[NUM_DEVICE], streams_mem[NUM_DEVICE];
static cudaEvent_t events_htod[NUM_DEVICE], events_dtoh[NUM_DEVICE];

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

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM][BK+1];
  __shared__ float Bs[BK][BN+1];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreadsBlocktile / BK;

  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreadsBlocktile / BN;

  float threadResults[TM][TN] = {0.0};

  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

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

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[threadRow * TM + i][dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx][threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM][resIdxN];
    }
  }
}
  

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // Divide work among available GPUs
  #pragma omp parallel for num_threads(num_device)
  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaSetDevice(i));
    // Copy B to device
    CUDA_CALL(cudaMemcpyAsync(b_d[i],
                              B,
                              K * N * sizeof(float),
                              cudaMemcpyHostToDevice,
                              streams_mem[i]));

    for (int j = 0; j < NUM_CHUNK; j++) {

      CUDA_CALL(cudaMemcpyAsync(a_d[i][j],
                                A + Mbegin[i][j] * K,
                                Msize[i][j] * K * sizeof(float),
                                cudaMemcpyHostToDevice,
                                streams_mem[i]));

      CUDA_CALL(cudaEventRecord(events_htod[i], streams_mem[i]));
      CUDA_CALL(cudaStreamWaitEvent(streams_knl[i], events_htod[i], 0));

      dim3 gridDim(CEIL_DIV(Msize[i][j], BM), CEIL_DIV(N, BN));
      dim3 blockDim(BM * BN / (TM * TN));
      matmul_kernel<<<gridDim, blockDim, 0, streams_knl[i]>>>(
                   a_d[i][j], b_d[i], c_d[i][j], Msize[i][j], N, K);
      
      CUDA_CALL(cudaEventRecord(events_htod[i], streams_knl[i]));
      CUDA_CALL(cudaStreamWaitEvent(streams_mem[i], events_htod[i], 0));

      CUDA_CALL(cudaMemcpyAsync(C + Mbegin[i][j] * N,
                                c_d[i][j],
                                Msize[i][j] * N * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                streams_mem[i]));
    }
  }

  // Ensure all operations are completed
  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamSynchronize(streams_mem[i]));
  }
}

void matmul_initialize(int M, int N, int K) {
  // Initialization
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
    CUDA_CALL(cudaStreamCreate(&streams_knl[i]));
    CUDA_CALL(cudaStreamCreate(&streams_mem[i]));
    CUDA_CALL(cudaEventCreate(&events_htod[i]));
    CUDA_CALL(cudaEventCreate(&events_dtoh[i]));

    for (int j = 0; j < NUM_CHUNK; j++) {
      CUDA_CALL(cudaMalloc(&a_d[i][j], (Mend[i][j] - Mbegin[i][j]) * K * sizeof(float)));
      CUDA_CALL(cudaMalloc(&c_d[i][j], (Mend[i][j] - Mbegin[i][j]) * N * sizeof(float)));
    }
  }
}

void matmul_finalize() {
  for (int i = 0; i < num_device; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaStreamDestroy(streams_knl[i]));
    CUDA_CALL(cudaStreamDestroy(streams_mem[i]));
    CUDA_CALL(cudaEventDestroy(events_htod[i]));
    CUDA_CALL(cudaEventDestroy(events_dtoh[i]));
    for (int j = 0; j < NUM_CHUNK; j++) {
      CUDA_CALL(cudaFree(a_d[i][j]));
      CUDA_CALL(cudaFree(c_d[i][j]));
    }
  }
}