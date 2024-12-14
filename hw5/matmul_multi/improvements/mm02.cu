#include "matmul_multi.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define TS 16

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define MAX_NUM_GPU 4
int num_devices = 0;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];
  
  int bi = blockIdx.y;
  int ti = threadIdx.y;
  int bj = blockIdx.x;
  int tj = threadIdx.x;

  float Cvalue = 0.0f;
  const int numTiles = (K+TS-1)/TS;

  int gi = bi * TS + ti;
  int gj = bj * TS + tj;

  int row, col;

  for (int bk = 0; bk < numTiles; bk++) {
    row = gi;
    col = bk * TS + tj;
    if (row < M && col < K) {
      Asub[ti][tj] = A[row * K + col];
    } else {
      Asub[ti][tj] = 0;
    }

    row = bk * TS + ti;
    col = gj;
    if (row < K && col < N) {
      Bsub[ti][tj] = B[row * N + col];
    } else {
      Bsub[ti][tj] = 0;
    }

    __syncthreads();

    for (int k = 0; k < TS; k++) {
      Cvalue += Asub[ti][k] * Bsub[k][tj];
    }

    __syncthreads();

  }

  if (gi < M && gj < N) {
    C[gi * N + gj] = Cvalue;
  }
}

// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  
  cudaStream_t streams[num_devices];
  
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamCreate(&streams[i]));
  }

  for (int i = 0; i < num_devices; i++) {
    int Mlocal = Mend[i] - Mbegin[i];
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(a_d[i],
                         A + Mbegin[i] * K,
                         Mlocal * K * sizeof(float),
                         cudaMemcpyHostToDevice,
                         streams[i]));
    CUDA_CALL(cudaMemcpyAsync(b_d[i],
                         B,
                         K * N * sizeof(float),
                         cudaMemcpyHostToDevice,
                         streams[i]));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    int Mlocal = Mend[i] - Mbegin[i];
    dim3 blockDim(TS, TS);
    dim3 gridDim((N + TS - 1) / TS, (Mlocal + TS - 1) / TS);

    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(a_d[i], b_d[i], c_d[i], M, N, K);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpyAsync(C + Mbegin[i] * N,
                         c_d[i],
                         (Mend[i] - Mbegin[i]) * N * sizeof(float),
                         cudaMemcpyDeviceToHost,
                         streams[i]));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamSynchronize(streams[i])); // 해당 스트림 대기
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  }
}

void matmul_initialize(int M, int N, int K) {

  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("Using %d devices\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // Try printing more detailed information here
    printf("GPU %d: %s\n", i, prop.name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Setup problem size for each GPU
  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (M / num_devices) * i;
    Mend[i] = (M / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = M;

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }
}

void matmul_finalize() {

  // Free all GPU memory
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
  }
}
