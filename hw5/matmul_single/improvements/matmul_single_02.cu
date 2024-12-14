#include "matmul_single.h"
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

  // 최종으로 완성할 cell의 global index를 계산
  int gi = bi * TS + ti;
  int gj = bj * TS + tj;

  int row, col;
  // k를 적절히 tiling
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
static float *a_d;
static float *b_d;
static float *c_d;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Upload A and B matrix to every GPU
  CUDA_CALL(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // Launch kernel on every GPU
  dim3 blockDim(TS, TS);
  dim3 gridDim((N + TS - 1) / TS, (M + TS - 1) / TS);


  matmul_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  CUDA_CALL(cudaDeviceSynchronize());

  // Download C matrix from GPUs
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_initialize(int M, int N, int K) {
  
  int num_devices;
  // Only root process do something
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Allocate device memory 
  CUDA_CALL(cudaMalloc(&a_d, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&b_d, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&c_d, M * N * sizeof(float)));
}

void matmul_finalize() {

  // Free GPU memory
  CUDA_CALL(cudaFree(a_d));
  CUDA_CALL(cudaFree(b_d));
  CUDA_CALL(cudaFree(c_d));
}
