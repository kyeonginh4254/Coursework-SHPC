#include "matmul_single.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define TS 64
#define WPT 2
#define RTS TS/WPT

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  const int row = threadIdx.y; // Local row index within the block
  const int col = threadIdx.x; // Local column index within the block
  const int globalRow = RTS * blockIdx.y + row; // Global row index
  const int globalCol = RTS * blockIdx.x + col; // Global column index

  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  float Cvalue[WPT][WPT] = {0.0f};

  const int numTiles = (K + TS - 1) / TS;

  for (int t = 0; t < numTiles; ++t) {
    int tiledRow = RTS * t + row;
    int tiledCol = RTS * t + col;

        // Load data into shared memory
        for (int w1 = 0; w1 < WPT; ++w1) {
            for (int w2 = 0; w2 < WPT; ++w2) {
                int rowA = globalRow * WPT + w1;
                int colA = tiledCol * WPT + w2;
                if (rowA < M && colA < K) {
                    Asub[row * WPT + w1][col * WPT + w2] = A[rowA * K + colA];
                } else {
                    Asub[row * WPT + w1][col * WPT + w2] = 0.0f;
                }

                int rowB = tiledRow * WPT + w1;
                int colB = globalCol * WPT + w2;
                if (rowB < K && colB < N) {
                    Bsub[row * WPT + w1][col * WPT + w2] = B[rowB * N + colB];
                } else {
                    Bsub[row * WPT + w1][col * WPT + w2] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial results
        for (int k = 0; k < TS; ++k) {
            for (int w1 = 0; w1 < WPT; ++w1) {
                for (int w2 = 0; w2 < WPT; ++w2) {
                    Cvalue[w1][w2] += Asub[row * WPT + w1][k] * Bsub[k][col * WPT + w2];
                }
            }
        }

        __syncthreads();
    }

    // Write the final result to global memory
    for (int w1 = 0; w1 < WPT; ++w1) {
        for (int w2 = 0; w2 < WPT; ++w2) {
            int rowC = globalRow * WPT + w1;
            int colC = globalCol * WPT + w2;
            if (rowC < M && colC < N) {
                C[rowC * N + colC] = Cvalue[w1][w2];
            }
        }
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
  dim3 blockDim(RTS, RTS);
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