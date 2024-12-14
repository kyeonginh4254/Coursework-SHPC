#include "matmul_multi.h"
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

    for (int t = 0; t < numTiles; t++) {
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
                for (int w2 = 0; w2 < WPT; w2++) {
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

//////////////////////////

// global datas

#define MAX_NUM_GPU 4
int num_devices = 0;
#define CHK 1024
#define NCHK 4096/CHK

cudaStream_t streams[MAX_NUM_GPU][NCHK];

static float *a_d[MAX_NUM_GPU][NCHK];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU][NCHK];
static int Mbegin[MAX_NUM_GPU][NCHK], Mend[MAX_NUM_GPU][NCHK];
float *pinned_A = nullptr, *pinned_B = nullptr, *pinned_C = nullptr;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  int chk = CHK;
  int nchk = NCHK;

  memcpy(pinned_A, A, M * K * sizeof(float));
  memcpy(pinned_B, B, K * N * sizeof(float));

  for (int i = 0; i < num_devices; ++i) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(b_d[i],
                              pinned_B,
                              K * N * sizeof(float),
                              cudaMemcpyHostToDevice,
                              streams[i][0]));

    CUDA_CALL(cudaStreamSynchronize(streams[i][0]));

    for (int n = 0; n < nchk; n++) {

      CUDA_CALL(cudaMemcpyAsync(a_d[i][n],
                                pinned_A + Mbegin[i][n] * K,
                                chk * K * sizeof(float),
                                cudaMemcpyHostToDevice,
                                streams[i][n]));

      dim3 blockDim(RTS, RTS);
      dim3 gridDim((N + TS - 1) / TS, (chk + TS - 1) / TS);

      matmul_kernel<<<gridDim, blockDim, 0, streams[i][n]>>>(a_d[i][n], b_d[i], c_d[i][n], chk, N, K);

      CUDA_CALL(cudaMemcpyAsync(pinned_C + Mbegin[i][n] * N,
                                c_d[i][n],
                                chk * N * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                streams[i][n]));

    }
  }

  memcpy(C, pinned_C, M * N * sizeof(float));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void matmul_initialize(int M, int N, int K) {

  CUDA_CALL(cudaHostAlloc((void**) &pinned_A, M * K * sizeof(float), cudaHostAllocWriteCombined));
  CUDA_CALL(cudaHostAlloc((void**) &pinned_B, K * N * sizeof(float), cudaHostAllocWriteCombined));
  CUDA_CALL(cudaHostAlloc((void**) &pinned_C, M * N * sizeof(float), cudaHostAllocDefault));

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
    for (int n = 0; n < NCHK; n++) {
      Mbegin[i][n] = (M / num_devices) * i + CHK * n;
      Mend[i][n] = (M / num_devices) * i + CHK * (n + 1);
    }
  }

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    for (int n = 0; n < NCHK; n++) {
      CUDA_CALL(cudaStreamCreate(&streams[i][n]));
      CUDA_CALL(cudaMalloc(&a_d[i][n], (Mend[i][n] - Mbegin[i][n]) * K * sizeof(float)));
      CUDA_CALL(cudaMalloc(&c_d[i][n], (Mend[i][n] - Mbegin[i][n]) * N * sizeof(float)));
    }
  }
}

void matmul_finalize() {

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaFree(b_d[i]));
    for (int n = 0; n < NCHK; n++) {
      CUDA_CALL(cudaStreamDestroy(streams[i][n]));
      CUDA_CALL(cudaFree(a_d[i][n]));
      CUDA_CALL(cudaFree(c_d[i][n]));
    }
  }

  CUDA_CALL(cudaFreeHost(pinned_A));
  CUDA_CALL(cudaFreeHost(pinned_B));
  CUDA_CALL(cudaFreeHost(pinned_C)); 

}