#include "matmul_multi.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

int num_devices = 0;

#define MAX_NUM_GPU 4
#define NUM_STREAMS 4

#define TILE_SIZE 64
#define WPS 8
cudaStream_t streams[MAX_NUM_GPU][NUM_STREAMS];

// Global pointers for Pinned Memory
float *pinned_A = nullptr;
float *pinned_B = nullptr;
float *pinned_C = nullptr;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
   int local_i = threadIdx.x;
    int local_j = threadIdx.y;
    int global_row = blockIdx.x * TILE_SIZE + local_i * WPS;
    int global_col = blockIdx.y * TILE_SIZE + local_j * WPS;

    __shared__ float Asub[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE+1];

    float sum[WPS][WPS] = {0.0f};

    int NUM_TILES = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < NUM_TILES; ++t) {
        int t_row = t * TILE_SIZE + local_i * WPS;
        int t_col = t * TILE_SIZE + local_j * WPS;

        for (int wi = 0; wi < WPS; ++wi) {
            for (int wj = 0; wj < WPS; ++wj) {
                int a_row = global_row + wi;
                int a_col = t_col + wj;
                int b_row = t_row + wi;
                int b_col = global_col + wj;

                if (a_row < M && a_col < K) {
                    Asub[local_i * WPS + wi][local_j * WPS + wj] = A[a_row * K + a_col];
                } else {
                    Asub[local_i * WPS + wi][local_j * WPS + wj] = 0.0f;
                }

                if (b_row < K && b_col < N) {
                    Bsub[local_i * WPS + wi][local_j * WPS + wj] = B[b_row * N + b_col];
                } else {
                    Bsub[local_i * WPS + wi][local_j * WPS + wj] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int wi = 0; wi < WPS; ++wi) {
                for (int wj = 0; wj < WPS; ++wj) {
                    sum[wi][wj] += Asub[local_i * WPS + wi][k] * Bsub[k][local_j * WPS + wj];
                }
            }
        }

        __syncthreads();
    }

    for (int wi = 0; wi < WPS; ++wi) {
        for (int wj = 0; wj < WPS; ++wj) {
            int c_idx = (global_row + wi) * N + (global_col + wj);
            if (global_row + wi < M && global_col + wj < N) {
                C[c_idx] = sum[wi][wj];
            }
        }
    }
}


// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU][NUM_STREAMS];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU][NUM_STREAMS];
static int Mbegin[MAX_NUM_GPU][NUM_STREAMS], Mend[MAX_NUM_GPU][NUM_STREAMS];

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    memcpy(pinned_A, A, M * K * sizeof(float));
    memcpy(pinned_B, B, K * N * sizeof(float));
  for (int i=0; i< num_devices; i++){
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(
        cudaMemcpyAsync(b_d[i], pinned_B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i][0])
        );
        
    for (int j=0; j< NUM_STREAMS ; j++)
    {
      CUDA_CALL(cudaStreamSynchronize(streams[i][j]));
      int each_m = Mend[i][j] - Mbegin[i][j];
      int global_m = ((each_m + WPS - 1) / WPS) * WPS;
      int global_n = ((N + WPS - 1) / WPS) * WPS;

      dim3 blockDim(TILE_SIZE / WPS, TILE_SIZE / WPS, 1); // Block당 스레드
      dim3 gridDim((global_m + TILE_SIZE - 1) / TILE_SIZE, 
                    (global_n + TILE_SIZE - 1) / TILE_SIZE, 
                    1);

                    
      CUDA_CALL(
      cudaMemcpyAsync(a_d[i][j], pinned_A + Mbegin[i][j] * K,
                         each_m * K * sizeof(float),
                         cudaMemcpyHostToDevice, 
                         streams[i][j]                         
                         )
    );

    matmul_kernel<<<gridDim, blockDim, 0, streams[i][j]>>>(a_d[i][j], b_d[i], c_d[i][j], each_m, N, K);

      CUDA_CALL(cudaMemcpyAsync(pinned_C + Mbegin[i][j] * N, c_d[i][j],
                         (Mend[i][j] - Mbegin[i][j]) * N * sizeof(float),
                         cudaMemcpyDeviceToHost, streams[i][j]));

    }
  }

  // Download C matrix from GPUs
  memcpy(C, pinned_C, M * N * sizeof(float));
  for (int i = 0; i < num_devices; i++) {
    for (int j=0; j< NUM_STREAMS; j++){    
  }
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void matmul_initialize(int M, int N, int K) {
  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  CUDA_CALL(cudaHostAlloc((void**)&pinned_A, sizeA, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc((void**)&pinned_B, sizeB, cudaHostAllocDefault));
  CUDA_CALL(cudaHostAlloc((void**)&pinned_C, sizeC, cudaHostAllocDefault));

  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("Using %d devices\n", num_devices);
  cudaDeviceProp props[num_devices];
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaGetDeviceProperties(&props[i], i));

    // Try printing more detailed information here
    printf("GPU %d: %s\n", i, props[i].name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Setup problem size for each GPU
for (int i = 0; i < num_devices; i++) {
    int rows_per_gpu = M / num_devices;
    int rows_per_stream = (rows_per_gpu + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int j = 0; j < NUM_STREAMS; j++) {
        Mbegin[i][j] = (rows_per_gpu * i) + (rows_per_stream * j);
        Mend[i][j] = min(Mbegin[i][j] + rows_per_stream, (rows_per_gpu * (i + 1)));
    }
}
  Mend[num_devices - 1][NUM_STREAMS - 1] = M;

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    // SM: Add new streams!
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    
    for (int j= 0; j< NUM_STREAMS; j++)
    {
    CUDA_CALL(cudaStreamCreate(&streams[i][j]));

    CUDA_CALL(cudaMalloc(&a_d[i][j], (Mend[i][j] - Mbegin[i][j]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i][j], (Mend[i][j] - Mbegin[i][j]) * N * sizeof(float)));
    }
  }
}

void matmul_finalize() {
  // Free all GPU memory
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaFree(b_d[i]));
    for (int j=0 ; j< NUM_STREAMS; j++){
    CUDA_CALL(cudaFree(a_d[i][j]));
    CUDA_CALL(cudaFree(c_d[i][j]));
    CUDA_CALL(cudaStreamDestroy(streams[i][j]));
    }
  }
    if (pinned_A) CUDA_CALL(cudaFreeHost(pinned_A));
    if (pinned_B) CUDA_CALL(cudaFreeHost(pinned_B));
    if (pinned_C) CUDA_CALL(cudaFreeHost(pinned_C));
}