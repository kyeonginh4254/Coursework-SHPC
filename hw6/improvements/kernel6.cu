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

#define TS 64
#define WPS 4
#define RTS (TS / WPS)

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  
  const int row = threadIdx.x / RTS; // Local row index within the block
  const int col = threadIdx.x % RTS; // Local column index within the block
  const int globalRow = RTS * blockIdx.x + row; // Global row index
  const int globalCol = RTS * blockIdx.y + col; // Global column index

  __shared__ float Asub[TS][TS + 1];
  __shared__ float Bsub[TS][TS + 1];

  float Cvalue[WPS][WPS] = {0.0f};

  const int numTiles = (K + TS - 1) / TS;

  for (int t = 0; t < numTiles; ++t) {
    int tiledRow = RTS * t + row;
    int tiledCol = RTS * t + col;

    // Load data into shared memory
    for (int w1 = 0; w1 < WPS; ++w1) {
      for (int w2 = 0; w2 < WPS; ++w2) {
        int rowA = globalRow * WPS + w1;
        int colA = tiledCol * WPS + w2;
        if (rowA < M && colA < K) {
          Asub[row * WPS + w1][col * WPS + w2] = A[rowA * K + colA];
        } else {
          Asub[row * WPS + w1][col * WPS + w2] = 0.0f;
        }

        int rowB = tiledRow * WPS + w1;
        int colB = globalCol * WPS + w2;
        if (rowB < K && colB < N) {
          Bsub[row * WPS + w1][col * WPS + w2] = B[rowB * N + colB];
        } else {
          Bsub[row * WPS + w1][col * WPS + w2] = 0.0f;
        }
      }
    }

    __syncthreads();

    // Compute partial results
    for (int k = 0; k < TS; ++k) {
      for (int w1 = 0; w1 < WPS; ++w1) {
        for (int w2 = 0; w2 < WPS; ++w2) {
          Cvalue[w1][w2] += Asub[row * WPS + w1][k] * Bsub[k][col * WPS + w2];
        }
      }
    }

    __syncthreads();
  }

  // Write the final result to global memory
  for (int w1 = 0; w1 < WPS; ++w1) {
    for (int w2 = 0; w2 < WPS; ++w2) {
      int rowC = globalRow * WPS + w1;
      int colC = globalCol * WPS + w2;
      if (rowC < M && colC < N) {
        C[rowC * N + colC] = Cvalue[w1][w2];
      }
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

      dim3 gridDim(CEIL_DIV(Msize[i][j], TS), CEIL_DIV(N, TS));
      dim3 blockDim(RTS * RTS);
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
      if (j == NUM_CHUNK) Mend[i][j] = M_device * (i+1);
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