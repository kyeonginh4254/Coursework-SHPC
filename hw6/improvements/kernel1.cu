#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

/* GLOBALS */

int mpi_rank, mpi_world_size;
int num_devices;

int M_node_begin, M_node_end, M_node_size;
#define NUM_DEVICE 4
int Mbegin[NUM_DEVICE], Mend[NUM_DEVICE];

static float *a_d[NUM_DEVICE], *b_d[NUM_DEVICE], *c_d[NUM_DEVICE];
static cudaStream_t streams[NUM_DEVICE];

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = tmp;
  }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  float *A_node = (float *)malloc(M_node_size * K * sizeof(float));
  float *C_node = (float *)malloc(M_node_size * N * sizeof(float));

  MPI_Scatter(A, M_node_size * K, MPI_FLOAT, A_node, M_node_size * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  #pragma omp parallel for num_threads(num_devices)
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    CUDA_CALL(cudaMemcpyAsync(b_d[i],
                              B,
                              K * N * sizeof(float),
                              cudaMemcpyHostToDevice,
                              streams[0]));

    int local_offset = Mbegin[i] - M_node_begin;
    int M_i_size = Mend[i] - Mbegin[i];

    CUDA_CALL(cudaMemcpyAsync(a_d[i],
                              &A_node[local_offset * K],
                              M_i_size * K * sizeof(float),
                              cudaMemcpyHostToDevice, 
                              streams[i]));

    dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16), 1);
    dim3 blockDim(16, 16, 1);

    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(a_d[i], b_d[i], c_d[i], M_i_size, N, K);

    // 결과를 호스트 메모리로 복사
    CUDA_CALL(cudaMemcpyAsync(&C_node[local_offset * N],
                              c_d[i],
                              M_i_size * N * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              streams[i]));
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamSynchronize(streams[i]));
  }

  MPI_Gather(C_node, M_node_size * N, MPI_FLOAT, C, M_node_size * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  free(A_node);
  free(C_node);
}

void matmul_initialize(int M, int N, int K) {

  // initialization
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("[rank %d] Number of devices: %d\n", mpi_rank, num_devices);

  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));
  }

  // M setting
  M_node_begin = M * mpi_rank / mpi_world_size;
  M_node_end = M * (mpi_rank + 1) / mpi_world_size;
  M_node_size = M_node_end - M_node_begin;

  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = M_node_begin + M_node_size * i / num_devices;
    Mend[i] = M_node_begin + M_node_size * (i + 1) / num_devices;
    if (i == num_devices - 1) {
      Mend[i] = M_node_end;
    }
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
    CUDA_CALL(cudaStreamCreate(&streams[i]));
  }
}

void matmul_finalize() {
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  }
}
