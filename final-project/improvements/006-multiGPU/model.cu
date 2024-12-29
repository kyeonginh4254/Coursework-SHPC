#include <mpi.h>
#include <cstdio>
#include <omp.h>
#include "layer.h"
#include "model.h"

#define BATCH_SIZE 1024
#define MINI_BATCH_SIZE 256
#define NUM_CHUNKS (BATCH_SIZE / MINI_BATCH_SIZE)
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Timer macro for CUDA
#define START_TIMER(start) cudaEventRecord(start, 0)
#define STOP_TIMER(start, stop, elapsed)                        \
  cudaEventRecord(stop, 0);                                     \
  cudaEventSynchronize(stop);                                   \
  cudaEventElapsedTime(&elapsed, start, stop)

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *emb_w;
Parameter *conv0_w, *conv0_b;
Parameter *conv1_w, *conv1_b;
Parameter *conv2_w, *conv2_b;
Parameter *conv3_w, *conv3_b;
Parameter *linear0_w, *linear0_b;
Parameter *linear1_w, *linear1_b;
Parameter *linear2_w, *linear2_b;
Parameter *linear3_w, *linear3_b;

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos = 0;

  emb_w = new Parameter({21635, 4096}, param + pos);
  pos += 21635 * 4096; 

  conv0_w = new Parameter({1024, 4096, 3}, param + pos);
  pos += 1024 * 4096 * 3; 
  conv0_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv1_w = new Parameter({1024, 4096, 5}, param + pos);
  pos += 1024 * 4096 * 5; 
  conv1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv2_w = new Parameter({1024, 4096, 7}, param + pos);
  pos += 1024 * 4096 * 7;
  conv2_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv3_w = new Parameter({1024, 4096, 9}, param + pos);
  pos += 1024 * 4096 * 9;
  conv3_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear0_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  linear0_b = new Parameter({2048}, param + pos);
  pos += 2048;

  linear1_w = new Parameter({1024, 2048}, param + pos);
  pos += 1024 * 2048;
  linear1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear2_w = new Parameter({512, 1024}, param + pos);
  pos += 512 * 1024;
  linear2_b = new Parameter({512}, param + pos);
  pos += 512;

  linear3_w = new Parameter({2, 512}, param + pos);
  pos += 2 * 512;
  linear3_b = new Parameter({2}, param + pos);
  pos += 2;

  if (pos != param_size) {
    fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
            pos, param_size);
    exit(EXIT_FAILURE);
  }
}

void free_parameters() {
  delete emb_w;
  delete conv0_w;
  delete conv0_b;
  delete conv1_w;
  delete conv1_b;
  delete conv2_w;
  delete conv2_b;
  delete conv3_w;
  delete conv3_b;
  delete linear0_w;
  delete linear0_b;
  delete linear1_w;
  delete linear1_b;
  delete linear2_w;
  delete linear2_b;
  delete linear3_w;
  delete linear3_b;
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *emb_a;
Activation *permute_a;
Activation *conv0_a, *relu0_a, *pool0_a;
Activation *conv1_a, *relu1_a, *pool1_a;
Activation *conv2_a, *relu2_a, *pool2_a;
Activation *conv3_a, *relu3_a, *pool3_a;
Activation *concat_a;
Activation *linear0_a, *linear1_a, *linear2_a, *linear3_a;

void alloc_activations() {
  emb_a = new Activation({BATCH_SIZE, SEQ_LEN, 4096});
  permute_a = new Activation({BATCH_SIZE, 4096, SEQ_LEN});
  conv0_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 2});
  pool0_a = new Activation({BATCH_SIZE, 1024});
  conv1_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 4});
  pool1_a = new Activation({BATCH_SIZE, 1024});
  conv2_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 6});
  pool2_a = new Activation({BATCH_SIZE, 1024});
  conv3_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 8});
  pool3_a = new Activation({BATCH_SIZE, 1024});
  concat_a = new Activation({BATCH_SIZE, 4096});
  linear0_a = new Activation({BATCH_SIZE, 2048});
  linear1_a = new Activation({BATCH_SIZE, 1024});
  linear2_a = new Activation({BATCH_SIZE, 512});
  linear3_a = new Activation({BATCH_SIZE, 2});
}

void free_activations() {
  delete emb_a;
  delete permute_a;
  delete conv0_a;
  delete pool0_a;
  delete conv1_a;
  delete pool1_a;
  delete conv2_a;
  delete pool2_a;
  delete conv3_a;
  delete pool3_a;
  delete concat_a;
  delete linear0_a;
  delete linear1_a;
  delete linear2_a;
  delete linear3_a;
}

#include <stdio.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <string.h>

/* CONV MACRO */

#define conv_BATCH_SIZE 32

#define C 4096
#define OC 1024

#define K3 3
#define K5 5
#define K7 7
#define K9 9

#define BS 32

#define BOS3 BS - K3 + 1
#define BOS5 BS - K5 + 1
#define BOS7 BS - K7 + 1
#define BOS9 BS - K9 + 1

#define BOC 8
#define TC 8

/* LINEAR MACRO */

#define M0 4096
#define N0 2048

#define M1 2048
#define N1 1024

#define M2 1024
#define N2 512

#define M3 512
#define N3 2

#define NUM_DEVICES 4

// Function to predict sentiment
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  printf("\n\nInitialization...\n");

  size_t DEVICE_SAMPLES = n_samples / NUM_DEVICES;

  /* PARAMETERS */

  Parameter* conv_w[4] = {conv0_w, conv1_w, conv2_w, conv3_w};
  Parameter* conv_b[4] = {conv0_b, conv1_b, conv2_b, conv3_b};
  size_t conv_K[4] = {3, 5, 7, 9};

  Parameter* linear_w[4] = {linear0_w, linear1_w, linear2_w, linear3_w};
  Parameter* linear_b[4] = {linear0_b, linear1_b, linear2_b, linear3_b};
  size_t linear_M[4] = {M0, M1, M2, M3};
  size_t linear_N[4] = {N0, N1, N2, N3};
  size_t os[4] = {14, 12, 10, 8};

  float *d_w[NUM_DEVICES];
  float *d_conv_w[NUM_DEVICES][4];
  float *d_conv_b[NUM_DEVICES][4];
  float *d_linear_w[NUM_DEVICES][4];
  float *d_linear_b[NUM_DEVICES][4];

  int *d_inputs[NUM_DEVICES][NUM_CHUNKS];
  float *d_out_permuted[NUM_DEVICES][NUM_CHUNKS];
  float *d_concat_a[NUM_DEVICES][NUM_CHUNKS];
  float *d_out[NUM_DEVICES][NUM_CHUNKS][4];
  float *d_conv_a[NUM_DEVICES][NUM_CHUNKS][4];
  float *d_pool_a[NUM_DEVICES][NUM_CHUNKS][4];
  float *d_linear_a[NUM_DEVICES][NUM_CHUNKS][4];

  cudaStream_t streams[NUM_DEVICES][NUM_CHUNKS][4];

  for (size_t i = 0; i < NUM_DEVICES; ++i) {

    CHECK_CUDA(cudaSetDevice(i));

    CHECK_CUDA(cudaMalloc(&d_w[i], emb_w->shape[0] * emb_w->shape[1] * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_w[i], emb_w->buf, emb_w->shape[0] * emb_w->shape[1] * sizeof(float), cudaMemcpyHostToDevice));

    for (size_t j = 0; j < 4; ++j) {
      CHECK_CUDA(cudaMalloc(&d_conv_w[i][j], OC * C * conv_K[j] * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&d_conv_b[i][j], OC * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(d_conv_w[i][j], conv_w[j]->buf, OC * C * conv_K[j] * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_conv_b[i][j], conv_b[j]->buf, OC * sizeof(float), cudaMemcpyHostToDevice));
    }

    for (size_t j = 0; j < 4; ++j) {
      CHECK_CUDA(cudaMalloc(&d_linear_w[i][j], linear_M[j] * linear_N[j] * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&d_linear_b[i][j], linear_M[j] * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(d_linear_w[i][j], linear_w[j]->buf, linear_M[j] * linear_N[j] * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_linear_b[i][j], linear_b[j]->buf, linear_M[j] * sizeof(float), cudaMemcpyHostToDevice));
    }

    for (size_t nk = 0; nk < NUM_CHUNKS; ++nk) {
      CHECK_CUDA(cudaMalloc(&d_inputs[i][nk], MINI_BATCH_SIZE * SEQ_LEN * sizeof(int)));
      CHECK_CUDA(cudaMalloc(&d_out_permuted[i][nk], MINI_BATCH_SIZE * 4096 * SEQ_LEN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&d_concat_a[i][nk], MINI_BATCH_SIZE * 4096 * sizeof(float)));

      for (size_t j = 0; j < 4; ++j) {
        CHECK_CUDA(cudaMalloc(&d_out[i][nk][j], MINI_BATCH_SIZE * OC * os[j] * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv_a[i][nk][j], MINI_BATCH_SIZE * OC * os[j] * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_pool_a[i][nk][j], MINI_BATCH_SIZE * OC * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_linear_a[i][nk][j], MINI_BATCH_SIZE * linear_N[j] * sizeof(float)));
      }
    }

    for (size_t nk = 0; nk < NUM_CHUNKS; nk++) {
      for (size_t j = 0; j < 4; j++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i][nk][j]));
      }
    }

  }

  printf("Initialization Done\n");

  int *batch_input[NUM_DEVICES][NUM_CHUNKS];

  if (mpi_rank == 0) {

    #pragma omp parallel for
    for (size_t i = 0; i < NUM_DEVICES; ++i) {

      printf("[GPU %zu] initialized\n", i);
      CHECK_CUDA(cudaSetDevice(i));

      size_t DEVICE_OFFSET = (n_samples / NUM_DEVICES) * i;

      for (size_t CHUNK_OFFSET = 0; CHUNK_OFFSET < DEVICE_SAMPLES; CHUNK_OFFSET += BATCH_SIZE) {

        for (size_t nk = 0; nk < NUM_CHUNKS; ++nk) {

          printf("[GPU %zu] Batch %zu - %zu initialized\n", i, DEVICE_OFFSET + CHUNK_OFFSET + nk * MINI_BATCH_SIZE, DEVICE_OFFSET + CHUNK_OFFSET + (nk + 1) * MINI_BATCH_SIZE - 1);

          batch_input[i][nk] = inputs + (DEVICE_OFFSET + CHUNK_OFFSET + nk * MINI_BATCH_SIZE) * SEQ_LEN;
          CHECK_CUDA(cudaMemcpyAsync(d_inputs[i][nk], batch_input[i][nk], MINI_BATCH_SIZE * SEQ_LEN * sizeof(int), cudaMemcpyHostToDevice, streams[i][nk][0]));

          dim3 block(32, 32);
          dim3 grid(CEIL_DIV(4096, 32), CEIL_DIV(SEQ_LEN, 32), MINI_BATCH_SIZE);
          Embedding_Permute_Kernel<<<grid, block, 0, streams[i][nk][0]>>>(d_inputs[i][nk], d_w[i], d_out_permuted[i][nk], MINI_BATCH_SIZE, SEQ_LEN, 4096);

          Conv1d(d_out_permuted[i][nk], d_conv_w[i], d_conv_b[i], d_conv_a[i][nk], d_out[i][nk], streams[i][nk]);
          for (size_t j = 0; j < 4; j++) {
            GetMax(d_conv_a[i][nk][j], d_pool_a[i][nk][j], os[j], streams[i][nk][j]);// conv tensor의 buffer를 float*로 만들기
          }

          Concat(d_pool_a[i][nk][0], d_pool_a[i][nk][1], d_pool_a[i][nk][2], d_pool_a[i][nk][3], d_concat_a[i][nk], streams[i][nk][0]);// conv tensor의 buffer를 float*로 만들기
          Linear_ReLU_CUDA(d_concat_a[i][nk], d_linear_w[i][0], d_linear_b[i][0], d_linear_a[i][nk][0], 4096, 2048, streams[i][nk][0]);// conv tensor의 buffer를 float*로 만들기
          Linear_ReLU_CUDA(d_linear_a[i][nk][0], d_linear_w[i][1], d_linear_b[i][1], d_linear_a[i][nk][1], 2048, 1024, streams[i][nk][0]);// conv tensor의 buffer를 float*로 만들기
          Linear_ReLU_CUDA(d_linear_a[i][nk][1], d_linear_w[i][2], d_linear_b[i][2], d_linear_a[i][nk][2], 1024, 512, streams[i][nk][0]);// conv tensor의 buffer를 float*로 만들기
          Linear_CUDA(d_linear_a[i][nk][2], d_linear_w[i][3], d_linear_b[i][3], d_linear_a[i][nk][3], 512, 2, streams[i][nk][0]);// conv tensor의 buffer를 float*로 만들기
          CHECK_CUDA(cudaMemcpyAsync(outputs + (DEVICE_OFFSET + CHUNK_OFFSET + nk * MINI_BATCH_SIZE) * 2, d_linear_a[i][nk][3], MINI_BATCH_SIZE * 2 * sizeof(float), cudaMemcpyDeviceToHost, streams[i][nk][0]));// conv tensor의 buffer를 float*로 만들기    
        }
      }
    }
  }
}