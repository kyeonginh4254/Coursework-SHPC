#include <mpi.h>
#include <cstdio>
#include "layer.h"
#include "model.h"

#define BATCH_SIZE 64

// Timer macro for CUDA
#define START_TIMER(start) cudaEventRecord(start, 0)
#define STOP_TIMER(start, stop, elapsed)                        \
    cudaEventRecord(stop, 0);                                   \
    cudaEventSynchronize(stop);                                 \
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

// Function to predict sentiment
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // SEQ_LEN = 16

    if (mpi_rank == 0) {
        // CUDA events for timing
        cudaEvent_t start, stop;
        float elapsed;

        // Create CUDA events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        /* Predict sentiment for each sentence */
        for (size_t n = 0; n < n_samples; n += BATCH_SIZE) {
            printf("Processing batch %zu\n", n / BATCH_SIZE);

            // Load a batch of sentences from the inputs
            // in: [n_samples * SEQ_LEN], out: [BATCH_SIZE * SEQ_LEN]
            int *batch_input = inputs + n * SEQ_LEN;

            START_TIMER(start);
            // Embedding layer
            // in: [BATCH_SIZE, SEQ_LEN], out: [BATCH_SIZE, SEQ_LEN, 4096]
            Embedding(batch_input, emb_w, emb_a);
            STOP_TIMER(start, stop, elapsed);
            printf("Embedding: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Permute operation
            // in: [BATCH_SIZE, SEQ_LEN, 4096], out: [BATCH_SIZE, 4096, SEQ_LEN]
            Permute(emb_a, permute_a);
            STOP_TIMER(start, stop, elapsed);
            printf("Permute: %.3f ms\n", elapsed);

            START_TIMER(start);
            // First convolutional layer
            // in: [BATCH_SIZE, 4096, SEQ_LEN], out: [BATCH_SIZE, 1024, SEQ_LEN - 2]
            Conv1D_ReLU_CUDA(permute_a, conv0_w, conv0_b, conv0_a);
            STOP_TIMER(start, stop, elapsed);
            printf("Conv0 + ReLU: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Max pooling for the first convolutional layer
            // in: [BATCH_SIZE, 1024, SEQ_LEN - 2], out: [BATCH_SIZE, 1024]
            GetMax(conv0_a, pool0_a);
            STOP_TIMER(start, stop, elapsed);
            printf("GetMax (pool0): %.3f ms\n", elapsed);

            START_TIMER(start);
            // Second convolutional layer
            // in: [BATCH_SIZE, 4096, SEQ_LEN], out: [BATCH_SIZE, 1024, SEQ_LEN - 4]
            Conv1D_ReLU_CUDA(permute_a, conv1_w, conv1_b, conv1_a);
            STOP_TIMER(start, stop, elapsed);
            printf("Conv1 + ReLU: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Max pooling for the second convolutional layer
            // in: [BATCH_SIZE, 1024, SEQ_LEN - 4], out: [BATCH_SIZE, 1024]
            GetMax(conv1_a, pool1_a);
            STOP_TIMER(start, stop, elapsed);
            printf("GetMax (pool1): %.3f ms\n", elapsed);

            START_TIMER(start);
            // Third convolutional layer
            // in: [BATCH_SIZE, 4096, SEQ_LEN], out: [BATCH_SIZE, 1024, SEQ_LEN - 6]
            Conv1D_ReLU_CUDA(permute_a, conv2_w, conv2_b, conv2_a);
            STOP_TIMER(start, stop, elapsed);
            printf("Conv2 + ReLU: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Max pooling for the third convolutional layer
            // in: [BATCH_SIZE, 1024, SEQ_LEN - 6], out: [BATCH_SIZE, 1024]
            GetMax(conv2_a, pool2_a);
            STOP_TIMER(start, stop, elapsed);
            printf("GetMax (pool2): %.3f ms\n", elapsed);

            START_TIMER(start);
            // Fourth convolutional layer
            // in: [BATCH_SIZE, 4096, SEQ_LEN], out: [BATCH_SIZE, 1024, SEQ_LEN - 8]
            Conv1D_ReLU_CUDA(permute_a, conv3_w, conv3_b, conv3_a);
            STOP_TIMER(start, stop, elapsed);
            printf("Conv3 + ReLU: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Max pooling for the fourth convolutional layer
            // in: [BATCH_SIZE, 1024, SEQ_LEN - 8], out: [BATCH_SIZE, 1024]
            GetMax(conv3_a, pool3_a);
            STOP_TIMER(start, stop, elapsed);
            printf("GetMax (pool3): %.3f ms\n", elapsed);

            START_TIMER(start);
            // Concatenate all pooled results
            // in: [BATCH_SIZE, 1024] x 4, out: [BATCH_SIZE, 1024 * 4]
            Concat(pool0_a, pool1_a, pool2_a, pool3_a, concat_a);
            STOP_TIMER(start, stop, elapsed);
            printf("Concat: %.3f ms\n", elapsed);

            START_TIMER(start);
            // First fully connected layer with ReLU
            // in: [BATCH_SIZE, 1024 * 4], out: [BATCH_SIZE, 2048]
            Linear_ReLU_CUDA(concat_a, linear0_w, linear0_b, linear0_a, 1);
            STOP_TIMER(start, stop, elapsed);
            printf("Linear0 + ReLU: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Second fully connected layer with ReLU
            // in: [BATCH_SIZE, 2048], out: [BATCH_SIZE, 1024]
            Linear_ReLU_CUDA(linear0_a, linear1_w, linear1_b, linear1_a, 1);
            STOP_TIMER(start, stop, elapsed);
            printf("Linear1 + ReLU: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Third fully connected layer with ReLU
            // in: [BATCH_SIZE, 1024], out: [BATCH_SIZE, 512]
            Linear_ReLU_CUDA(linear1_a, linear2_w, linear2_b, linear2_a, 1);
            STOP_TIMER(start, stop, elapsed);
            printf("Linear2 + ReLU: %.3f ms\n", elapsed);

            START_TIMER(start);
            // Final fully connected layer
            // in: [BATCH_SIZE, 512], out: [BATCH_SIZE, 2]
            Linear_ReLU_CUDA(linear2_a, linear3_w, linear3_b, linear3_a, 0);
            STOP_TIMER(start, stop, elapsed);
            printf("Linear3 (final): %.3f ms\n", elapsed);

            START_TIMER(start);
            // Copy the computation result to the outputs
            // in: [BATCH_SIZE, 2], out: [n_samples * 2]
            memcpy(outputs + n * 2, linear3_a->buf, BATCH_SIZE * 2 * sizeof(float));
            STOP_TIMER(start, stop, elapsed);
            printf("Memcpy to output: %.3f ms\n\n", elapsed);
        }

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}