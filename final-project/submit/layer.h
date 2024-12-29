#pragma once

#include "tensor.h"


__global__ void Embedding_Permute_Kernel(const int *__restrict__ in, 
                                        const float *__restrict__ w, 
                                        float *__restrict__ out,
                                        size_t B, size_t s, size_t H);

void Conv1d(float *d_permute_a, float **d_conv_w, float **d_conv_b, float **d_conv_a, float **d_out, cudaStream_t *streams);

void GetMax(float *d_in, float *d_out, size_t s, cudaStream_t stream);
void Concat(float *d_in1, float *d_in2, float *d_in3, float *d_in4, float *d_out, cudaStream_t stream);
void Linear_ReLU_CUDA(float *d_in, float *d_w, float *d_b, float *d_out, size_t N, size_t M, cudaStream_t stream);
void Linear_CUDA(float *d_in, float *d_w, float *d_b, float *d_out, size_t N, size_t M, cudaStream_t stream);