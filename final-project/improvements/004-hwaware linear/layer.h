#pragma once

#include "tensor.h"


/* Operations (layers) */
void Embedding(int *in, Tensor *w, Tensor *out);
void Permute(Tensor *in, Tensor *out);
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ReLU(Tensor *inout);
void GetMax(Tensor *in, Tensor *out);
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out);
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);

/* Example of using CUDA kernel */
void Conv1D_ReLU_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Linear_ReLU_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);