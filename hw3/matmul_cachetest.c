#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 2048

void matmul(float *A, float *B, float *C, int M, int N, int K,
            int num_threads) {
    printf("\nBLOCK_SIZE: %d\n", BLOCK_SIZE);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(guided) collapse(2)
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                int i_end = (ii + BLOCK_SIZE > M) ? M : ii + BLOCK_SIZE;
                int k_end = (kk + BLOCK_SIZE > K) ? K : kk + BLOCK_SIZE;
                int j_end = (jj + BLOCK_SIZE > N) ? N : jj + BLOCK_SIZE;
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        float sum = C[i*N + j];
                        for (int k = kk; k < k_end; ++k) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] = sum;
                    }
                }
            }
        }
    }
}