#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <sched.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

void matmul(float *A, float *B, float *C, int M, int N, int K,
            int num_threads) {
    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);


        int ii = 0;
        int kk = 0;
        int jj = 0;

        int i_end = (ii + BLOCK_SIZE > M) ? M : ii + BLOCK_SIZE;
        int k_end = (kk + BLOCK_SIZE > K) ? K : kk + BLOCK_SIZE;
        int j_end = (jj + BLOCK_SIZE > N) ? N : jj + BLOCK_SIZE;

        for (int i = ii; i < i_end; ++i) {
            for (int j = jj; j < j_end; ++j) {
                for (int k = kk; k < k_end; ++k) {
                    float a = A[i*K + k];
                    float b = B[k*N + j];
                    C[i*N + j] += a*b;
                    A[i*K + k] = a+1;
                    B[k*N + j] = b+1;
                }
            }
        }
    }
}