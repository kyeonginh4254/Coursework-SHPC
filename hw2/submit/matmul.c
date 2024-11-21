#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdatomic.h>

#define BLOCK_SIZE 128

struct thread_arg {
  const float *A;
  const float *B_T;
  float *C;
  int M;
  int N;
  int K;
  atomic_int *work_queue;
  int num_threads;
} args[256];
static pthread_t threads[256];

static inline float horizontal_add(__m512 a) {
  __m256 low = _mm512_castps512_ps256(a);
  __m256 high = _mm512_extractf32x8_ps(a, 1);

  __m256 sum = _mm256_add_ps(low, high);
  __m128 sum_low = _mm256_castps256_ps128(sum);
  __m128 sum_high = _mm256_extractf128_ps(sum, 1);

  sum_low = _mm_add_ps(sum_low, sum_high);
  sum_low = _mm_hadd_ps(sum_low, sum_low);
  sum_low = _mm_hadd_ps(sum_low, sum_low);
  return _mm_cvtss_f32(sum_low);
}

static void *matmul_kernel(void *arg) {
  struct thread_arg *input = (struct thread_arg *)arg;
  const float *A = (*input).A;
  const float *B_T = (*input).B_T;
  float *C = (*input).C;
  int M = (*input).M;
  int N = (*input).N;
  int K = (*input).K;
  atomic_int *work_queue = input->work_queue;

  int block_index;
  while ((block_index = atomic_fetch_add(work_queue, 1)) < ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)) {
    int bi = block_index / ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int bj = block_index % ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int i_start = bi * BLOCK_SIZE;
    int i_end = (i_start + BLOCK_SIZE > M) ? M : i_start + BLOCK_SIZE;
    int j_start = bj * BLOCK_SIZE;
    int j_end = (j_start + BLOCK_SIZE > N) ? N : j_start + BLOCK_SIZE;

    for (int bk = 0; bk < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++bk) {
      int k_start = bk * BLOCK_SIZE;
      int k_end = (k_start + BLOCK_SIZE > K) ? K : k_start + BLOCK_SIZE;

      for (int i = i_start; i < i_end; ++i) {
        for (int j = j_start; j < j_end; ++j) {
          int k = k_start;
          
          for (; k < k_end - 15; k += 16) {
            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
            __m512 b_vec = _mm512_loadu_ps(&B_T[j * K + k]);

            __m512 mult = _mm512_mul_ps(a_vec, b_vec);

            float add = horizontal_add(mult);
            C[i * N + j] += add;

          }
          if (k < k_end) {
            __mmask16 mask = (1 << (k_end - k)) - 1;

            __m512 a_vec = _mm512_maskz_loadu_ps(mask, &A[i * K + k]);
            __m512 b_vec = _mm512_maskz_loadu_ps(mask, &B_T[j * K + k]);

            __m512 mult = _mm512_mul_ps(a_vec, b_vec);

            float add = horizontal_add(mult);
            C[i * N + j] += add;
          }
        }
      }
    }
  }

  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K, int num_threads) {

  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  float *B_T = (float *)aligned_alloc(64, sizeof(float) * K * N);

  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      B_T[n * K + k] = B[k * N + n];
    }
  }

  atomic_int work_queue = 0;

  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B_T = B_T, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].work_queue = &work_queue;
    err = pthread_create(&threads[t], NULL, matmul_kernel, (void *)&args[t]);
    if (err) {
      printf("pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }

  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      printf("pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }
}
