#define TS 16

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                    int M, int N, int K) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float Cval = 0.0f;

    for (int k = 0; k < K; k++) {
        Cval += A[i * K + k] * B[k * N + j];
    }
 
    // Store the result
    C[i * N + j] = Cval;
}
