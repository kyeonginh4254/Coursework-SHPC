#define TS 16

__kernel void sgemm(__global const float *A,
                    __global const float *B,
                    __global float *C,
                    int M, int N, int K) {
    const int cRow = get_global_id(0);
    const int cCol = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    int globalRow, globalCol;

    __local float localA[TS][TS];
    __local float localB[TS][TS];

    float valC = 0.0f;
    const int numTiles = (K+TS-1)/TS;

    for (int t = 0; t < numTiles; t++) {
        globalRow = cRow;
        globalCol = t * TS + localCol;
        if (globalRow < M && globalCol < K) {
            localA[localRow][localCol] = A[globalRow * K + globalCol];
        } else {
            localA[localRow][localCol] = 0.0f;
        }

        globalRow = t * TS + localRow;
        globalCol = cCol;
        if (globalRow < M && globalCol < K) {
            localB[localRow][localCol] = B[globalRow * N + globalCol];
        } else {
            localB[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            valC += localA[localRow][k] * localB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = valC;
    }
}
