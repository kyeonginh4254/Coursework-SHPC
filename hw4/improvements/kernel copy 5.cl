#define TS 64
#define WPT 8
#define RTS TS/WPT

__kernel void sgemm(__global const float *A,
                    __global const float *B,
                    __global float *C,
                    int M, int N, int K) {

    const int rowOffset = get_global_id(1) * WPT; // 0..((M+TS-1) / TS) * WPT
    const int colOffset = get_global_id(0) * WPT; // 0..((N+TS-1) / TS) * WPT

    int ci, cj;

    __local float subA[TS][TS]; // M * K의 축소
    __local float subB[TS][TS]; // K * N의 축소

    float Cvalue[RTS][RTS];
    for (ci = 0; ci < RTS; ci++) {
        for (cj = 0; cj < RTS; cj++) {
            Cvalue[ci][cj] = 0.0f;
        }
    }

    int rowA, colA, rowB, colB;

    const int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++) {

        for (ci = 0; ci < RTS; ci++) {
            for (cj = 0; cj < RTS; cj++) {

                rowA = rowOffset + ci;
                colA = t * TS + get_local_id(0) * RTS + cj;

                if (rowA < M && colA < K) {
                    subA[get_local_id(1) * RTS + ci][get_local_id(0) * RTS + cj] = A[rowA * K + colA];
                } else {
                    subA[get_local_id(1) * RTS + ci][get_local_id(0) * RTS + cj] = 0.0f;
                }

                rowB = t * TS + get_local_id(0) * RTS + ci;
                colB = colOffset + cj;

                if (rowB < K && colB < N) {
                    subB[get_local_id(0) * RTS + ci][get_local_id(1) * RTS + cj] = B[rowB * N + colB];
                } else {
                    subB[get_local_id(0) * RTS + ci][get_local_id(1) * RTS + cj] = 0.0f;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (ci = 0; ci < RTS; ci++) {
            for (int k = 0; k < RTS; k++) {
                float a_ik = subA[get_local_id(1) * RTS + ci][t * RTS + k];
                for (cj = 0; cj < RTS; cj++) {
                    Cvalue[ci][cj] += a_ik * subB[t * RTS + k][get_local_id(0) * RTS + cj];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
    }

    for (ci = 0; ci < RTS; ci++) {
        for (cj = 0; cj < RTS; cj++) {
            int row = rowOffset + ci;
            int col = colOffset + cj;
            if (row < M && col < N) {
                C[row * N + col] = Cvalue[ci][cj];
            }
        }
    }
}
