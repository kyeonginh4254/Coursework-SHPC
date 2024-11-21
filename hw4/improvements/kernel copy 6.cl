#define TS 64
#define WPT 8
#define RTS TS/WPT

__kernel void sgemm(__global const float *A,
                    __global const float *B,
                    __global float *C,
                    int M, int N, int K) {
                        
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

                rowA = get_global_id(1) * RTS + ci;
                colA = t * TS + get_local_id(0) * RTS + cj;

                if (rowA < M && colA < K) {
                    subA[get_local_id(1) * RTS + ci][get_local_id(0) * RTS + cj] = A[rowA * K + colA];
                } else {
                    subA[get_local_id(1) * RTS + ci][get_local_id(0) * RTS + cj] = 0.0f;
                }

                rowB = t * TS + get_local_id(1) * RTS + ci;
                colB = get_global_id(0) * RTS + cj;

                if (rowB < K && colB < N) {
                    subB[get_local_id(1) * RTS + ci][get_local_id(0) * RTS + cj] = B[rowA * K + colA];
                } else {
                    subB[get_local_id(1) * RTS + ci][get_local_id(0) * RTS + cj] = 0.0f;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (ci = 0; ci < RTS; ci++) {
            for (int k = 0; k < TS; k++) {
                float a_ik = subA[get_local_id(1) * RTS + ci][k];
                for (cj = 0; cj < RTS; cj++) {
                    Cvalue[ci][cj] += a_ik * subB[k][get_local_id(0) * RTS + cj];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
    }

    for (ci = 0; ci < RTS; ci++) {
        for (cj = 0; cj < RTS; cj++) {
            int row = get_global_id(1) * RTS + ci;
            int col = get_global_id(0) * RTS + cj;
            if (row < M && col < N) {
                C[row * N + col] = Cvalue[ci][cj];
            }
        }
    }
}
