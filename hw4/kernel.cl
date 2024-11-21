#define TS 32
#define WPT 2
#define RTS TS/WPT

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                    int M, int N, int K) {
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int globalRow = RTS * get_group_id(1) + row;
    const int globalCol = RTS * get_group_id(0) + col;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float Cvalue[WPT][WPT] = {0.0f};

    const int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++) {
        int tiledRow = RTS * t + row;
        int tiledCol = RTS * t + col;

        for (int w1 = 0; w1 < WPT; w1++) {
            for (int w2 = 0; w2 < WPT; w2++) {
                
                int rowA = globalRow * WPT + w1;
                int colA = tiledCol * WPT + w2;
                if (rowA < M && colA < K) {
                    Asub[row * WPT + w1][col * WPT + w2] = A[rowA * K + colA];
                } else {
                    Asub[row * WPT + w1][col * WPT + w2] = 0.0f;
                }

                int rowB = tiledRow * WPT + w1;
                int colB = globalCol * WPT + w2;
                if (rowB < K && colB < N) {
                    Bsub[row * WPT + w1][col * WPT + w2] = B[rowB * N + colB];
                } else {
                    Bsub[row * WPT + w1][col * WPT + w2] = 0.0f;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k < TS; k++) {
            for (int w1 = 0; w1 < WPT; w1++) {
                for (int w2 = 0; w2 < WPT; w2++) {
                    Cvalue[w1][w2] += Asub[row * WPT + w1][k] * Bsub[k][col * WPT + w2];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result
    for (int w1 = 0; w1 < WPT; w1++) {
        for (int w2 = 0; w2 < WPT; w2++) {
            int rowC = globalRow * WPT + w1;
            int colC = globalCol * WPT + w2;
            if (rowC < M && colC < N) {
                C[rowC * N + colC] = Cvalue[w1][w2];
            }
        }
    }
}
