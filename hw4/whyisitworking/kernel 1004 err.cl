#define TS 32
#define WPT 2
#define RTS TS/WPT

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                    int M, int N, int K) {
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int globalRow = RTS * get_group_id(1) + row;
    const int globalCol = RTS * get_group_id(0) + col;
    int tiledRow, tiledCol;

    // Local memory for tiles of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float Cvalue[WPT][WPT];
    for (int w1 = 0; w1 < WPT; w1++) {
        for (int w2 = 0; w2 < WPT; w2++) {
            Cvalue[w1][w2] = 0.0f;
        }
    }

    const int numTiles = K / TS;

    // Iterate over tiles
    for (int t = 0; t < numTiles; t++) {
        tiledRow = RTS * t + row;
        tiledCol = RTS * t + col;
        for (int w1 = 0; w1 < WPT; w1++) {
            for (int w2 = 0; w2 < WPT; w2++) {
                Asub[row * WPT + w1][col * WPT + w2] = A[(globalRow * WPT + w1) * K + (tiledCol * WPT + w2)];
                Bsub[row * WPT + w1][col * WPT + w2] = B[(tiledRow * WPT + w2) * N + (globalCol * WPT + w2)];
            }
        }

        // Synchronize to ensure all data is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial products
        for (int k=0; k < TS; k++) {
            for (int w1 = 0; w1 < WPT; w1++) {
                for (int w2 = 0; w2 < WPT; w2++) {
                    Cvalue[w1][w2] += Asub[row * WPT + w1][k] * Bsub[k][col * WPT + w2];
                }
            }
        }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result
    for (int w1 = 0; w1 < WPT; w1++) {
        for (int w2 = 0; w2 < WPT; w2++) {
            C[(globalRow * WPT + w1) * N + (globalCol * WPT + w2)] = Cvalue[w1][w2];
        }
    }
}
