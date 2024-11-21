#define TS 32
#define WPT 2
#define RTS TS/WPT

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                    int M, int N, int K) {
    const int row = get_local_id(1);
    const int col = get_local_id(0);
    const int globalRow = TS * get_group_id(1) + row;
    const int globalCol = RTS * get_group_id(0) + col;
    int tiledRow, tiledCol;

    // Local memory for tiles of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float Cvalue[WPT];
    for (int w = 0; w < WPT; w++) {
        Cvalue[w] = 0.0f;
    }

    const int numTiles = K / TS;

    // Iterate over tiles
    for (int t = 0; t < numTiles; t++) {
        tiledRow = TS * t + row;
        tiledCol = RTS * t + col;
        for (int w = 0; w < WPT; w++) {
            Asub[row][col * WPT + w] = A[globalRow * K + (tiledCol * WPT + w)];
            Bsub[row][col * WPT + w] = B[tiledRow * N + (globalCol * WPT + w)];
        }

        // Synchronize to ensure all data is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial products
        for (int k=0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                Cvalue[w] += Asub[row][k] * Bsub[k][col * WPT + w];
            }
        }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result
    for (int w = 0; w < WPT; w++) {
        C[globalRow * N + (globalCol * WPT + w)] = Cvalue[w];
    }
}
