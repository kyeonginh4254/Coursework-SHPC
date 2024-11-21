#define TSM 64                 // The tile-size in dimension M
#define TSN 64                 // The tile-size in dimension N
#define TSK 32                 // The tile-size in dimension K
#define WPTN 8                 // The work-per-thread in dimension N
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPT (TSK / RTSN)       // The loads-per-thread for a tile

#define TRANSPOSEX 16          // Tile size for transpose in X dimension
#define TRANSPOSEY 16          // Tile size for transpose in Y dimension

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                   int M, int N, int K) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TSM * get_group_id(0) + row;
    const int globalCol = RTSN * get_group_id(1) + col;
    int tiledIndex;
    
    __local float Asub[TSK][TSM], Bsub[TSN][TSK];

    float Cvalue[WPTN];
    for (int w = 0; w < WPTN; w++) {
        Cvalue[w] = 0.0f;
    }

    const int numTiles = K / TSK;

    for (int t = 0; t < numTiles; t++) {
        for (int l = 0; l < LPT; l++) {
            tiledIndex = globalRow * K + TSK * t + LPT * col + l;
            Asub[LPT * col + l][row] = A[tiledIndex];
            Bsub[row][LPT * col + l] = B[tiledIndex];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSK; k++) {
            for (int w = 0; w < WPTN; w++) {
                Cvalue[w] += Asub[k][row] * Bsub[col * WPTN + w][k];
            }
        }
    }

    for (int w = 0; w < WPTN; w++) {
        C[globalRow * N + globalCol * WPTN + w] = Cvalue[w];
    }

}

__kernel void transpose(const int P, const int Q,
                        __global const float* input,
                        __global float* output) {
    
    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0) * TRANSPOSEX + tx; // row index
    const int ID1 = get_group_id(1) * TRANSPOSEY + ty; // column index

    __local float buffer[TRANSPOSEY][TRANSPOSEX];

    // Load data from input to buffer
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID0 * Q + ID1];
    } else {
        buffer[ty][tx] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate new indices after transposition
    const int newID0 = get_group_id(1) * TRANSPOSEY + ty; // now column index
    const int newID1 = get_group_id(0) * TRANSPOSEX + tx; // now row index

    // Write data from buffer to output
    if (newID1 < Q && newID0 < P) {
        output[newID1 * P + newID0] = buffer[tx][ty];
    }
}
