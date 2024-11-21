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
    int tiledRow, tiledCol;
    
    __local float Asub[TSK][TSM], Bsub[TSN][TSK];

    float Cvalue[WPTN];
    for (int w = 0; w < WPTN; w++) {
        Cvalue[w] = 0.0f;
    }

    const int numTiles = K / TSK;

    for (int t = 0; t < numTiles; t++) {
        for (int l = 0; l < LPT; l++) {
            Asub[col * LPT + l][row] = A[globalRow * K + t * TSK + col * LPT + l];
            Bsub[][col * LPT + l] = B[]
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TSK; k++) {
            for (int w = 0; w < WPTN; w++) {
                Cvalue[w] += Asub[k][row] * Bsub[col + w * RTSN][k];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPTN; w++) {
        C[(globalCol + w * RTSN) * M + globalRow] = Cvalue[w];
    }
}

// Simple transpose kernel for a P * Q matrix
__kernel void transpose(const int P, const int Q,
                        __global const float* input,
                        __global float* output) {
    
    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q
 
    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];
 
    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }
 
    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;
 
    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}
