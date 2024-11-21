#define TSM 64                 // The tile-size in dimension M
#define TSN 64                 // The tile-size in dimension N
#define TSK 32                 // The tile-size in dimension K
#define WPTN 8                 // The work-per-thread in dimension N
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPT (TSK/RTSN)       // The loads-per-thread for a tile

#define TS 16

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                    int M, int N, int K) {
    // Get the row and column indices
    int row = get_global_id(1); // M dimension
    int col = get_global_id(0); // N dimension

    // Local memory for tiles of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Accumulator for the dot product
    float Cvalue = 0.0f;

    // Number of tiles
    int numTiles = (K + TS - 1) / TS;

    // Iterate over tiles
    for (int t = 0; t < numTiles; t++) {
        // Load elements into local memory
        int tiledRow = row;
        int tiledCol = t * TS + get_local_id(0);
        if (tiledRow < M && tiledCol < K)
            Asub[get_local_id(1)][get_local_id(0)] = A[tiledRow * K + tiledCol];
        else
            Asub[get_local_id(1)][get_local_id(0)] = 0.0f;

        tiledRow = t * TS + get_local_id(1);
        tiledCol = col;
        if (tiledRow < K && tiledCol < N)
            Bsub[get_local_id(1)][get_local_id(0)] = B[tiledRow * N + tiledCol];
        else
            Bsub[get_local_id(1)][get_local_id(0)] = 0.0f;

        // Synchronize to ensure all data is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial products
        for (int k = 0; k < TS; k++) {
            Cvalue += Asub[get_local_id(1)][k] * Bsub[k][get_local_id(0)];
        }

        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}