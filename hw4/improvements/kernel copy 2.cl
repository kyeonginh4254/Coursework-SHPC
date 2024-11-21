#define TS 64
#define WPT 8
#define RTS TS/WPT

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                    int M, int N, int K) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    int tiledRow, tiledCol, tiledWorkRow, tiledWorkCol;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float Cvalue[RTS][RTS];
    for (int wi = 0; wi < RTS; wi++) {
        for (int wj = 0; wj < RTS; wj++) {
            Cvalue[wi][wj] = 0.0f;
        }
    }

    int numTiles = (K + RTS - 1) / RTS;

    for (int t = 0; t < numTiles; t++) {

        tiledRowA = row;
        tiledColA = t * TS + get_local_id(0);
        tiledRowB = t * TS + get_local_id(1);
        tiledColB = col;

        for (int wRow = 0; wRow < RTS; wRow++) {
            for (int wCol =0; wCol < RTS; wCol++) {

                tiledWorkRowA = tiledRowA + wRow;
                tiledWorkColA = tiledColA + wCol;
                if (tiledWorkRowA < M && tiledWorkColA < K)
                    Asub[get_local_id(1) + wRow][get_local_id(0) + wCol] = A[tiledWorkRowA * K + tiledWorkColA];
                else
                    Asub[get_local_id(1) + wRow][get_local_id(0) + wCol] = 0.0f;
                
                tiledWorkRowB = tiledRowB + wRow;
                tiledWorkColB = tiledColB + wCol;
                if (tiledWorkRowB < K && tiledWorkRowB < N)
                    Bsub[get_local_id(1) + wRow][get_local_id(0) + wCol] = AB[tiledWorkRowB * N + tiledWorkColB];
                else
                    Bsub[get_local_id(1) + wRow][get_local_id(0) + wCol] = 0.0f;
            
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < RTS; k++) {
            for (int wRow = 0; wRow < RTS; wRow++) {
                for (int wCol = 0; wCol < RTS; wCol++) {
                    Cvalue[wRow][wCol] += Asub[get_local_id(1) + wRow][k] * Bsub[k][get_local_id(0) + wCol];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result
    for (int w = 0; wRow < RTS; wRow++) {
        for (int wCol = 0; wCol < RTS; wCol++) {
            if (row < M && col < N) {
                C[row * N + col] = Cvalue;
            }
        }
    }
}
