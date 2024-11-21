#define TS 64
#define WPT 8
#define RTS TS/WPT

__kernel void sgemm(__global const float *A, __global const float *B, __global float *C,
                    int M, int N, int K) {
    const int row = get_global_id(1); // 0..((M+TS-1) / TS) * RTS
    const int col = get_global_id(0); // 0..((N+TS-1) / TS) * RTS

    __local float Asub[TS][TS], Bsub[TS][TS];

    float Cvalue[RTS][RTS];
    for (int wi = 0; wi < RTS; wi++) {
        for (int wj = 0; wj < RTS; wj++) {
            Cvalue[wi][wj] = 0.0f;
        }
    }

    int rowWorkA, colWorkA, rowWorkB, colWorkB;
    int cRow, cCol;

    int numTiles = (K + RTS - 1) / RTS;

    for (int t = 0; t < numTiles; t++) {

        for (int wRow = 0; wRow < RTS; wRow++) {
            for (int wCol = 0; wCol < RTS; wCol++) {

                rowWorkA = row * WPT + wRow;
                colWorkA = (t * TS + get_local_id(0)) * WPT + wCol;

                if (rowWorkA < M && colWorkA < K)
                    Asub[get_local_id(1) * WPT + wRow][get_local_id(0) * WPT + wCol] = A[rowWorkA * K + colWorkA];
                else
                    Asub[get_local_id(1) * WPT + wRow][get_local_id(0) * WPT + wCol] = 0.0f;

                rowWorkB = (t * TS + get_local_id(1)) * WPT + wRow;
                colWorkB = col * WPT + wCol;
                
                if (rowWorkB < K && colWorkB < N)
                    Bsub[get_local_id(1) * WPT + wRow][get_local_id(0) * WPT + wCol] = B[rowWorkB * N + colWorkB];
                else
                    Bsub[get_local_id(1) * WPT + wRow][get_local_id(0) * WPT + wCol] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < RTS; k++) {
            for (int wi = 0; wi < RTS; wi++) {
                for (int wj = 0; wj < RTS; wj++) {
                    Cvalue[wi][wj] += Asub[get_local_id(1) * WPT + wi][k] * Bsub[k][get_local_id(0) * WPT + wj];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int wi = 0; wi < RTS; wi++) {
        for (int wj = 0; wj < RTS; wj++) {
            int cRow = row * WPT + wi;
            int cCol = col * WPT + wj;
            if (cRow < M && cCol < N) {
                C[cRow * N + cCol] = Cvalue[wi][wj];
            }
        }
    }
}
