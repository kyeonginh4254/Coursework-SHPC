#!/bin/bash

THREAD_COUNTS=(1 2 4 8 16 32 64 128 256)

# 행렬 크기 설정
M=1024
N=1024
K=1024

for t in "${THREAD_COUNTS[@]}"
do
  echo "Running with $t threads..."
  srun --nodes=1 --exclusive numactl --physcpubind 0-63 ./main -t $t $M $N $K
done