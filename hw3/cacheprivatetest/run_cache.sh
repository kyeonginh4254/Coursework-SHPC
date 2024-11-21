#!/bin/bash

THREAD_COUNTS=(1 16)

# 행렬 크기 설정
M=2048
N=2048
K=2048

for t in "${THREAD_COUNTS[@]}"
do
  echo "\n\n>>> Running with BLOCK_SIZE=32, $t threads..."
  srun --nodes=1 --exclusive numactl --physcpubind 0-63 ./main32 -t $t $M $N $K
done

for t in "${THREAD_COUNTS[@]}"
do
  echo "\n\n>>> Running with BLOCK_SIZE=128, $t threads..."
  srun --nodes=1 --exclusive numactl --physcpubind 0-63 ./main128 -t $t $M $N $K
done

for t in "${THREAD_COUNTS[@]}"
do
  echo "\n\n>>> Running with BLOCK_SIZE=1024, $t threads..."
  srun --nodes=1 --exclusive numactl --physcpubind 0-63 ./main1024 -t $t $M $N $K
done