#!/bin/bash

srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main32 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main48 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main64 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main96 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main128 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main192 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main256 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main384 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main512 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main768 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main1024 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main1536 -t 1 -n 1 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main2048 -t 1 -n 1 2048 2048 2048