# mm00.cu

shpc159@elogin3:~/hw5/matmul_multi$ ./run.sh 4096 4096 4096 -n 5 -v
Options:
  Problem size: M = 4096, N = 4096, K = 4096
  Number of iterations: 5
  Print matrix: off
  Validation: on

Initializing matrices...Done!
Using 4 devices
GPU 0: NVIDIA TITAN RTX
GPU 1: NVIDIA TITAN RTX
GPU 2: NVIDIA TITAN RTX
GPU 3: NVIDIA TITAN RTX
Calculating...(iter=0) 4.755904 sec
Calculating...(iter=1) 4.756084 sec
Calculating...(iter=2) 4.756259 sec
Calculating...(iter=3) 4.744249 sec
Calculating...(iter=4) 4.728465 sec
Validating...
Result: VALID
Avg. time: 4.748192 sec
Avg. throughput: 28.945532 GFLOPS

# mm01.cu
shpc159@elogin3:~/hw5/matmul_multi$ ./run.sh 4096 4096 4096 -n 5 -v
srun: job 1054209 queued and waiting for resources
srun: job 1054209 has been allocated resources
Options:
  Problem size: M = 4096, N = 4096, K = 4096
  Number of iterations: 5
  Print matrix: off
  Validation: on

Initializing matrices...Done!
Using 4 devices
GPU 0: NVIDIA TITAN RTX
GPU 1: NVIDIA TITAN RTX
GPU 2: NVIDIA TITAN RTX
GPU 3: NVIDIA TITAN RTX
Calculating...(iter=0) 0.108153 sec
Calculating...(iter=1) 0.107683 sec
Calculating...(iter=2) 0.108002 sec
Calculating...(iter=3) 0.107824 sec
Calculating...(iter=4) 0.141875 sec
Validating...
Result: VALID
Avg. time: 0.114707 sec
Avg. throughput: 1198.172767 GFLOPS


# mm02.cu
shpc159@elogin3:~/hw5/matmul_multi$ ./run.sh 
srun: job 1079809 queued and waiting for resources
srun: job 1079809 has been allocated resources
Options:
  Problem size: M = 16384, N = 4096, K = 4096
  Number of iterations: 5
  Print matrix: off
  Validation: on

Initializing matrices...Done!
Using 4 devices
GPU 0: NVIDIA TITAN RTX
GPU 1: NVIDIA TITAN RTX
GPU 2: NVIDIA TITAN RTX
GPU 3: NVIDIA TITAN RTX
Calculating...(iter=0) 0.244921 sec
Calculating...(iter=1) 0.286246 sec
Calculating...(iter=2) 0.236094 sec
Calculating...(iter=3) 0.253223 sec
Calculating...(iter=4) 0.240404 sec
Validating...
Result: VALID
Avg. time: 0.252178 sec
Avg. throughput: 2180.032840 GFLOPS