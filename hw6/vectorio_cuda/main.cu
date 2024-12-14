#include <iostream>
#include <vector>
#include <sys/time.h>

// 1G elements
#define N 1024 * 1024 * 128

// test 5 times
#define LOOPS 5

// macro for error checking
#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static double start_time[8];

static double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

using namespace std;


// naive add kernel
__global__ void add(float *A, float *B, float *C, int size){
   int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 16;

   if(tid < size){
      for(int loop = 0 ; loop < 16 ; ++loop){
         C[tid + loop] = A[tid + loop] + B[tid + loop];
      }
   }
}

// optimized (memory coalesced) add kernel
__global__ void add_coalesced(float *A, float *B, float *C, int size){
   for(int loop = (blockIdx.x * blockDim.x * 16) + threadIdx.x ; loop < (blockIdx.x * blockDim.x * 16) + blockDim.x * 16 ; loop += blockDim.x){
      C[loop] = A[loop] + B[loop];
   }  
}

// vector add kernel
__global__ void add_vector(float *A, float *B, float *C, int size){
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if((tid * 16) < size){
      float4 a1 = (reinterpret_cast<float4*>(A))[tid];
      float4 b1 = (reinterpret_cast<float4*>(B))[tid];

      float4 a2 = (reinterpret_cast<float4*>(A))[tid+1];
      float4 b2 = (reinterpret_cast<float4*>(B))[tid+1];

      float4 a3 = (reinterpret_cast<float4*>(A))[tid+2];
      float4 b3 = (reinterpret_cast<float4*>(B))[tid+2];

      float4 a4 = (reinterpret_cast<float4*>(A))[tid+3];
      float4 b4 = (reinterpret_cast<float4*>(B))[tid+3];

      (reinterpret_cast<float4*>(C))[tid] = {a1.x + b1.x, a1.y + b1.y, a1.z + b1.z, a1.w + b1.w};
      (reinterpret_cast<float4*>(C))[tid+1] = {a2.x + b2.x, a2.y + b2.y, a2.z + b2.z, a2.w + b2.w};
      (reinterpret_cast<float4*>(C))[tid+2] = {a3.x + b3.x, a3.y + b3.y, a3.z + b3.z, a3.w + b3.w};
      (reinterpret_cast<float4*>(C))[tid+3] = {a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w};
      
   }
}


int main(void){

   float *d_A, *d_B, *d_C, *d_D, *d_E; // device memory
   vector<float> A(N), B(N), C(N), D(N), E(N); // host memory

   CHECK_CUDA(cudaMalloc(&d_A, N*sizeof(float)));
   CHECK_CUDA(cudaMalloc(&d_B, N*sizeof(float)));
   CHECK_CUDA(cudaMalloc(&d_C, N*sizeof(float)));
   CHECK_CUDA(cudaMalloc(&d_D, N*sizeof(float)));
   CHECK_CUDA(cudaMalloc(&d_E, N*sizeof(float)));


   // initialize A and B   
   for(int i = 0; i < N; i++){
      A[i] = i;
      B[i] = -i;
   }

   CHECK_CUDA(cudaMemcpy(d_A, A.data(), N*sizeof(float), cudaMemcpyHostToDevice));
   CHECK_CUDA(cudaMemcpy(d_B, B.data(), N*sizeof(float), cudaMemcpyHostToDevice));
   
   dim3 block(512);
   dim3 grid(((N / 16) + block.x - 1) / block.x);

   // warm-up
   for(int loop = 0 ; loop < LOOPS ; ++loop)
      add<<<grid, block>>>(d_A, d_B, d_C, N);

   // measure the performance of the naive add kernel
   CHECK_CUDA(cudaDeviceSynchronize());
   timer_start(7);
   for(int loop = 0 ; loop < LOOPS ; ++loop){
      add<<<grid, block>>>(d_A, d_B, d_C, N);
   }
   CHECK_CUDA(cudaGetLastError());
   CHECK_CUDA(cudaDeviceSynchronize());
   cout << "Normal I/O : " << timer_stop(7) / LOOPS << " sec" << endl;
   CHECK_CUDA(cudaMemcpy(C.data(), d_C, N*sizeof(float), cudaMemcpyDeviceToHost));

   // measure the performance of the optimized add kernel
   CHECK_CUDA(cudaDeviceSynchronize());
   timer_start(7);
   for(int loop = 0 ; loop < LOOPS ; ++loop){
      add_coalesced<<<grid, block>>>(d_A, d_B, d_D, N);
   }
   CHECK_CUDA(cudaGetLastError());
   CHECK_CUDA(cudaDeviceSynchronize());
   cout << "Coalesced I/O : " << timer_stop(7) / LOOPS << " sec" << endl;
   CHECK_CUDA(cudaMemcpy(D.data(), d_D, N*sizeof(float), cudaMemcpyDeviceToHost));

   // measure the performance of the vector add kernel
   CHECK_CUDA(cudaDeviceSynchronize());
   timer_start(7);
   for(int loop = 0 ; loop < LOOPS ; ++loop){
      add_vector<<<grid, block>>>(d_A, d_B, d_E, N);
   }
   CHECK_CUDA(cudaGetLastError());
   CHECK_CUDA(cudaDeviceSynchronize());
   cout << "Vector I/O : " << timer_stop(7) / LOOPS << " sec" << endl;
   CHECK_CUDA(cudaMemcpy(E.data(), d_E, N*sizeof(float), cudaMemcpyDeviceToHost));

   // check results   
   for(int i = 0; i < N; i++){
      if(C[i] != D[i]){
         cout << "Mismatch1 at " << i << " : " << C[i] << " != " << D[i] << endl;
         break;
      }

      if(D[i] != E[i]){
         cout << "Mismatch2 at " << i << " : " << C[i] << " != " << D[i] << endl;
         break;
      }

      if(C[i] != E[i]){
         cout << "Mismatch3 at " << i << " : " << C[i] << " != " << D[i] << endl;
         break;
      }
   }
   

   CHECK_CUDA(cudaFree(d_A));
   CHECK_CUDA(cudaFree(d_B));
   CHECK_CUDA(cudaFree(d_C));
   CHECK_CUDA(cudaFree(d_D));
   CHECK_CUDA(cudaFree(d_E));

   return 0;
}
