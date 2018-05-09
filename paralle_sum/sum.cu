#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define N 1024*1024
#define THREADSOFBLOCK 1024
#define MULT 32


#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void parallel_sum_1(int *input, int *output, int n, int blocksize){
     int index = threadIdx.x + blockIdx.x*blockDim.x; /*get the absolute index of threads*/

     output[index] = 0;
     for (int i = 0; i<blocksize; i++){
        output[index] += input[blocksize*index + i];
     }
}

/*parellel sum with consecutive memory access*/
__global__ void reduction_sum_refine(int *input, int *output, int n){
    int sizeofblock = n / gridDim.x; /*size of block by handled by a thread block*/

    int start = threadIdx.x*sizeofblock;

    int step = sizeofblock /2;

    while (step<=1){
      if (threadIdx.x <step){
        input[start+threadIdx.x] +=input[start+threadIdx.x+step];
      }
      __syncthreads();
      step = step/2;
      printf("%d ", step);
    }
    if(threadIdx.x == 0) output[blockIdx.x] = input[start];
}



/*using share memory with consecutive access memory*/
__global__ void parallel_sum_reduction_consecutive_with_shared_mem(int *a, int *b, int block_size) {
   __shared__ int tmpmem[THREADSOFBLOCK];
   int step = block_size / 2;

   tmpmem[threadIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x];
   __syncthreads();

   while (step >= 1) {
     if (threadIdx.x < step) {
       tmpmem[threadIdx.x] += tmpmem[threadIdx.x + step];
     }
     __syncthreads();
     step /= 2;
   }
   if (threadIdx.x == 0) b[blockIdx.x] = tmpmem[0];
}


/*share memory with add element at the begining*/
__global__ void parallel_sum_coal(int *a, int *b, int block_size) {
   __shared__ int tmpmem[THREADSOFBLOCK];
   int step = block_size / 2;
   int startingAddr = 2*blockDim.x*blockIdx.x;


   tmpmem[threadIdx.x] = a[startingAddr + blockDim.x + threadIdx.x] + a[startingAddr + threadIdx.x];
   __syncthreads();

   while (step >= 1) {
     if (threadIdx.x < step) {
       tmpmem[threadIdx.x] += tmpmem[threadIdx.x + step];
     }
     __syncthreads();
     step /= 2;
   }
   if (threadIdx.x == 0) b[blockIdx.x] = tmpmem[0];
 }






int main(){
  cudaSetDevice(0);

  float cpuTime, gpuSumTime, gpuSetupTime, gpuTotalTime;
  int blocksize = N/ THREADSOFBLOCK;
  /*initialized*/
  /*allocate a size of n array to sum*/
  int *host_ptr = (int *) malloc(N*sizeof(int));
  /*allocate a size of number of threads array to put our sum*/
  int *host_out_ptr = (int *) malloc((N/MULT)*sizeof(int));
  int *dev_ptr;
  int *dev_out_ptr;
  for(int i=0; i<N; i++) {
      host_ptr[i] = i;
  }
  int expectedSum = (N*(N-1))/2;
  int cpuSum=0;
  int gpuSum =0;

  /*time for CPU sum*/
  cstart();
  for(int i=0; i<N; i++) {
       cpuSum += host_ptr[i];
  }
  cend(&cpuTime);
  printf("CPU time: %f\n", cpuTime);

  /*allocation space at device*/
  cudaMalloc(&dev_ptr, N*sizeof(int));
  cudaMalloc(&dev_out_ptr, N/MULT*sizeof(int));

  gpuTotalTime = 0.0;
  /*test for gpu time*/
  gstart();
  cudaMemcpy(dev_ptr, host_ptr, N*sizeof(int), cudaMemcpyHostToDevice);
  gend(&gpuSetupTime);

  gpuTotalTime+= gpuSetupTime;

  gstart();
  parallel_sum_coal<<<512, 1024>>>(dev_ptr, dev_out_ptr, n);
  //parallel_sum_a1<<<128, 128>>>(dev_ptr, dev_out_ptr, n, blocksize);
  gend(&gpuSumTime);

  /*time for moving out from device and calculation of time*/
  gstart();
  cudaMemcpy(host_out_ptr, dev_out_ptr, (N/MULT)*sizeof(int), cudaMemcpyDeviceToHost);
  gpuSum=0;
    for(int i=0; i<numofThreads; i++) {
        gpuSum += host_out_ptr[i];
  }
  gend(&gpuAssert);
  printf("GPU time1: %f, time2: %f, total: %f\n", gpuAssert, gpuSumTime, gpuAssert+gpuSumTime);

  if(expectedSum != cpuSum) printf("CPU error!\n");
  if(expectedSum != gpuSum) printf("GPU error!\n");

  free(host_ptr);
  free(host_out_ptr);
  cudaFree(dev_ptr);
  cudaFree(dev_out_ptr);

  gerror( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

}
