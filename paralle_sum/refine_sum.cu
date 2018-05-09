#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define N 1024*1024
#define THREADS_PER_BLOCK 1024
 12 #define NUM_BLOCKS N/THREADS_PER_BLOCK
 13 #define MULT 32
 14 
 15 __global__ void parallel_sum_naive(int *a, int *b, int mult) {
 16   int ix = threadIdx.x + blockDim.x * blockIdx.x;
 17   int sum = 0;
 18   for (int i = 0; i < mult; i++) {
 19     sum += a[ix * mult + i];
 20   }
 21   b[ix] = sum;
 22 }
 23 
 24 // parallel reduction
 25 __global__ void parallel_sum_reduction(int *a, int *b, int block_size) {
 26   int start = blockIdx.x * block_size;
 27   int step = 1;
 28 
 29   while (step <= block_size/2) {
 30     if (threadIdx.x < (block_size / step / 2)) {
 31       a[start + 2 * threadIdx.x * step] += a[start + 2 * threadIdx.x * step + step];
 32     }
 33     __syncthreads();
 34     step *= 2;
 35   }
 36   if (threadIdx.x == 0) b[blockIdx.x] = a[start];
 37 }
 38 
 39 __global__ void parallel_sum_reduction_with_shared_mem(int *a, int *b, int block_size) {
 40   __shared__ int tmpmem[THREADS_PER_BLOCK];
 41   int step = 1;
 42 
 43   tmpmem[threadIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x];
 44   __syncthreads();
                                                                                                                         1,1           Top

