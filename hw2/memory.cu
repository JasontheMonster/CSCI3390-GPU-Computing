#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"


#define gerror(ans) { gpuAssert ((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{ 
  if (code !=cudaSuccess)
  {
    fprintf(stderr, "GPUassert:%s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


__device__ int a[1];

__global__ void printfaddress(int *r){
  printf("Content of address %p from GPU = %d\n", r, r[0]);
}

__global__ void increase_memory(int *r, int n){
  int index = threadIdx.x + blockDim.x*blockIdx.x;

  if(index <n){
    r[index] = r[index] +index;
  }
}


int main(void){
  cudaSetDevice(0);


  int *dev_ptr;
  int n = 128;

  int *host_ptr = (int *) malloc(n*sizeof(int));
  for (int i = 0; i <n; i++){
    host_ptr[i] = i;
  }

  cudaMalloc((void**)&dev_ptr, sizeof(int)*n);
  cudaMemcpy(dev_ptr, host_ptr, sizeof(int)*n, cudaMemcpyHostToDevice);

  dim3 numthreadsperblock (1024, 1);
  dim3 numblockspergrid((n+1023)/1024, 1);

  increase_memory<<<numblockspergrid, numthreadsperblock>>>(dev_ptr, n);

  cudaMemcpy( host_ptr, dev_ptr, sizeof(int)*n, cudaMemcpyDeviceToHost);

  for (int i =0; i <n; i++){
    printf("%d ", host_ptr[i]);
  }
  printf("\n");
  free(host_ptr);
  cudaFree(dev_ptr); 

  int a_host = 66;

  printf("Address of a from CPUs = %p\n", a);

  int *address_of_a;
  cudaGetSymbolAddress(( void **) &address_of_a, a);

  printf("Address of a from GPU = %p\n", address_of_a);

  cudaMemcpyToSymbol(a, &a_host, sizeof(int), 0, cudaMemcpyHostToDevice);

  printf("Content of address from GPU = %d\n", a[0]);

  printfaddress<<<1, 2>>>( address_of_a);
  cudaDeviceSynchronize();

  gerror( cudaPeekAtLastError() );
  cudaDeviceSynchronize();


}
