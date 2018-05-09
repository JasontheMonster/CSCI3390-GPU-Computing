#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

/******************************************************************
void
transpose_write_coalease(a, b, n)

int *a
int *b;
int n;

Performance GPU version of transposing a square matrix a to b with dimension = n

return: void
*/

#define DIM 32 /*dimension of patch*/

__global__ void transpose_read_coalease(int *a, int *b, int n){
  int x_index = threadIdx.x + blockIdx.x*blockDim.x; 
  int y_index = threadIdx.y + blockIdx.y*blockDim.y;
  
  int index = y_index*n+x_index; 
  int new_index = x_index*n + y_index;

  b[new_index] = a[index];
}

__global__ void transpose_shared_memory(int *a, int *b, int n){
  __shared__ int tile [DIM*DIM]; /*shared memory block */
  int x_index = threadIdx.x + blockIdx.x*blockDim.x; 
  int y_index = threadIdx.y + blockIdx.y*blockDim.y;
  

  /**transposing the block **/
  int x_new_index = threadIdx.x + blockIdx.y*blockDim.x;
  int y_new_index = threadIdx.y + blockIdx.x*blockDim.y;


  int index = y_index*n+x_index; 
  int new_index = x_new_index + y_new_index*n;

  tile[threadIdx.y + DIM*threadIdx.x] = a[index];

  __syncthreads();

  b[new_index] = tile[threadIdx.x + DIM*threadIdx.y];
}


void printMatrix(int* m, int a, int b) {
   int i, j;
    for (i = 0; i < a; i++) {
      for (j = 0; j < b; j++) {
        printf("%d\t", *(m + (i * b + j)));
      }
      printf("\n");
   }
 }
int check_correctness(int *a, int *b, int row, int col){
  int boolean = 1;
  for (int i = 0; i< row; i++){
    for (int j =0; j < col; j++){
        if (*(a+ (i*col + j))!= *(b+(j*col + i))){
          printf("a: %d\n", a[i*col+j]);
          printf("b: %d\n", b[j*col+i]);
          boolean = 0;
          return boolean;    
         } 
      }
    }
    return boolean;
}

int main(void){
  cudaSetDevice(0);

  /* Initialize */
  //int dim = NUMOFBLOCK*BLOCKDIM;
  int size = DIM*256;
  int n = size *size;

  int *a = (int *)malloc (n*sizeof(int));
  int *b = (int *)malloc(n*sizeof(int));
  for (int i = 0; i<n; i++){
    a[i] = i;
    b[i] = 0;
  }

  /* Display input */
  //printMatrix(a, ROW, COLUMN);
  //printf("\n");

  /* Allocate memory in GPU */
  int *dev_a;
  int *dev_b;
  cudaMalloc((void **) &dev_a, n*sizeof(int));
  cudaMalloc((void **) &dev_b, n*sizeof(int));
  cudaMemcpy(dev_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, n*sizeof(int), cudaMemcpyHostToDevice);


  dim3 numofblocks(size/DIM, size/DIM);
  dim3 numfofthreads(DIM, DIM);

  float time;
  gstart();
  /* Compute */
  transpose_shared_memory<<<numofblocks, numfofthreads>>>(dev_a, dev_b, size);

  gend(&time);

  printf("shared memory time %f\n", time);

  gstart();
  /* Compute */
  transpose_read_coalease<<<numofblocks, numfofthreads>>>(dev_a, dev_b, size);

  gend(&time);

  printf("global memory time %f\n", time);
  /* Move result from GPU to CPU */
  cudaMemcpy(a, dev_a, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, dev_b, n*sizeof(int), cudaMemcpyDeviceToHost);
  
  /* Display result */
  /* Free the space occupied in both GPU and CPU */
  free(a);
  free(b);
  cudaFree(dev_a);
  cudaFree(dev_b);
}


