#include <stdio.h>
#include <stdlib.h>


/******************************************************************
void
transpose_write_coalease(a, b, n)

int *a
int *b;
int n;

Performance GPU version of transposing a square matrix a to b with dimension = n

return: void
*/

#define ROW 3
#define COLUMN 4

__global__ void transpose_write_coalease(int * a, int *b){
  int index = threadIdx.x + blockDim.x * blockIdx.x; /*get the absolute index of thread*/
  int new_index = blockIdx.x + gridDim.x * threadIdx.x;
  b[new_index] = a[index];
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

int main(void){
  cudaSetDevice(0);

  /* Initialize */
  int n = ROW * COLUMN;
  int *a = (int *)malloc (n*sizeof(int));
  int *b = (int *)malloc(n*sizeof(int));
  for (int i = 0; i<n; i++){
    a[i] = i;
    b[i] = 0;
  }

  /* Display input */
  printMatrix(a, ROW, COLUMN);
  printf("\n");

  /* Allocate memory in GPU */
  int *dev_a;
  int *dev_b;
  cudaMalloc((void **) &dev_a, n*sizeof(int));
  cudaMalloc((void **) &dev_b, n*sizeof(int));
  cudaMemcpy(dev_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

  /* Compute */
  transpose_write_coalease<<<ROW, COLUMN>>>(dev_a, dev_b);

  /* Move result from GPU to CPU */
  cudaMemcpy(a, dev_a, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, dev_b, n*sizeof(int), cudaMemcpyDeviceToHost);
  
  /* Display result */
  printMatrix(b, COLUMN, ROW);

  /* Free the space occupied in both GPU and CPU */
  free(a);
  free(b);
  cudaFree(dev_a);
  cudaFree(dev_b);
}


