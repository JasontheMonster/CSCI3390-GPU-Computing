#include <stdio.h>
#include <stdlib.h>
#include "timerc.h"

#define SIZE 64/*size of array*/
#define THREADSOFBLOCK 16/*number of threads per block*/


__global__ void comulative_sum_2(int *a, int *b){
    /*get the starting address of each block of data*/
    int starting_addr = 2*blockDim.x*blockIdx.x;
    int length_of_block = 2*blockDim.x;
    

    for (int step = 1; step <= blockDim.x; step *=2){
        int index = starting_addr+step + 2*step*(threadIdx.x/step) + (threadIdx.x % step);
        int shift_back_index = (threadIdx.x % step) + 1;
        a[index] += a[index - shift_back_index];
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0){
        b[blockIdx.x] = a[(starting_addr+length_of_block*(blockIdx.x+1)-1)/2];
    }
}





__global__ void cumulative_sum(int *a, int*b){
  /*get the starting address of each block of data*/
  int starting_addr = 2*blockDim.x*blockIdx.x;
  int length_of_block = 2*blockDim.x;


  /*phase one*/
  int step_size = 1; /*start by adding contagious element*/
  while(step_size<= length_of_block /2){
      /*use one threads to add two element*/
      if(threadIdx.x < blockDim.x / step_size){
        int first_group = 2*step_size-1;
        int offset = 2*step_size*threadIdx.x;
        a[starting_addr + first_group + offset] += a[starting_addr + step_size -1 + threadIdx.x*step_size*2];
      }
      __syncthreads();
      step_size*=2;
  }
  /*phase two*/

  int number_of_ops = 1;
  int back_step_size = length_of_block /2; /*start by dealing with last two element*/

  while(back_step_size>=2){
    if(threadIdx.x < 2*number_of_ops -1){
      a[starting_addr + back_step_size - 1 + (back_step_size / 2) + threadIdx.x * back_step_size] += a[starting_addr + back_step_size - 1 + threadIdx.x * back_step_size];
    }
    __syncthreads();
    number_of_ops*=2;
    back_step_size /=2;
  }

  /*phase three*/
  /*put last element of each block to b for finalized*/
  if(threadIdx.x ==0){
    b[blockIdx.x] = a[starting_addr + blockDim.x*2-1];
  }
}

/*add element of blocks by last element of the last block*/
__global__ void finalized_sum(int *a, int *b, int length_of_block){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  
  if(idx >= length_of_block){
    a[idx] += b[blockIdx.x-1];
  }
  __syncthreads();

}


int main(){
    /*initialize host and device space*/
    int *host_input_ptr = (int *) malloc (SIZE*sizeof(int));
    int *device_input; /*only need input. not output*/
    int *device_output;
    /*initialized sizes*/
    int length_of_block = 2 * THREADSOFBLOCK;
    /*trick to use length of block derive number of blocks*/
    
    int number_of_blocks = (SIZE + length_of_block-1) / length_of_block;
    int *host_output_ptr = (int *) malloc (number_of_blocks*sizeof(int));
   
    printf("%d, ", number_of_blocks);
    cudaMalloc(&device_input, SIZE*sizeof(int));
    cudaMalloc(&device_output,number_of_blocks*sizeof(int));

    float prepare_time;
    float finalized_time;
    float cpuTime;

    /*initialized input array*/

    for(int i = 0; i < SIZE; i++) host_input_ptr[i] = 1;

    /*move host input array to device and perform kernel*/
    cudaMemcpy(device_input, host_input_ptr, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    gstart();
  
    //cumulative_sum<<<number_of_blocks, THREADSOFBLOCK>>>(device_input, device_output);
    comulative_sum_2<<<number_of_blocks, THREADSOFBLOCK>>>(device_input, device_output);
    //comulative_sum_2<<<1, SIZE/THREADSOFBLOCK/2>>>(device_output, device_output);


    cudaMemcpy(host_input_ptr, device_input, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_output_ptr, device_output, number_of_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    gend(&prepare_time);

    printf("GPU took %f seconds\n", prepare_time);
   

    
    //for (int i = 0; i < number_of_blocks; i++) printf("%d, ", host_output_ptr[i]);
    
    
    /*GPU version 
    for (int i = 0; i <number_of_blocks; i++) host_output_ptr[i+1] += host_output_ptr[i];
    
    gstart();
    cudaMemcpy(device_output, host_output_ptr, number_of_blocks*sizeof(int), cudaMemcpyHostToDevice);
    finalized_sum<<<number_of_blocks, THREADSOFBLOCK>>>(device_input, device_output, length_of_block);

    cudaMemcpy(host_input_ptr, device_input, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    gend(&finalized_time);  

    printf("it took %f seconds\n", finalized_time);

    */
    
    
    cstart();
    for (int i = 1; i < number_of_blocks; i++){
      host_output_ptr[i] += host_output_ptr[i-1];
    }
    
  

    for (int i = 0; i< number_of_blocks; i++) 
    printf("%d, ", host_output_ptr[i]);
     

    for (int j = length_of_block; j < SIZE; j++){
        int offset = j/ length_of_block-1;
        host_input_ptr[j] += host_output_ptr[offset+1];
        printf("%d, ", host_input_ptr[j]); 
    }
  

    cend(&finalized_time);

    printf("CPU summing time, %f\n", finalized_time);
    
    printf("total time: %f \n", finalized_time+prepare_time);
    int counter = 0;
    
    for(int i = 0; i < SIZE; i++) {
       printf("%d, ", host_input_ptr[i]);
       if (host_input_ptr[i] != i+1) counter++;
    }

    if (counter) printf("There are %d errors", counter);
    else printf("success");
    printf("\n");


    /*free heap space*/
    free(host_output_ptr);
    free(host_input_ptr);
    cudaFree(device_input);
    cudaFree(device_output);
}   
