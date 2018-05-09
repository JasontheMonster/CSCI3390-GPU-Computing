
/*author: Shikun Wang*/
#include <stdio.h>
#include "timerc.h"

__global__ void hello(int size, int depth)
{ printf("block: %d thread: %d depth: %d\n",blockIdx.x,threadIdx.x,depth);
  if(size==1)
	return;
  if(threadIdx.x==0&&blockIdx.x==0)
{printf("callling from depth %d\n", depth);
hello<<<2,size/2>>>(size/2,depth+1);
}
cudaDeviceSynchronize();
printf("finish from depth %d \n",depth); 


}

__global__ void naive_recursive_sum(int* v,int n,int start)
{
    if(n==1)
    {
      if(threadIdx.x==0)
          v[start]=v[start]+v[start+n];
    }
    
    if(n>=2)
    {
        naive_recursive_sum<<<1,2>>>(v,n/2,start+threadIdx.x*n);
        __syncthreads();
        if(threadIdx.x==0)
        {
            cudaDeviceSynchronize();
            v[start]+=v[start+n];
            
        }
        
    }
    

}


//better recursive sum
__global__ void better_recursive_sum(int* v_input,int* v_output, int n, int flag)
{
	//store the original block numbers the thread is in
    if (n ==64){
        flag = blockIdx.x;
    }

    //use for recursion hodler
    int* v_input_fixed=v_input+blockIdx.x*n;
    
    //bash case: add the two value and store in the ouput[blockID]
    if(n==2 && threadIdx.x==0){
        v_output[flag] = v_input_fixed[0] + v_input_fixed[1];
    }else{
        //copy the element from nextblock by adding up
        int s=n/2;
        if(threadIdx.x<s){
            v_input_fixed[threadIdx.x]+=v_input_fixed[threadIdx.x+s];
        }
        //make sure threads are all finished copying
        __syncthreads();
        
        //thread 0 call another kernel recursively
        if(threadIdx.x==0)
        {    better_recursive_sum<<<1,s/2>>>(v_input_fixed,v_output, n/2, flag);
            
        }
    }
}


int main()
{
    //initialzied array
    int k = 10;
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 1024);
    int n = 1<<(k);
    int blocksize = 32;
    int numberofblock = (n+blocksize-1) / blocksize;
    //print correct answer
    printf("expected sum: %d\n",n);
    
    float gpu2Time;
    float cpu2Time;

    //prepare input
    int * host_v= (int *) malloc(sizeof(int) * n);
    int * device_v;
    cudaMalloc((void**) &device_v, sizeof(int) * n);
    for(int i=0 ;i<n; i++)
        host_v[i] = 1;
    
    gstart();
    //prepare output buffer
    int * host_output = (int *) malloc(sizeof(int)*numberofblock);
    int * dev_output;
    cudaMalloc((void **) &dev_output, sizeof(int) * numberofblock);
   
    cudaMemcpy(device_v, host_v, sizeof(int)*n,cudaMemcpyHostToDevice);
    
    //call kernel
    //naive_recursive_sum<<<1,2>>>(device_v,n/2,0);
    better_recursive_sum<<<numberofblock, blocksize>>>(device_v,dev_output, 64, 0);
    
    cudaMemcpy(host_output, dev_output, sizeof(int) * numberofblock, cudaMemcpyDeviceToHost);

    gend(&gpu2Time);
    /*naive finisher*/
    cstart();
    int total = 0;
    //cudaMemcpy(&total,device_v,sizeof(int),cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numberofblock; i++){
        total+= host_output[i];
    }
    printf("my answer is %d \n",total);
    cend(&cpu2Time);
    
    printf("The gputime using less naive is %f, and the CPU completion time is %f\n", gpu2Time, cpu2Time);
    
        
    
cudaDeviceSynchronize();
return 0;




}
