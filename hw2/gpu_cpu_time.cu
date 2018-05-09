#include <stdio.h>
#include<stdlib.h>
#include "timerc.h"
//This program run the same program on both GPU and CPU of the server, then measure the running time of both execution

//kernel program
__global__ void mykernel(void){

	double v = 0;
	for (int i = 0; i < 1000; i++){
		for (int j = 0; j < i*i; j++){
			v = v + i + i*j/2.34;
		}
	}

}

__global__ void warp_divergence1(){

	if (threadIdx.x <16){
	double v = 0;
	for (int i = 0; i < 1000; i++){
		for (int j = 0; j < i*i; j++){
			v = v + i + i*j/2.34;
		}
	}
}else{
	double v = 0;
	for (int i = 0; i < 1000; i++){
		for (int j = 0; j < i*i; j++){
			v = v + i + i*j/2.34;
		}
	}


}
}

__global__ void warp_divergence1(){

	if (threadIdx.x <8){
	double v = 0;
	for (int i = 0; i < 1000; i++){
		for (int j = 0; j < i*i; j++){
			v = v + i + i*j/2.34;
		}
	}
}else{
	double v = 0;
	for (int i = 0; i < 1000; i++){
		for (int j = 0; j < i*i; j++){
			v = v + i + i*j/2.34;
		}
	}


}
}



int main(){


	cudaSetDevice(0);
	float time = 0;
	cstart();
	//run it on CPU
	double v = 0;
	for (int i = 0; i < 1000; i++){
		for (int j = 0; j < i; j++){
			v = v + i + i*j/2.34;
		}
	}
	cend(&time);

	printf("cpu time = %f\n",time); fflush(stdout);

	gstart();
  //run the kernel with 10 blocks and 1 threads
	mykernel<<<1,32>>>();
	gend(&time);

	printf("gpu time = %f\n",time);

	gstart();
  //run the kernel with 10 blocks and 1 threads
	warp_divergence1<<<1,32>>>();
	gend(&time);

	printf("gpu time = %f\n",time);fflush(stdout);

	gstart();
  //run the kernel with 10 blocks and 1 threads
	warp_divergence1<<<1,64>>>();
	gend(&time);

	printf("gpu time = %f\n",time);fflush(stdout);





return 0;
}
