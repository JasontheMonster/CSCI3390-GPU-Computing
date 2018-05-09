#include <stdio.h>
#include "timerc.h"
//This program run the same program on both GPU and CPU of the server, then measure the running time of both execution

//kernel program
__global__ void mykernel(void){

	double v = 0;
	for (int i = 0; i < 1000; i++){
    printf("test");
		for (int j = 0; j < i*i; j++){
			v = v + i + i*j/2.34;		
		}	
	}

}



int main(){

	float time;
	gstart();
  //run the kernel with 10 blocks and 1 threads
	mykernel<<<10,1>>>();
	gend(&time);

	printf("gpu time = %f\n",time);

	cstart();
  //run it on CPU
	double v = 0;
	for (int i = 0; i < 1000; i++){
		for (int j = 0; j < i; j++){
			v = v + i + i*j/2.34;		
		}	
	}
	cend(&time);

	printf("cpu time = %f\n",time);


return 0;
}
