/*Shikun Wang (Jason)*/

#include<stdio.h>
#include "timerc.h"


#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__host__ __device__ void printgame(int* game, int dim){
    for (int y = 0; y < dim ; y++){
        for (int x = 0 ; x < dim ; x++){
            printf("%d ", game[y*dim + x]);
        }
        printf("\n");
    }
    printf("\n");
}


__host__ __device__ inline int positive_mod(int s, int m){
    
    if (s >=0){
        return s % m;
    }else{
        return m + (s % m);
    }
    
}

__host__ __device__ int countneigh(int *game, int x, int y, int dim){
    
    int n = 0;
    
    int xp1 = positive_mod(x+1, dim);
    int xm1 = positive_mod(x-1, dim);
    int yp1 = positive_mod(y+1, dim);
    int ym1 = positive_mod(y-1, dim);
    
    n = game[y*dim   + xm1] +
        game[y*dim   + xp1] +
        game[yp1*dim + x] +
        game[ym1*dim + x]+
        game[ym1*dim + xm1] +
        game[yp1*dim + xp1] +
        game[yp1*dim + xm1] +
        game[ym1*dim + xp1] ;
    
    return n;
    
}


void setrandomconfi(int *game, int dim, float p){
    
    for (int i = 0 ; i < dim*dim ; i++){
        game[i] = ((double) rand() / (RAND_MAX)) < p;
    }
}

void play_game_cpu(int *game_new, int * game_old, int dim){
    
    // there order, either y first or x first, affects speed of the CPU code quite a bit
    for (int y = 0; y < dim ; y++){
        for (int x = 0 ; x < dim ; x++){
            
            // first copy input to output. Then make transitions.
            game_new[y*dim + x] = game_old[y*dim + x];
            
            int num_neigh_cells = countneigh(game_old, x, y, dim);

            //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
            //Any live cell with more than three live neighbours dies, as if by overpopulation.
            if (game_old[y*dim + x] == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
                game_new[y*dim + x] = 0;
            }
            //Any live cell with two or three live neighbours lives on to the next generation.
            if (game_old[y*dim + x] ==1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
                game_new[y*dim + x] = 1;
            }
            //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
            if (game_old[y*dim + x] == 0 && num_neigh_cells == 3){
                game_new[y*dim + x] = 1;
            }
            
        }
    }
}

__global__ void play_game_gpu_simple(int *game_new, int * game_old, int dim){
    
    /*get the index of x and y*/
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    // first copy input to output. Then make transitions.
    
    game_new[y*dim + x] = game_old[y*dim + x];
    
    int num_neigh_cells = countneigh(game_old, x, y, dim);
    
    //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
    //Any live cell with more than three live neighbours dies, as if by overpopulation.
    if (game_old[y*dim + x] == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
        game_new[y*dim + x] = 0;
    }
    //Any live cell with two or three live neighbours lives on to the next generation.
    if (game_old[y*dim + x] ==1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
        game_new[y*dim + x] = 1;
    }
    //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    if (game_old[y*dim + x] == 0 && num_neigh_cells == 3){
        game_new[y*dim + x] = 1;
    }
    
    
    
}

int main(){

    float cpu_time;

    int gamedim = 1024;
    int gamesize = gamedim*gamedim;
    
    
    
    int num_iterations = 100;
    
    int * h_game_1 = (int *) malloc( sizeof(int)  * gamesize  );
    int * h_game_2 = (int *) malloc( sizeof(int)  * gamesize  );
    
    int * dev_game1;
    int * dev_game2;
    
    
    cudaMalloc(&dev_game1, sizeof(int)*gamesize);
    cudaMalloc(&dev_game2, sizeof(int)*gamesize);
    
    setrandomconfi(h_game_1, gamedim,0.6);
    //printgame(h_game_1,gamedim);
    cudaMemcpy(dev_game1, h_game_1, sizeof(int)*gamesize, cudaMemcpyHostToDevice);
    
    
    // this evolves the game in the CPU
    //printgame(h_game_1,gamedim);
    cstart();
    for (int t = 1; t <= num_iterations/2 ; t++){
        play_game_cpu(h_game_2, h_game_1, gamedim);
        play_game_cpu(h_game_1, h_game_2, gamedim);

    }
    cend(&cpu_time);

    dim3 threads_per_block(32, 32);
    dim3 num_blocks((gamedim + 31) / 32,(gamedim+31) / 32);
    
    float gpu_time;
    gstart();
    for (int t = 1; t <= num_iterations/2 ; t++){
        play_game_gpu_simple<<<num_blocks, threads_per_block>>>(dev_game2, dev_game1, gamedim);
        play_game_gpu_simple<<<num_blocks, threads_per_block>>>(dev_game1, dev_game2, gamedim);
        
    }
    gend(&gpu_time);
    
    cudaMemcpy(h_game_2, dev_game1, sizeof(int)*gamesize, cudaMemcpyDeviceToHost);
    
    int r = 0;
    for(int i = 0; i < gamesize;i++)
    {
        r += positive_mod(h_game_2[i] - h_game_1[i], 2);
    }
    
    printf("Error = %d\n", r);
    
    printf("Time to run game on the CPU = %f\n",cpu_time);
    printf("Time to run game on the GPU = %f\n",gpu_time);

    free(h_game_1);
    free(h_game_2);

	return 0;
}
