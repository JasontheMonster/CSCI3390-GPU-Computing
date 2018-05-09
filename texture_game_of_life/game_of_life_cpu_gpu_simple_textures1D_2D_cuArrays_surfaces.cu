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

texture<int> game1_tex;
texture<int> game2_tex;

texture<int, 2> game1_tex2D;
texture<int, 2> game2_tex2D;

texture<int, 2> game1_tex2D_array;
texture<int, 2> game2_tex2D_array;

surface<void, 2> game1_surface;
surface<void, 2> game2_surface;


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
    
    // there order, either y first or x first, affects speed of the CPU code quite a bit
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

__global__ void play_game_gpu_texture1D(int *game_new, int dim, int dir){
    
    
    // there order, either y first or x first, affects speed of the CPU code quite a bit
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    if( dir == 1) {
        // first copy input to output. Then make transitions.
        int s = tex1Dfetch(game1_tex, y*dim+x);
        game_new[y*dim + x] = s;
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex1Dfetch(game1_tex,y*dim   + xm1) +
        tex1Dfetch(game1_tex,y*dim   + xp1) +
        tex1Dfetch(game1_tex,yp1*dim + x) +
        tex1Dfetch(game1_tex,ym1*dim + x)+
        tex1Dfetch(game1_tex,ym1*dim + xm1) +
        tex1Dfetch(game1_tex,yp1*dim + xp1) +
        tex1Dfetch(game1_tex,yp1*dim + xm1) +
        tex1Dfetch(game1_tex,ym1*dim + xp1) ;
        

        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }

        
        
    }
    else {
        // first copy input to output. Then make transitions.
        int s = tex1Dfetch(game2_tex, y*dim+x);
        game_new[y*dim + x] = s;
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex1Dfetch(game2_tex,y*dim   + xm1) +
        tex1Dfetch(game2_tex,y*dim   + xp1) +
        tex1Dfetch(game2_tex,yp1*dim + x) +
        tex1Dfetch(game2_tex,ym1*dim + x)+
        tex1Dfetch(game2_tex,ym1*dim + xm1) +
        tex1Dfetch(game2_tex,yp1*dim + xp1) +
        tex1Dfetch(game2_tex,yp1*dim + xm1) +
        tex1Dfetch(game2_tex,ym1*dim + xp1) ;
        
        
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }

        
    }
   }

__global__ void play_game_gpu_texture2D(int *game_new, int dim, int dir){
    
    
    // there order, either y first or x first, affects speed of the CPU code quite a bit
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    if( dir == 1) {
        // first copy input to output. Then make transitions.
        int s = tex2D(game1_tex2D, x,y);
        game_new[y*dim + x] = s;
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex2D(game1_tex2D,xm1, y) +
        tex2D(game1_tex2D,xp1, y) +
        tex2D(game1_tex2D,x,yp1) +
        tex2D(game1_tex2D,x, ym1)+
        tex2D(game1_tex2D,xm1, ym1) +
        tex2D(game1_tex2D,xp1, ym1) +
        tex2D(game1_tex2D,xm1,yp1) +
        tex2D(game1_tex2D,xp1,yp1) ;
        
        
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }
        
        
        
    }
    else {
        // first copy input to output. Then make transitions.
        int s = tex2D(game2_tex2D, x,y);
        game_new[y*dim + x] = s;
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex2D(game2_tex2D,xm1, y) +
        tex2D(game2_tex2D,xp1, y) +
        tex2D(game2_tex2D,x,yp1) +
        tex2D(game2_tex2D,x, ym1)+
        tex2D(game2_tex2D,xm1, ym1) +
        tex2D(game2_tex2D,xp1, ym1) +
        tex2D(game2_tex2D,xm1,yp1) +
        tex2D(game2_tex2D,xp1,yp1) ;
        
        
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }
    }
}

__global__ void play_game_gpu_texture2D_arrays(int *game_new, int dim, int dir){
    
    
    // there order, either y first or x first, affects speed of the CPU code quite a bit
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    if( dir == 1) {
        // first copy input to output. Then make transitions.
        int s = tex2D(game1_tex2D_array, x,y);
        game_new[y*dim + x] = s;
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex2D(game1_tex2D_array,xm1, y) +
        tex2D(game1_tex2D_array,xp1, y) +
        tex2D(game1_tex2D_array,x,yp1) +
        tex2D(game1_tex2D_array,x, ym1)+
        tex2D(game1_tex2D_array,xm1, ym1) +
        tex2D(game1_tex2D_array,xp1, ym1) +
        tex2D(game1_tex2D_array,xm1,yp1) +
        tex2D(game1_tex2D_array,xp1,yp1) ;
        
        
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }
        
        
        
    }
    else {
        // first copy input to output. Then make transitions.
        int s = tex2D(game2_tex2D_array, x,y);
        game_new[y*dim + x] = s;
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex2D(game2_tex2D,xm1, y) +
        tex2D(game2_tex2D_array,xp1, y) +
        tex2D(game2_tex2D_array,x,yp1) +
        tex2D(game2_tex2D_array,x, ym1)+
        tex2D(game2_tex2D_array,xm1, ym1) +
        tex2D(game2_tex2D_array,xp1, ym1) +
        tex2D(game2_tex2D_array,xm1,yp1) +
        tex2D(game2_tex2D_array,xp1,yp1) ;
        
        
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            game_new[y*dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            game_new[y*dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            game_new[y*dim + x] = 1;
        }
    }
}

__global__ void play_game_gpu_texture2D_surfaces( int dim, int dir){
    
    
    // there order, either y first or x first, affects speed of the CPU code quite a bit
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    if( dir == 1) {
        // first copy input to output. Then make transitions.
        int s = tex2D(game1_tex2D_array, x,y);
        surf2Dwrite(s, game2_surface, 4*x, y, cudaBoundaryModeTrap);
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex2D(game1_tex2D_array,xm1, y) +
        tex2D(game1_tex2D_array,xp1, y) +
        tex2D(game1_tex2D_array,x,yp1) +
        tex2D(game1_tex2D_array,x, ym1)+
        tex2D(game1_tex2D_array,xm1, ym1) +
        tex2D(game1_tex2D_array,xp1, ym1) +
        tex2D(game1_tex2D_array,xm1,yp1) +
        tex2D(game1_tex2D_array,xp1,yp1) ;
        
        
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            surf2Dwrite(0, game2_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            surf2Dwrite(1, game2_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            surf2Dwrite(1, game2_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        
        
        
    }
    else {
        // first copy input to output. Then make transitions.
        int s = tex2D(game2_tex2D_array, x,y);
        surf2Dwrite(s, game1_surface, 4*x, y, cudaBoundaryModeTrap);
        
        int num_neigh_cells;
        
        int xp1 = positive_mod(x+1, dim);
        int xm1 = positive_mod(x-1, dim);
        int yp1 = positive_mod(y+1, dim);
        int ym1 = positive_mod(y-1, dim);
        
        num_neigh_cells = tex2D(game2_tex2D,xm1, y) +
        tex2D(game2_tex2D_array,xp1, y) +
        tex2D(game2_tex2D_array,x,yp1) +
        tex2D(game2_tex2D_array,x, ym1)+
        tex2D(game2_tex2D_array,xm1, ym1) +
        tex2D(game2_tex2D_array,xp1, ym1) +
        tex2D(game2_tex2D_array,xm1,yp1) +
        tex2D(game2_tex2D_array,xp1,yp1) ;
        
        
        
        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (s == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3) ){
            surf2Dwrite(0, game1_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (s == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3) ){
            surf2Dwrite(1, game1_surface, 4*x, y, cudaBoundaryModeTrap);
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (s == 0 && num_neigh_cells == 3){
            surf2Dwrite(1, game1_surface, 4*x, y, cudaBoundaryModeTrap);
        }
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
    
    cudaArray* dev_arr1;
    cudaArray* dev_arr2;
    
    
    cudaMalloc(&dev_game1, sizeof(int)*gamesize);
    cudaMalloc(&dev_game2, sizeof(int)*gamesize);
    
    setrandomconfi(h_game_1, gamedim,0.6);
    //printgame(h_game_1,gamedim);
    cudaMemcpy(dev_game1, h_game_1, sizeof(int)*gamesize, cudaMemcpyHostToDevice);
    
    cudaBindTexture(NULL, game1_tex, dev_game1, gamesize*sizeof(int));
    cudaBindTexture(NULL, game2_tex, dev_game2, gamesize*sizeof(int));
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, game1_tex2D, dev_game1, desc, gamedim, gamedim, sizeof(int)*gamedim);
    cudaBindTexture2D(NULL, game2_tex2D, dev_game2, desc, gamedim, gamedim, sizeof(int)*gamedim);
    
    cudaMallocArray(&dev_arr1, &desc, gamedim, gamedim, cudaArraySurfaceLoadStore);
    cudaMallocArray(&dev_arr2, &desc, gamedim, gamedim, cudaArraySurfaceLoadStore);
    
    cudaBindTextureToArray(game1_tex2D_array, dev_arr1, desc);
    cudaBindTextureToArray(game2_tex2D_array, dev_arr2, desc);
    
    cudaBindSurfaceToArray(game1_surface, dev_arr1);
    cudaBindSurfaceToArray(game2_surface, dev_arr2);
    
    cudaMemcpyToArray(dev_arr1, 0, 0, h_game_1, sizeof(int)*gamesize, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(dev_arr2, 0, 0, h_game_2, sizeof(int)*gamesize, cudaMemcpyHostToDevice);
    
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
        //play_game_gpu_simple<<<num_blocks, threads_per_block>>>(dev_game2, dev_game1, gamedim);
        //play_game_gpu_simple<<<num_blocks, threads_per_block>>>(dev_game1, dev_game2, gamedim);
        
        //play_game_gpu_texture2D<<<num_blocks, threads_per_block>>>(dev_game2, gamedim, 1);
        //play_game_gpu_texture2D<<<num_blocks, threads_per_block>>>(dev_game1, gamedim, 2);
        
        play_game_gpu_texture2D_arrays<<<num_blocks, threads_per_block>>>(dev_game1, gamedim, 1);
        cudaMemcpyToArray(dev_arr1, 0,0, dev_game1, sizeof(int)*gamesize, cudaMemcpyDeviceToDevice);
        
        play_game_gpu_texture2D_arrays<<<num_blocks, threads_per_block>>>(dev_game1, gamedim, 1);
        cudaMemcpyToArray(dev_arr1, 0,0, dev_game1, sizeof(int)*gamesize, cudaMemcpyDeviceToDevice);
        
        //play_game_gpu_texture2D_surfaces<<<num_blocks, threads_per_block>>>(gamedim, 1);
        //play_game_gpu_texture2D_surfaces<<<num_blocks, threads_per_block>>>(gamedim, 2);

        
    }
    gend(&gpu_time);
    
    //cudaMemcpy(h_game_2, dev_game1, sizeof(int)*gamesize, cudaMemcpyDeviceToHost);
    cudaMemcpyFromArray(h_game_2, dev_arr1, 0,0, sizeof(int)*gamesize, cudaMemcpyDeviceToHost);
    
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
