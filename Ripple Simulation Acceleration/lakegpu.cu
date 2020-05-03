/*
sjoshi26 shashank joshi
akwatra archit kwatra
vkarri vivek reddy karri
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <iostream>
#include <math.h>

// Variable and constant already defined on the lake.cu code so defined as extern.

using namespace cooperative_groups;

// namespace cg = cooperative_groups;

#define __DEBUG

#ifndef TSCALE
#define TSCALE 1.0
#endif

#ifndef VSQR
#define VSQR 1.0
#endif

extern int tpdt(double *, double, double);


/* -----------------Error Check and time Recording setup for GPU side of execution --------------------------------*/
#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}
/* -----------------Error Check and time Recording setup for GPU side of execution --------------------------------*/


// Device-Specific code to compute f_pebble function.
__device__ double f_pebble(double p, double t)
{
  return -__expf(-TSCALE * t) * p;
}

// GPU - Specific evolve13pt function defined in V2, adopted for GPU accelaration on a single CPU.
// 1D Grid and 2D Block Style is used. Threads are alse defined as 2D.
__global__ void evolve13pt_gpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double t, double end_time, int* numiters, int TotalThreads){

  double dt = h / 2;

  grid_group g = this_grid();

  double *temp_d;

  int idx_p_1;

  int i_1;
  int j_1;
  int idx, blockId;

  while(1) {

    blockId = blockIdx.x + blockIdx.y * gridDim.x;
    idx_p_1 = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    for (; idx_p_1 < n*n ; idx_p_1 += TotalThreads) {

      i_1=idx_p_1/n;
      j_1=idx_p_1%n;
      idx = (j_1+2) + (i_1+2)*(n+4);

      if (idx >= (2*n + 2) && idx <= (((n+1)*n) + n + 1)) {

        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) * (uc[idx-1] + uc[idx+1] + uc[idx + n + 4] + uc[idx - n - 4] +
                                                            0.25*(uc[idx - n - 5] + uc[idx - n - 3] + uc[idx + n + 3] + uc[idx + n + 5])+
                                                            0.125*(uc[idx-2] + uc[idx+2] + uc[idx - 2*(n + 4)] - uc[idx + 2*(n + 4)]) -
                                                            5.5 * uc[idx])/(h * h) + f_pebble(pebbles[idx_p_1],t);
        if (idx_p_1 )
        printf("%f %d\n", un[idx], idx_p_1);
      }
   }

   // Synchronize the entire grid
   g.sync();

    // Check and updte the time, if crosses break.
    if(t + dt > end_time) break;
    else{
      t = t + dt;
      // Pointer Switching optimization instead of copying data to and fro from CPU data to GPU data.
      temp_d = uc;
      uc = un;
      un = uo;
      uo = temp_d;
    }

    g.sync();
  }

}



void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
  int pi=0;
  CUdevice dev;
  cuDeviceGet(&dev,0); // get handle to device 0
  cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev);

  if (pi == 1){
    // printf("Co-operative Launch Property is supported on this GPU\n");
  }
  else{
    printf("Co-operative Launch Property is Not supported on this GPU\n");
    // exit(1);
  }

    cudaEvent_t kstart, kstop;
    float ktime;

    /* HW2: Define your local variables here */
   int narea = (n+4) * (n+4);
   double t;
   double dt;

    t = 0.;
    // dt = h / 2.;

    double *uc, *uo;
    int *numitersHost; // Host Side Data

    numitersHost = (int *)calloc(1, sizeof(int));

    // un = (double*)calloc(narea, sizeof(double));
    uc = (double*)calloc(narea, sizeof(double));
    uo = (double*)calloc(narea, sizeof(double));
    //pb = (double*)calloc(n*n, sizeof(double));

    /* Set up device timers */
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaEventCreate(&kstart));
    CUDA_CALL(cudaEventCreate(&kstop));

    // Device Side data
    double *un_cuda,*uc_cuda,*uo_cuda, *pb;
    int *numiters;

  /* HW2: Add CUDA kernel call preperation code here */

    cudaMalloc((void**)&un_cuda, (narea)*sizeof(double));
    cudaMalloc((void**)&uc_cuda, (narea)*sizeof(double));
    cudaMalloc((void**)&uo_cuda, (narea)*sizeof(double));
    cudaMalloc((void**)&pb, (n*n)*sizeof(double));
    cudaMalloc((void**)&numiters, sizeof(int));

    cudaMemcpy(uc_cuda, u1, sizeof(double)*narea, cudaMemcpyHostToDevice);
    cudaMemcpy(uo_cuda, u0, sizeof(double)*narea, cudaMemcpyHostToDevice);
    cudaMemcpy(pb, pebbles, sizeof(double)*(n*n), cudaMemcpyHostToDevice);
    cudaMemcpy(numiters, numitersHost, sizeof(int), cudaMemcpyHostToDevice);

    /* HW2: Add main lake simulation loop here */

    int numblocksperSM;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numblocksperSM, evolve13pt_gpu, nthreads*nthreads, 0);

    double blocks = sqrt(numblocksperSM);

    int BLKS_x = (int)floor(blocks);
    int BLKS_y = int(numblocksperSM/BLKS_x);

    if (n*n < BLKS_x*BLKS_y*nthreads*nthreads){
       BLKS_x = n / nthreads;
       BLKS_y = BLKS_x + n%nthreads;
    }

    int TotalThreads = BLKS_x*BLKS_y*nthreads*nthreads;

    std::cout << "\nNumber of Blocks Used: " << BLKS_x*BLKS_y << " Possible: " << numblocksperSM << std::endl;
    std::cout << "Number of Total Threads possible for concurrent execution: " << TotalThreads << std::endl;

    dim3 block_dim(nthreads, nthreads, 1);
    dim3 grid_dim(BLKS_x, BLKS_y, 1);

    void *kernelArgs[] = {
      (void *)&un_cuda,  (void *)&uc_cuda, (void *)&uo_cuda, (void *)&pb,
      (void *)&n, (void *)&h, (void *)&t,  (void *)&end_time, (void *)&numiters, (void *)&TotalThreads};

    /* Start GPU computation timer */
    CUDA_CALL(cudaEventRecord(kstart, 0));

    CUDA_CALL(cudaLaunchCooperativeKernel((void *)evolve13pt_gpu, grid_dim, block_dim, kernelArgs, 0, NULL));

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(u, un_cuda, sizeof(double)*narea, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(uc, uc_cuda, sizeof(double)*narea, cudaMemcpyDeviceToHost));
	  CUDA_CALL(cudaMemcpy(uo, uo_cuda, sizeof(double)*narea, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(numitersHost, numiters, sizeof(int), cudaMemcpyDeviceToHost));

    /* Stop GPU computation timer */
    CUDA_CALL(cudaEventRecord(kstop, 0));
    CUDA_CALL(cudaEventSynchronize(kstop));
    CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
    printf("GPU computation: %f msec\n", ktime);

    // std::cout << "\nNumber of Iterations " << *numitersHost << std::endl;


    /* HW2: Add post CUDA kernel call processing and cleanup here */

    // cudaMemcpy(u, un_cuda, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(un_cuda);
    cudaFree(uc_cuda);
    cudaFree(uo_cuda);
    cudaFree(pb);
    cudaFree(numiters);

    free(uc);
    free(uo);
    free(numitersHost);

    /* timer cleanup */
    CUDA_CALL(cudaEventDestroy(kstart));
    CUDA_CALL(cudaEventDestroy(kstop));
}
