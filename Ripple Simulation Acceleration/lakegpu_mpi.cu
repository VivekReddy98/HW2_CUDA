/*
sjoshi26 shashank joshi
akwatra archit kwatra
vkarri vivek reddy karri
*/


#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>


// Variable and constant already defined on the lake_mpi.cu code so defined as extern.

extern int tpdt(double *, double, double);
extern int npebs;
extern int npoints_y;
extern int npoints_x;
extern double end_time;
extern int nthreads;
extern int narea;
extern int numproc, rank;

#define __DEBUG

#ifndef TSCALE
#define TSCALE 1.0
#endif

#ifndef VSQR
#define VSQR 1.0
#endif

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


// GPU - Specific and Rectangular generalization of the evolve13pt function defined in V2, adopted for MPI-CUDA Hybrid.
// 1D Grid and 2D Block Style is used. Threads are alse defined as 2D.
__global__ void evolve13pt_gpu(double *un, double *uc, double *uo, double *pebbles, int n_x, int n_y, double h, double dt, double t){

  int idx_p_1;
  int Neigh;  // North, East, North, South
  int immNeigh; // NorthEast, NorthWest, SouthEast, SouthWest
  int NeighNeigh; // NorthNorth, WestWest, EastEast, SouthSouth

  int i_1;
  int j_1;

  int idx;

  idx_p_1 = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x+threadIdx.x;

   i_1=idx_p_1/n_y;
   j_1=idx_p_1%n_y;

   idx = (j_1+2) + (i_1+2)*(n_y+4);

   if (idx >= (2*n_y + 2) && idx <= (((n_x+1)*n_y) + n_y + 1)){

     Neigh = uc[idx-1] + uc[idx+1] + uc[idx + n_y + 4] + uc[idx - n_y - 4]; // W, E, S, N

     immNeigh = 0.25*(uc[idx - n_y - 5] + uc[idx - n_y - 3] + uc[idx + n_y + 3] + uc[idx + n_y + 5]);  // NW, NE, SW, SE;

     NeighNeigh = 0.125*(uc[idx-2] + uc[idx+2] + uc[idx - 2*(n_y + 4)] - uc[idx + 2*(n_y + 4)]);  // WW, EE, NN, SS;

     un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) * (Neigh + immNeigh + NeighNeigh - 5.5 * uc[idx])/(h * h) + f_pebble(pebbles[idx_p_1],t);
   }

}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
  int BLKS_X, BLKS_Y;

	/* HW2: Define your local variables here CPU Side */

  double t;
  double dt;
  t = 0.;
  dt = h / 2.;

  double *uc, *uo ; // Host Side mem pointers

  //un = (double*)calloc(narea, sizeof(double));
  uc = (double*)calloc(narea, sizeof(double));
  uo = (double*)calloc(narea, sizeof(double));
	//pb = (double*)calloc(npoints_y*npoints_x, sizeof(double));

  /* Set up device timers */
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaEventCreate(&kstart));
  CUDA_CALL(cudaEventCreate(&kstop));

  double *un_cuda,*uc_cuda, *uo_cuda, *pb; // Device Side Memory Access pointers


  /* HW2: Add CUDA kernel call preperation code here */

  BLKS_Y = npoints_y/nthreads;
  BLKS_Y += npoints_y%nthreads ? 1 : 0;

  BLKS_X = npoints_x/nthreads;
  BLKS_X += npoints_x%nthreads ? 1 : 0;

  cudaMalloc((void**)&un_cuda, (narea)*sizeof(double));
  cudaMalloc((void**)&uc_cuda, (narea)*sizeof(double));
  cudaMalloc((void**)&uo_cuda, (narea)*sizeof(double));
  cudaMalloc((void**)&pb, (npoints_x*npoints_y)*sizeof(double));

  cudaMemcpy(uc_cuda, u1, sizeof(double)*narea, cudaMemcpyHostToDevice);
  cudaMemcpy(uo_cuda, u0, sizeof(double)*narea, cudaMemcpyHostToDevice);
  cudaMemcpy(pb, pebbles, sizeof(double)*(npoints_x*npoints_y), cudaMemcpyHostToDevice);


  dim3 block_dim(nthreads, nthreads);
  dim3 grid_dim(BLKS_X, BLKS_Y);


	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* HW2: Add main lake simulation loop here */
  while(1)
  {
    evolve13pt_gpu<<<grid_dim,block_dim>>>(un_cuda, uc_cuda, uo_cuda, pb, npoints_x, npoints_y, h, dt, t);

       cudaMemcpy(uc, un_cuda, sizeof(double)*narea, cudaMemcpyDeviceToHost);
  	   cudaMemcpy(uo, uc_cuda, sizeof(double)*narea, cudaMemcpyDeviceToHost);

       cudaMemcpy(uc_cuda, uc,  sizeof(double)*narea, cudaMemcpyHostToDevice);
       cudaMemcpy(uo_cuda, uo,  sizeof(double)*narea, cudaMemcpyHostToDevice);

      if(!tpdt(&t,dt,end_time)) break;
  }

  cudaMemcpy(u, un_cuda, sizeof(double), cudaMemcpyDeviceToHost);

     /* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
  cudaFree(un_cuda);
  cudaFree(uc_cuda);
  cudaFree(uo_cuda);
  cudaFree(pb);

	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));

  free(uc);
  free(uo);
  //free(pb);
}
