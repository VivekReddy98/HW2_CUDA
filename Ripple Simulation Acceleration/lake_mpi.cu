/*
sjoshi26 shashank joshi
akwatra archit kwatra
vkarri vivek reddy karri
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

// Variables Made global and such function argument signature have also been hanged.
int npebs;
int npoints_y;  // Because of the split for MPI, Dimensions for npoints in x and y dimension differ and thus dealing with rectangular matrices.
int npoints_x;
double end_time;
int nthreads;
int narea;
int  numproc, rank;

void init(double *u, double *pebbles);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, double h);
void init_pebbles(double *p);
void run_cpu(double *u, double *u0, double *u1, double *pebbles, double h, double end_time);
int index13pt(int i, int j, int n); // n is the number of points, i is the row index, j is the column index,
void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

// Function in reference to lakegpu_mpi.cu file
extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, double h, double end_time, int nthreads);

// MPI Related Funtion For Boundary Communication.
void getBoundaryOnce(double *);

// Allocating Number of Pebbles and the points According to the node.
void scheduleResources(int argc, char *argv[]);

int main(int argc, char *argv[])
{
/* process information */
  int len;

  /* current process hostname */
  char hostname[MPI_MAX_PROCESSOR_NAME];

  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* get the number of procs in the comm */
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  /* get my rank in the comm */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* get some information about the host I'm running on */
  MPI_Get_processor_name(hostname, &len);


  if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    exit(0);
  }

  printf("%d, %d", rank, numproc);
  fflush(stdout);

  scheduleResources(argc, argv);
  fflush(stdout);

  /* ---------------------Allocating Required Dynamic Arrays -----------------------------------------------*/
  // As it can be seen the space allocated for pebs matrix and other matrices are different, because, to avoid ugly if-else statements in the
  //    13pt stencil code, we have decided to padd 2-columns and 2-rows of zeros on top and bottom of the matrices and hence the difference for the size.
  double *u_i0, *u_i1;
  double *u_cpu, *u_gpu, *pebs;
  double h;

  double elapsed_cpu, elapsed_gpu;
  struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

  u_i0 = (double*)calloc(narea, sizeof(double));
  u_i1 = (double*)calloc(narea, sizeof(double));
  pebs = (double*)calloc(npoints_x*npoints_y, sizeof(double));

  u_cpu = (double*)calloc(narea, sizeof(double));
  u_gpu = (double*)calloc(narea, sizeof(double));

  printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints_x, npoints_y, end_time, nthreads);

  h = (XMAX - XMIN)/npoints_y;

  init_pebbles(pebs);  // Matrix initialization is handled in these functions.
  init(u_i0, pebs);
  init(u_i1, pebs);

  /* ---------------------------------------------------------------------------------------------*/


  /* -------------------------------------------Executing the code and recording the time ---------------------------------------*/
  char filename[13] = "lake_i_n.dat";

  sprintf(&filename[7], "%d", rank);
  //filename[7] = *(itoa(rank));

  print_heatmap(filename, u_i0, h);

  gettimeofday(&cpu_start, NULL);
  run_cpu(u_cpu, u_i0, u_i1, pebs, h, end_time);   // Code which runs CPU Version of MPI.
  gettimeofday(&cpu_end, NULL);

  elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                  cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  printf("CPU took %f seconds\n", elapsed_cpu);

  gettimeofday(&gpu_start, NULL);
  run_gpu(u_gpu, u_i0, u_i1, pebs, h, end_time, nthreads); // Code which runs GPU Accelerated Version of MPI. (MPI-CUDA HYBRID), Function is found in lakegpu_mpi.cu
  gettimeofday(&gpu_end, NULL);
  elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
  printf("GPU took %f seconds\n", elapsed_gpu);

  char filename2[13] = "lake_f_n.dat";

  sprintf(&filename2[7], "%d", rank);

  print_heatmap(filename2, u_cpu, h);

  /* -------------------------------------------Executing the code and recording the time ---------------------------------------*/

  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_cpu);
  free(u_gpu);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}

void scheduleResources(int argc, char **argv){

      int npebs_total = atoi(argv[2]);

      npoints_y = atoi(argv[1]);
      end_time  = (double)atof(argv[3]);
      nthreads  = atoi(argv[4]);

      npoints_x = npoints_y/numproc;
      npebs     = npebs_total/numproc;

      if (rank == numproc-1) {
          npoints_x += npoints_y%numproc;
          npebs += npebs_total%numproc;
      }

      narea	 = (npoints_x+4) * (npoints_y+4);

      printf("Rank: %d, numProc: %d, n_x: %d, n_y: %d", rank, numproc, npoints_x, npoints_y);
}

int index13pt(int i, int j, int n) {
    /* Function Which maps indexes from normal i,j of (n_x*n_y) matrix into (n_x+4)*(n_y+4) matrix space.  */
    return (j+2) + (i+2)*(n+4);
}

void init_pebbles(double *p)
{
  // Function which randomly throws pebles into water.
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  //memset(p, 0, sizeof(double) * npoints_x * npoints_y);

  for( k = 0; k < npebs ; k++)
  {
    i = rand() % (npoints_x - 4) + 2;
    j = rand() % (npoints_y - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * npoints_y;
    p[idx] = (double) sz;
  }
}

double f(double p, double t)
{
  return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

void init(double *u, double *pebbles)
{
  // Function Which Intializes values into the normal arrays.
  int i, j, idx_p, idx_u;

  for(i = 0; i < npoints_x ; i++)
  {
    for(j = 0; j < npoints_y ; j++)
    {
      idx_p = j + i * npoints_y;
      idx_u = index13pt(i, j, npoints_y); // Modified Indexing because of Padding zeros in the end.
      u[idx_u] = f(pebbles[idx_p], 0.0);
    }
  }
}

void evolve13pt(double *un, double *uc, double *uo, double *pebbles, double h, double dt, double t)
{

  /* No if-else Checks because of padding, and then the function is very straight forward to understand*/

  int i, j, idx_p, idx;

  int Neigh;  // North, East, North, South

  int immNeigh; // NorthEast, NorthWest, SouthEast, SouthWest

  int NeighNeigh; // NorthNorth, WestWest, EastEast, SouthSouth

  for( i = 0; i < npoints_x; i++)
  {
    for( j = 0; j < npoints_y; j++)
    {
        idx = index13pt(i, j, npoints_y);

        idx_p = j + i * npoints_y; // Indexing for Pebbles

        Neigh = uc[idx-1] + uc[idx+1] + uc[idx + npoints_y + 4] + uc[idx - npoints_y - 4]; // W, E, S, N

        immNeigh = 0.25*(uc[idx - npoints_y - 5] + uc[idx - npoints_y - 3] + uc[idx + npoints_y + 3] + uc[idx + npoints_y + 5]);  // NW, NE, SW, SE;

        NeighNeigh = 0.125*(uc[idx-2] + uc[idx+2] + uc[idx - 2*(npoints_y + 4)] - uc[idx + 2*(npoints_y + 4)]);  // WW, EE, NN, SS;

        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) * (Neigh + immNeigh + NeighNeigh - 5.5 * uc[idx])/(h * h) + f(pebbles[idx_p],t);

    }
  }
}

void print_heatmap(const char *filename, double *u, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < npoints_x; i++ )
  {
    for( j = 0; j < npoints_y; j++ )
    {
      idx = index13pt(i, j, npoints_y); // Modified Indexing because of Padding zeros in the end.;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, double h, double end_time)
{
  //int narea	 = (n+4) * (n+4); // INdexing change for 13 pt stencil;

  double *un, *uc, *uo;
  double t, dt;

  un = (double*)calloc(narea, sizeof(double));
  uc = (double*)calloc(narea, sizeof(double));
  uo = (double*)calloc(narea, sizeof(double));

  memcpy(uo, u0, sizeof(double) * narea);
  memcpy(uc, u1, sizeof(double) * narea);

  // Every Node computes their share of u1 and uo and communicates them to the neighbouring nodes.

  getBoundaryOnce(uo);

  getBoundaryOnce(uc);

  t = 0.;
  dt = h / 2.;

  while(1)
  {
    evolve13pt(un, uc, uo, pebbles, h, dt, t); // 13pt evolve funtion for their share of nodes.

    getBoundaryOnce(un); // Computes un for one time-step and updates the boundary to-and from neoighbouring nodes.

    memcpy(uo, uc, sizeof(double) * narea);
    memcpy(uc, un, sizeof(double) * narea);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * narea); // After processing exceeds the given time, copy the matrix into the final staging matrix.
}

//
void getBoundaryOnce(double *u){

  // Non-Blocking Primitives in combination with waits have been used to implement the boundary communication

  // Note: It is a better practice to recieve first and then send in non-blocking way of MPI.

  if (numproc == 1) return ;
  //Sending and Recieving the last two arrays
   MPI_Status stSendT, stRcvT, stSendB, stRcvB;
   MPI_Request sendRqstT, rcvRqstT, sendRqstB, rcvRqstB;

   // Careful indexing because this a matrix.
   int SendTop =  2*(npoints_y+4);

   int SendBottom = (npoints_x)*(npoints_y+4);

   int RcvBottom = (npoints_x+2)*(npoints_y+4);

   int numElem = 2*(npoints_y+4);


   // Last node will only have to communicate the upper row values
   if(rank == numproc-1){
       MPI_Irecv(u, numElem, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &rcvRqstT);
       MPI_Isend(u+SendTop, numElem, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &sendRqstT);

       MPI_Wait(&sendRqstT, &stSendT);
       MPI_Wait(&rcvRqstT, &stRcvT);
   }

   // First node will only have to communicate the lower row values
   else if (rank == 0) {
       MPI_Irecv(u+RcvBottom, numElem, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &rcvRqstB);
       MPI_Isend(u+SendBottom, numElem, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &sendRqstB);

       MPI_Wait(&sendRqstB, &stSendB);
       MPI_Wait(&rcvRqstB, &stRcvB);
   }

   // All the other nodes have to communicate upper and lower row values.
   else {
       MPI_Irecv(u, numElem, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &rcvRqstT);
       MPI_Irecv(u+RcvBottom, numElem, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &rcvRqstB);

       MPI_Isend(u+SendTop, numElem, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &sendRqstT);
       MPI_Isend(u+SendBottom, numElem, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &sendRqstB);

       MPI_Wait(&sendRqstT, &stSendT);
       MPI_Wait(&rcvRqstT, &stRcvT);

       MPI_Wait(&sendRqstB, &stSendB);
       MPI_Wait(&rcvRqstB, &stRcvB);
   }

   //to ensure everything is sent and received accordingly
   MPI_Barrier(MPI_COMM_WORLD);
}


/* Already Provided 5pt stencil function */
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] +
                  uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}
