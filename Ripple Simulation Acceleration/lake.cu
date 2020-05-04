/*
vkarri vivek reddy karri
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);
void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time);
int index13pt(int i, int j, int n); // n is the number of points, i is the row index, j is the column index,

// 13pt generalization for the evaolve function
void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);

// Function in reference to lakegpu.cu file
extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads);

int main(int argc, char *argv[])
{

  if(argc != 5)
  {
    printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
    return 0;
  }

  int     npoints   = atoi(argv[1]);
  int     npebs     = atoi(argv[2]);
  double  end_time  = (double)atof(argv[3]);
  int     nthreads  = atoi(argv[4]);
  int 	  narea	    = (npoints+4) * (npoints+4);

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
  pebs = (double*)calloc(npoints*npoints, sizeof(double));

  u_cpu = (double*)calloc(narea, sizeof(double));
  u_gpu = (double*)calloc(narea, sizeof(double));

  printf("Running %s with (%d x %d) grid, until %f, with %d threads\n", argv[0], npoints, npoints, end_time, nthreads);

  h = (XMAX - XMIN)/npoints;

  init_pebbles(pebs, npebs, npoints);
  init(u_i0, pebs, npoints);
  init(u_i1, pebs, npoints);

  print_heatmap("lake_i.dat", u_i0, npoints, h);

  /* ---------------------------------------------------------------------------------------------*/


  /* -------------------------------------------Executing the code and recording the time ---------------------------------------*/

  gettimeofday(&gpu_start, NULL);
  run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads); // Code which runs GPU Accelarated Version of 13pt stencil evolve function.
  gettimeofday(&gpu_end, NULL);
  elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                  gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
  printf("GPU took %f seconds\n", elapsed_gpu);


  print_heatmap("lake_f.dat", u_gpu, npoints, h);

  /* -------------------------------------------Executing the code and recording the time ---------------------------------------*/


  free(u_i0);
  free(u_i1);
  free(pebs);
  free(u_cpu);
  free(u_gpu);

  return 1;
}

int index13pt(int i, int j, int n) {
    /* Function Which maps indexes from normal i,j of (n_x*n_y) matrix into (n_x+4)*(n_y+4) matrix space.  */
    return (j+2) + (i+2)*(n+4);
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time)
{
  int narea	 = (n+4) * (n+4); // INdexing change for 13 pt stencil;

  double *un, *uc, *uo;
  double t, dt;

  un = (double*)calloc(narea, sizeof(double));
  uc = (double*)calloc(narea, sizeof(double));
  uo = (double*)calloc(narea, sizeof(double));

  memcpy(uo, u0, sizeof(double) * narea);
  memcpy(uc, u1, sizeof(double) * narea);

  t = 0.;
  dt = h / 2.;

  while(1)
  {
    evolve13pt(un, uc, uo, pebbles, n, h, dt, t);

    memcpy(uo, uc, sizeof(double) * narea);
    memcpy(uc, un, sizeof(double) * narea);

    if(!tpdt(&t,dt,end_time)) break;
  }

  memcpy(u, un, sizeof(double) * narea);
}

void init_pebbles(double *p, int pn, int n)
{
  // Function which randomly throws pebles into water.
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
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

void init(double *u, double *pebbles, int n)
{
  // Function Which Intializes values into the normal arrays.
  int i, j, idx_p, idx_u;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx_p = j + i * n;
      idx_u = index13pt(i, j, n); // Modified Indexing because of Padding zeros in the end.
      u[idx_u] = f(pebbles[idx_p], 0.0);
    }
  }
}

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

void evolve13pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  /* No if-else Checks because of padding, and then the function is clean and very straight forward to understand*/
  int i, j, idx_p, idx;

  int Neigh;  // North, East, North, South

  int immNeigh; // NorthEast, NorthWest, SouthEast, SouthWest

  int NeighNeigh; // NorthNorth, WestWest, EastEast, SouthSouth

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
        idx = index13pt(i, j, n);

        idx_p = j + i * n; // Indexing for Pebbles

        Neigh = uc[idx-1] + uc[idx+1] + uc[idx + n + 4] + uc[idx - n - 4]; // W, E, S, N

        immNeigh = 0.25*(uc[idx - n - 5] + uc[idx - n - 3] + uc[idx + n + 3] + uc[idx + n + 5]);  // NW, NE, SW, SE;

        NeighNeigh = 0.125*(uc[idx-2] + uc[idx+2] + uc[idx - 2*(n + 4)] - uc[idx + 2*(n + 4)]);  // WW, EE, NN, SS;

        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) * (Neigh + immNeigh + NeighNeigh - 5.5 * uc[idx])/(h * h) + f(pebbles[idx_p],t);

    }
  }
}

void print_heatmap(const char *filename, double *u, int n, double h)
{
  int i, j, idx;

  FILE *fp = fopen(filename, "w");

  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = index13pt(i, j, n); // Modified Indexing because of Padding zeros in the end.;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}
