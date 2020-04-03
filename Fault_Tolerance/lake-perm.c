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
#include "jemalloc/jemalloc.h"

#define _USE_MATH_DEFINES

// Some Required Constants
#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

// Backup Files to Store Persistent Variables
#define BACK_FILE "/tmp/vkarri.app.back" //COMMENT: remove this comment after
#define MMAP_FILE "/tmp/vkarri.app.mmap" //COMMENT: entering you unity-id
#define MMAP_SIZE ((size_t)1 << 30)

/*
Data Structures for Persistent Storage
Declared Global Because, persistent storage Saves Heap and only Global Variables
         but not Stack Allocated Variables
*/
PERM double *pebs;
PERM double *uo;
PERM double *uc;
PERM double t;
PERM int initialization_flag = 0;

// Useful Functions
void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);
double total(double *u, int n);
void run_cpu(double *u, int n, double h, double end_time);


// Global Variable to know if the execution is a restart.
int do_restore;

int main(int argc, char *argv[])
{
  // To check if the start/re-start is a restore
  do_restore = argc > 1 && strcmp("-r", argv[1]) == 0;
  const char *mode = (do_restore) ? "r+" : "w+";

  // Persistent memory initialization/backup location
  perm(PERM_START, PERM_SIZE);
  mopen(MMAP_FILE, mode, MMAP_SIZE);
  bopen(BACK_FILE, mode);

  // Setting up the Variables
  int     npoints   = 128; //atoi(argv[1]);
  int     npebs     = 8; //atoi(argv[2]);  //
  double  end_time  = 1.0; //(double)atof(argv[3]);
  int 	  narea	    = (npoints+4) * (npoints+4);

  // u_cpu is a temp variable and is filled only after the execution of run_cpu functions and hence not svaed to persistent storage
  double *u_cpu;

  // Value of h computed
  double h = (XMAX - XMIN)/npoints;

  // Init persistent variables
  // initialization_flag is an guarenteed indication that, uo, uc and other initializations have been made
  if (do_restore && initialization_flag) {
    printf("restarting...\n");

    // restore the heap and saved global variables
    restore();
  }
  else{

    // Allocate pebs, uo, uc
    printf("Running %s with (%d x %d) grid, until %f\n", argv[0], npoints, npoints, end_time);
    pebs = (double*)malloc(sizeof(double) * narea);
    uo = (double*)malloc(sizeof(double) * narea);
    uc = (double*)malloc(sizeof(double) * narea);

    // Initialize pebs, uo, uc
    init_pebbles(pebs, npebs, npoints);
    init(uo, pebs, npoints);
    init(uc, pebs, npoints);

    // Set the initial time
    t = 0.;

    print_heatmap("lake_i.dat", uo, npoints, h);

    initialization_flag = 1;

    // If initialization_flag is set to one, then that is an indication that the initial states have been saved
    mflush(); /* a flush is needed to save some global state */
    backup();
  }

  // Allocate u_cpu i.e. final answer
  u_cpu = (double*)malloc(sizeof(double) * narea);

  // double elapsed_cpu;
  // struct timeval cpu_start, cpu_end;

  // gettimeofday(&cpu_start, NULL);
  run_cpu(u_cpu, npoints, h, end_time);
  // gettimeofday(&cpu_end, NULL);

  // elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
  // printf("CPU took %f seconds\n", elapsed_cpu);

  print_heatmap("lake_f.dat", u_cpu, npoints, h);

  // free all the heap allocated variables
  free(uo);
  free(uc);
  free(pebs);
  free(u_cpu);

  // Cleanup
  mclose();
  bclose();
  remove(BACK_FILE);
  remove(MMAP_FILE);

  return 1;
}

void run_cpu(double *un, int n, double h, double end_time)
{

  // Compute dt
  double dt = h / 2.;

  while(1)
  {
    // print time to visualize Backup
    printf("Time is: %f \n", t);

    // Evolve function compute a single updation
    evolve(un, uc, uo, pebs, n, h, dt, t);

    // Break if the time has reached the limit
    if(!tpdt(&t,dt,end_time)) break;

    // Copy the varibles into necessary locations
    memcpy(uo, uc, sizeof(double) * n * n);
    memcpy(uc, un, sizeof(double) * n * n);

    // Backup the updated PERM Variables and heap and time
    // If the curent iteration is not saved, the restart will start at the same iteration
    backup();
  }

}

void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  // srand(10) to enforce consistency
  srand(10);
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
  int i, j, idx;

  for(i = 0; i < n ; i++)
  {
    for(j = 0; j < n ; j++)
    {
      idx = j + i * n;
      u[idx] = f(pebbles[idx], 0.0);
    }
  }
}

// Nothing has been changed in the evolve function
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  // Loop over all the points
  for( i = 0; i < n; i++)
    {
     for( j = 0; j < n; j++)
        {
          idx = j + i * n;

          /* imposing the boundary conditions */
          if(i < 2 || i > n - 3 || j < 2 || j > n - 3)
          {
                un[idx] = 0.;
          }

          // Else Compute the un[idx]
          else
          {
          un[idx] = 2 * uc[idx] - uo[idx] + VSQR * (dt * dt) * ((uc[idx - 1] + uc[idx + 1] + uc[idx + n] + uc[idx - n] +
                                                                0.25 * (uc[idx+n- 1] + uc[idx+n+1] + uc[idx-n-1] + uc[idx-n+1]) +
                                                                0.125 * (uc[idx-2] + uc[idx + 2] + uc[idx-2*n] + uc[idx+2*n]) -
                                                                5.5 * uc[idx])*(1/(h * h)) + f(pebbles[idx], t));

          }
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
      idx = j + i * n;
      fprintf(fp, "%f %f %f\n", i*h, j*h, u[idx]);
    }
  }

  fclose(fp);
}

// double total(double *u, int n){
//   double sum = 0.;
//   int i, j, idx;
//
//   for(i = 0; i < n ; i++)
//   {
//     for(j = 0; j < n ; j++)
//     {
//       idx = j + i*n;
//       sum += u[idx];
//     }
//   }
//
//   return sum;
// }
