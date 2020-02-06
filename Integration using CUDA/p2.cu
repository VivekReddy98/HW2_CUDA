/* vkarri Vivek Reddy Karri */

#include <stdio.h>
#include <cuda.h>
#include <math.h>

/* first grid point */
#define   XI              0.0
/* last grid point */
#define   XF              M_PI
/* Num Threads per block, Can at max be 512 */
#define   N_THRS          512

// Custom DS for passing value into the cuda function.
typedef struct DS {
    //"real" grid indices
    double I;
    double F;
    int limit;
    double h;
} ds;

// Print Function to plot the values
void print_function_data(int np, double *x, double *y, double *dydx);

// Device Function to Calculate x, y and area of one rectangular block (Not cummulative)
__global__ void integratePartial(double *x, double *y, double *inf, ds* custom_ds)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i == 0) inf[0] = 0.0;

  double xi = 0.0, xi_1 = 0.0, yi = 0.0, yi_1 = 0.0;

  if (i > 0 && i <= custom_ds->limit) {

      xi = custom_ds->I + (custom_ds->F - custom_ds->I) * (double)(i - 1)/(double)(custom_ds->limit - 1);
      yi = sin(xi);

      xi_1 = custom_ds->I + (custom_ds->F - custom_ds->I) * (double)(i - 2)/(double)(custom_ds->limit - 1);
      yi_1 = sin(xi_1);

      x[i] = xi;
      y[i] = yi;
      inf[i] = ((yi + yi_1) * custom_ds->h / 2);
  }
}

int main(int argc, char** argv){

  int NGRID;
  int BLKS;

  if(argc > 1) {
      NGRID = atoi(argv[1]);//input for no of grids
  }
  else
  {
      printf("Please specify the expected number of arguments which is two\n");
      exit(0);
  }

  // Set the values for number of BLKS.
  BLKS = NGRID/N_THRS;
  BLKS += NGRID%N_THRS ? 1 : 0;

  printf("Number of Blocks: %d, Grid_Points: %d\n", BLKS, NGRID);

  // Custom DS to store the values of constants required to compute various values.
  ds custom;

  custom.I = XI;
  custom.F = XF;
  custom.limit = NGRID;
  custom.h = (XF - XI) / (NGRID - 1);

  // Device Arrays
  double *d_x, *d_y, *d_inf;
  ds *d_custom;

  // Allocate Host Memory and store a reference of those pointers.
  cudaMalloc(&d_custom, sizeof(ds));
  cudaMalloc(&d_x, (NGRID+1)*sizeof(double));
  cudaMalloc(&d_y, (NGRID+1)*sizeof(double));
  cudaMalloc(&d_inf, (NGRID+1)*sizeof(double));

  cudaMemcpy(d_custom, &custom, sizeof(ds), cudaMemcpyHostToDevice);

  // Host Arrays
  double *x, *y, *inf;

  // Allocate arrays for host memory
  x = (double*)calloc(NGRID+1, sizeof(double));
  y = (double*)calloc(NGRID+1, sizeof(double));
  inf = (double *)calloc(NGRID+1, sizeof(double));

  integratePartial<<<BLKS, N_THRS>>>(d_x, d_y, d_inf, d_custom); // Invoke the kernel to find the area

  // Copy the Computed Values into Host Memory
  cudaMemcpy(y, d_y, (NGRID+1)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x, (NGRID+1)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(inf, d_inf, (NGRID+1)*sizeof(double), cudaMemcpyDeviceToHost);
	
  // Find the cummulative addition to find Indefinite Integral
  int i = 1;
  for(i = 1 ; i <= NGRID; ++i){
      inf[i] += inf[i-1];
  }

  print_function_data(NGRID, x, y, inf);
  
  // Free Local Host Memory 
  free(x);
  free(y);
  free(inf);

  // Free Device Memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_inf);
  cudaFree(d_custom);
}

void print_function_data(int np, double *x, double *y, double *dydx)
{
        int   i;

        char filename[1024];
        sprintf(filename, "fn-%d.dat", np);

        FILE *fp = fopen(filename, "w");

        //int cummulativeArea = dydx[0];

        for(i = 1; i < np+1; i++)
        {
                //printf("%f %f %f\n", x[i], y[i], dydx[i]);
                //cummulativeArea += dydx[i];
                fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
        }

        fclose(fp);
}
