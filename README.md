# Accelerating 2D Ripple Propagation Simulation 
Assigment Completed as a part of CSC 548 Paralell Systems at NC State University.


## Task 1: Indefinite Integral of a 1D sine function using CUDA.
1) Approach is specified in the p2.README file.

## Task 2: Accelerating 2D Ripple Propagation Simulation using Parallel Programming (CUDA & MPI)
1) Converted 5pt Stencil to 13pt Stencil Evolve function to provide faster convergence of the Ripple.
2) Implemented CUDA only variant and tested for various GRID sizes
3) Implemented CUDA-MPI variant and tested for various GRID sizes on four diffent nodes, with different memory address spaces.
4) Performance comparison and the implementational nuances are specified in folder/p3.README.
 
## Task 3: Accelerating 2D Ripple Propagation Simulation using OpenMP and OpenACC
1) Achieved an Acceleration of 50x using OpenACC and around 30x using OpenMP.
2) Optimizations applied and performance results are in OpenMP.README and OpenACC.README

## Task 4: Fault Tolerance in Serial Ripple Simulation Program using Persistent Memory Allocation

Reference: https://dl.acm.org/doi/10.1145/1345206.1345220 (A very good read to understand subtle Performance Optimizations on GPU)

