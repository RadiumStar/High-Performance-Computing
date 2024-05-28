#ifndef MATRIX_MULTIPLY_H 
#define MATRIX_MULTIPLY_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

double matrix_multiply(double* a, double* b, double* c, int m, int n, int k); 

#endif

