#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<iostream>

void print_matrix(int m, int n, double* mat) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			printf("%.2f \t", mat[i * n + j]);
		printf("\n");
	}
}

int main(int argc, char* argv[]) {
	double start, stop;
	int m, n, k;
	// read size of input matrix
	// m = 3, n = 2, k = 3;
	
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	k = atoi(argv[3]);
	int rank, size, block;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	block = m / size; // calculate 'block' rows in A in each process
	double* a = new double[m * n]();
	double* b = new double[n * k]();
	double* c = new double[m * k]();
	double* buffer_a = new double[block * n]();	// buffer of matrix A in each process
	double* ans = new double[m * k]();

	if (!rank) {
		srand((unsigned)time(0));
		// generate random matrix A & B
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				a[i * n + j] = (double)rand() / (double)(RAND_MAX) * 100;
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				b[i * k + j] = (double)rand() / (double)(RAND_MAX) * 100;
			}
		}
		// start
		start = MPI_Wtime();
		// scatter matrix A, each process get block lines of A
		MPI_Scatter(a, block * n, MPI_DOUBLE, buffer_a, block * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// broadcast the whole matrix B to each process
		MPI_Bcast(b, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// gather 'ans' to 'c'
		MPI_Gather(ans, block * k, MPI_DOUBLE, c, block * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// calculate the rest of A since m % size may not equal to 0
		
		for (int i = (size - 1) * block; i < m; i++) {
			for (int j = 0; j < k; j++) {
				double temp = 0;
				for (int t = 0; t < n; t++)
					temp += a[i * n + t] * b[t * k + j];
				c[i * k + j] = temp;
			}
		}
		
		stop = MPI_Wtime();

		// print matrix A, B, and result C
		
		
		// printf("matrix A: \n");
		// print_matrix(m, n, a);
		// printf("matrix B: \n");
		// print_matrix(n, k, b);
		// printf("matrix C: \n");
		// print_matrix(m, k, c);
		
		printf("running time: %lf s\n", stop - start);
	}
	// other sub process
	else {
		double* buffer = new double[n * block]();
		MPI_Scatter(a, block * n, MPI_DOUBLE, buffer, block * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(b, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		for (int i = 0; i < block; i++) {
			for (int j = 0; j < k; j++) {
				double temp = 0;
				for (int t = 0; t < n; t++)
					temp += buffer[i * n + t] * b[t * k + j];
				ans[i * k + j] = temp;
			}
		}
		MPI_Gather(ans, block * k, MPI_DOUBLE, c, block * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		delete[] buffer;
	}
	delete[] a;
	MPI_Finalize();
	return 0;
}
