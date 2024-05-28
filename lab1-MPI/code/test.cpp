#include "matrix_multiply.h"

int main(int argc, char* argv[]) {
	printf("input the size of matrix(m n k, 512 - 2048)"); 
	int m = atoi(argv[1]); 
	int n = atoi(argv[2]); 
	int k = atoi(argv[3]);  
	// scanf("%d%d%d", &m, &n, &k); 
	double* a = new double[m * n];
	double* b = new double[n * k]; 
	double* c = new double[m * k]; 

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

	double running_time = matrix_multiply(a, b, c, m, n, k); 

	printf("running time: %lf s\n", running_time);
	return 0;
}
