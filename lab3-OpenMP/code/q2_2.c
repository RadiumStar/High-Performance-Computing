#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define NUM_THREADS 8
#define SIZE 2048

int A[SIZE][SIZE];
int B[SIZE][SIZE];
int C[SIZE][SIZE];

double matrix_multiply(int size, int num_threads) {
    struct timeval begin_time, end_time; 
    gettimeofday(&begin_time, NULL); 

    #pragma omp parallel num_threads(num_threads) 
    {
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                C[i][j] = 0;
                for (int k = 0; k < size; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    gettimeofday(&end_time, NULL);
    long seconds = end_time.tv_sec - begin_time.tv_sec;
    long microseconds = end_time.tv_usec - begin_time.tv_usec;
    double elapsed_time = seconds + microseconds / 1e6;
    return elapsed_time * 1000; 
}

int main() {
    // initialize matrix A & B
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = B[i][j] = i + j;
        }
    }

    for (int i = 512; i <= SIZE; i <<= 1) {
        for (int j = 1; j <= NUM_THREADS; j++) {
            double running_time = matrix_multiply(i, j); 
            printf("Matrix Size: %d; Threads: %d; Running time: %.4f ms\n", i, j, running_time); 
        }
    }
    return 0;
}
