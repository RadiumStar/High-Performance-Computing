/*
 * gcc test.c -o test -I. -L. -Wl,-rpath=. -libmyfor
 */

#include "myfor.h"
#include <stdio.h>
#include <sys/time.h>

#define NUM_THREADS 2
#define SIZE 4

int A[SIZE][SIZE];
int B[SIZE][SIZE];
int C[SIZE][SIZE];

void* matrix_multiply(void* args) {
    ParallelType* p = (ParallelType*)args; 
    int start = p->start; 
    int end = p->end; 
    int increment = p->increment;

    for (int i = start; i < end; i+= increment) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0; 
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[j][k];
            }
        }
    }

    return NULL; 
}

int main() {
    // initialize matrix A & B
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = B[i][j] = i + j;
        }
    }

    ParallelType args; 
    struct timeval begin_time, end_time; 
    gettimeofday(&begin_time, NULL); 

    parallel_for(0, SIZE, 1, matrix_multiply, NULL, NUM_THREADS); 

    gettimeofday(&end_time, NULL);
    long seconds = end_time.tv_sec - begin_time.tv_sec;
    long microseconds = end_time.tv_usec - begin_time.tv_usec;
    double elapsed_time = seconds + microseconds / 1e6;
    printf("Matrix Size: %d; Threads: %d; Running time: %.4f ms\n", SIZE, NUM_THREADS, elapsed_time);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d ", C[i][j]); 
        }
        printf("\n"); 
    }
}