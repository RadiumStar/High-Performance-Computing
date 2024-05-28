#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>


#define SIZE 512
int NUM_THREADS = 8; 

int A[SIZE][SIZE];
int B[SIZE][SIZE];
int C[SIZE][SIZE];

// struct of each part of matrix data including start row and end row
struct thread_data {
    int thread_id;
    int start_row;
    int end_row;
};

void print_matrix(int mat[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void* matrix_multiply(void *threadarg) {
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
    int start_row = my_data->start_row;
    int end_row = my_data->end_row;
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j]; //
            }
        }
    }
    
    pthread_exit(NULL);
}

int main() {
    printf("input number of threads: "); 
    scanf("%d", &NUM_THREADS); 

    struct timeval begin_time, end_time; 
    // create NUM_THREADS of threads
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data_array[NUM_THREADS];
    // pay attention: the SIZE % NUM_THREADS should be 0
    int rows_per_thread = SIZE / NUM_THREADS;
    
    // initialize matrix A & B
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = B[i][j] = i + j;
        }
    }
    
    // begin the matrix multiply
    gettimeofday(&begin_time, NULL);
    
    // create threads, we divide matrix A into 'rows_per_thread'
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data_array[i].thread_id = i;
        thread_data_array[i].start_row = i * rows_per_thread;
        // deal with the situation where SIZE % NUM_THREADS != 0
        if (i == NUM_THREADS - 1) {
            thread_data_array[i].end_row = SIZE;
        }
        else {
            thread_data_array[i].end_row = (i + 1) * rows_per_thread;
        }
        // create thread to calculate the part of the matrix multiplication
        pthread_create(&threads[i], NULL, matrix_multiply, (void *)&thread_data_array[i]);
    }
    
    
    // join all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end_time, NULL);
    
    
    // printf("Matrix A: \n"); 
    // print_matrix(A);
    // printf("Matrix B: \n");
    // print_matrix(B);
    // printf("Matrix C: \n");
    // print_matrix(C);
    
    // 打印运算时间
    long seconds = end_time.tv_sec - begin_time.tv_sec;
    long microseconds = end_time.tv_usec - begin_time.tv_usec;
    double elapsed_time = seconds + microseconds / 1e6;
    printf("%lf ms elapsed\n", elapsed_time * 1000); 
    return 0;
}
