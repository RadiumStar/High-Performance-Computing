#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define SIZE 1000
#define NUM_THREADS 8
#define NUM_ELEMENTS_PER_THREAD 10

int a[SIZE]; 
int global_index = 0; 
int global_sum = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; 

void* sum(void* threadarg) {
    int local_sum = 0; 
    int begin_index, end_index; 

    while (1) {
        // need to lock to prevent visit global variable at the same time
        pthread_mutex_lock(&mutex); 
        // if index out of array bound, then quit
        if (global_index >= SIZE) {
            pthread_mutex_unlock(&mutex); 
            break; 
        }
        begin_index = global_index; 
        end_index = begin_index + NUM_ELEMENTS_PER_THREAD;
        global_index += NUM_ELEMENTS_PER_THREAD; 
        pthread_mutex_unlock(&mutex); 

        if (end_index > SIZE) end_index = SIZE; 

        for (int i = begin_index; i < end_index; i++) {
            local_sum += a[i];
        }
    }

    pthread_mutex_lock(&mutex); 
    global_sum += local_sum; 
    pthread_mutex_unlock(&mutex); 

    pthread_exit(NULL); 
}

int main() {
    struct timeval begin_time, end_time; 
    pthread_t threads[NUM_THREADS]; 

    // Initialize array
    for (int i = 0; i < SIZE; i++) {
        a[i] = i; 
    }

    gettimeofday(&begin_time, NULL);
    // create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, sum, NULL); 
    }

    // join threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL); 
    }
    gettimeofday(&end_time, NULL);

    printf("sum of array = %d\n", global_sum); 

    long seconds = end_time.tv_sec - begin_time.tv_sec;
    long microseconds = end_time.tv_usec - begin_time.tv_usec;
    double elapsed_time = seconds + microseconds / 1e6;
    printf("%lf ms elapsed\n", elapsed_time * 1000); 

    return 0; 
}