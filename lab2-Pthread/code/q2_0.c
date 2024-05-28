#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define SIZE 1000

int a[SIZE]; 
int global_index = 0; 
int global_sum = 0;

int main() {
    clock_t begin_time, end_time; 

    // Initialize array
    for (int i = 0; i < SIZE; i++) {
        a[i] = i; 
    }

    begin_time = clock();
    
    for (int i = 0; i < SIZE; i++) {
        global_sum += a[i];
    }

    end_time = clock();

    printf("sum of array = %d\n", global_sum); 

    double running_time = (double)(end_time - begin_time) / CLOCKS_PER_SEC;
    printf("running time: %f ms\n", running_time * 1000);
    
    return 0; 
}