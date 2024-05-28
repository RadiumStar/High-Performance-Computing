#include "myfor.h"

void* parallel_for(int start, int end, int increment, void* (*functor)(void* ), void* arg, int num_threads) {
    pthread_t threads[num_threads]; 
    ParallelType cells[num_threads]; 
    int size_per_thread = (end - start) / num_threads; 
    for (int i = 0; i < num_threads; i++) {
        cells[i].start = start + i * size_per_thread; 
        cells[i].end = i == (num_threads - 1) ? end : (start + (i + 1) * size_per_thread); 
        cells[i].increment = increment; 
        cells[i].functor = functor; 
        cells[i].arg = arg;
        
        pthread_create(&threads[i], NULL, functor, (void*)&cells[i]); 
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL); 
    }

    return NULL; 
}