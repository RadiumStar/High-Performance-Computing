#ifndef MYFOR_H
#define MYFOR_H
#include <pthread.h>

typedef struct {
    int start; 
    int end; 
    int increment; 
    void* (*functor)(void* ); 
    void* arg; 
} ParallelType; 

void* parallel_for(int start, int end, int increment, void* (*functor)(void* ), void* arg, int num_threads); 


#endif

