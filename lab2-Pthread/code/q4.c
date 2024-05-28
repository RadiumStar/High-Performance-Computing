#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 10

int NUM_POINTS = 1000000; 

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int total_points_under_curve = 0; // 曲线下方的点的数量
int total_points = 0; // 总采样点数量

void *generate_points(void *arg) {
    long thread_id = (long) arg;
    int points_per_thread = NUM_POINTS / NUM_THREADS;
    int points_under_curve = 0;
    
    for (int i = 0; i < points_per_thread; i++) {
        double x = (double) rand() / RAND_MAX;
        double y = (double) rand() / RAND_MAX;
        
        if (y <= x * x) {
            points_under_curve++;
        }
    }
    
    pthread_mutex_lock(&mutex);
    total_points_under_curve += points_under_curve;
    total_points += points_per_thread;
    pthread_mutex_unlock(&mutex);
    
    pthread_exit(NULL);
}

int main() {
    printf("input number of sample points: "); 
    scanf("%d", &NUM_POINTS); 

    pthread_t threads[NUM_THREADS];
    srand(time(0));
    
    // create thread
    for (long i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, generate_points, (void *)&i);
    }
    
    // join thread
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // result area
    double area = (double)total_points_under_curve / total_points;
    printf("Estimated area: %f\n", area);
    
    return 0;
}
