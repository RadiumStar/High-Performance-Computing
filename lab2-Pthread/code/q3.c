#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define NUM_THREADS 3

int a = 1;
int b = -3;
int c = 2;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_var = PTHREAD_COND_INITIALIZER;
int threads_completed = 0;

void* calculate_b2(void* threadarg) {
    int* data = (int*)threadarg;
    *data = b * b;
    pthread_mutex_lock(&mutex);
    threads_completed++;
    pthread_cond_signal(&cond_var);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void* calculate_4ac(void* threadarg) {
    int* data = (int*)threadarg;
    *data = 4 * a * c;
    pthread_mutex_lock(&mutex);
    threads_completed++;
    pthread_cond_signal(&cond_var);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void* calculate_2a(void* threadarg) {
    int* data = (int*)threadarg;
    *data = 2 * a;
    pthread_mutex_lock(&mutex);
    threads_completed++;
    pthread_cond_signal(&cond_var);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main() {
    clock_t begin_time, end_time;
    pthread_t threads[NUM_THREADS];
    int thread_datas[NUM_THREADS];

    printf("input a, b, c(int): ");
    scanf("%d%d%d", &a, &b, &c);

    begin_time = clock();
    // create threads
    pthread_create(&threads[0], NULL, calculate_b2, (void*)&thread_datas[0]);
    pthread_create(&threads[1], NULL, calculate_4ac, (void*)&thread_datas[1]);
    pthread_create(&threads[2], NULL, calculate_2a, (void*)&thread_datas[2]);

    // wait for all threads to complete
    pthread_mutex_lock(&mutex);
    while (threads_completed < NUM_THREADS) {
        pthread_cond_wait(&cond_var, &mutex);
    }
    pthread_mutex_unlock(&mutex);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    double x1 = (-b + sqrt((double)(thread_datas[0] - thread_datas[1]))) / thread_datas[2];
    double x2 = (-b - sqrt((double)(thread_datas[0] - thread_datas[1]))) / thread_datas[2];

    end_time = clock();

    printf("result of functions: x1 = %.2f; x2 = %.2f\n", x1, x2);

    double running_time = (double)(end_time - begin_time) / CLOCKS_PER_SEC;
    printf("running time: %f ms\n", running_time * 1000);
    return 0;
}
