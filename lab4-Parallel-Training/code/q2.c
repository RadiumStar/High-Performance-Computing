# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include <sys/time.h>
# include <mpi.h>

# define M 500
# define N 500

const double epsilon = 0.001;

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(NULL, NULL); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double* u = (double*)malloc(sizeof(double) * M * N);
    double* w = (double*)malloc(sizeof(double) * M * N);
    int tag = 0; 
    
    if (rank == 0) {
        struct timeval begin_time, end_time; 
        gettimeofday(&begin_time, NULL); 
        double diff, my_diff; 

        int iterations = 0;
        int iterations_print = 1; 
        double mean = 0.0;

        printf("  A program to solve for the steady state temperature distribution\n");
        printf("  over a rectangular plate.\n" );
        printf("\n" );
        printf("  Spatial grid of %d by %d points.\n", M, N);
        printf("  The iteration will be repeated until the change is <= %e\n", epsilon); 

        // initialize 1
        int block = M / size; 
        double local_mean = 0.0;
        ++tag; 
        for (int i = 1; i < size; i++) {
            MPI_Send(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
        }
        // figure out the rest part
        for (int i = (size - 1) * block + (size == 1); i < M - 1; i++) {
            w[i * N + 0] = w[i * N + N - 1] = 100.0; 
            local_mean += w[i * N + 0] + w[i * N + N - 1]; 
        }
        ++tag; 
        for (int i = 1; i < size; i++) {
            MPI_Recv(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        }
        MPI_Reduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        
        // initialize 2
        block = N / size; 
        local_mean = 0.0; 

        ++tag; 
        for (int i = 1; i < size; i++) {
            MPI_Send(w + (i - 1) * block, block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
        }
        // figure out the rest part
        for (int i = (size - 1) * block; i < N; i++) {
            w[i] = 0.0; 
            w[(M - 1) * N + i] = 100.0; 
            local_mean += w[i] + w[(M - 1) * N + i]; 
        }
        ++tag; 
        for (int i = 1; i < size; i++) {
            MPI_Recv(w + (i - 1) * block, block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        }
        MPI_Reduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        

        // initialize 3
        block = M / size; 

        ++tag; 
        for (int i = 1; i < size; i++) {
            MPI_Send(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
            MPI_Send(&mean, 1, MPI_DOUBLE, i, tag + 1, MPI_COMM_WORLD); 
        }
        tag += 2; 
        // figure out the rest part
        for (int i = (size - 1) * block + (size == 1); i < M - 1; i++) {
            for (int j = 0; j < N - 1; j++) {
                w[i * N + j] = mean; 
            }
        }
        for (int i = 1; i < size; i++) {
            MPI_Recv(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        }
        MPI_Barrier(MPI_COMM_WORLD); 
        

        diff = epsilon; 
        while (epsilon <= diff) {
            // save old
            ++tag; 
            for (int i = 1; i < size; i++) {
                MPI_Send(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD); 
            // figure out the rest part
            for (int i = (size - 1) * block; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    u[i * N + j] = w[i * N + j]; 
                }
            }
            ++tag; 
            for (int i = 1; i < size; i++) {
                MPI_Recv(u + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Barrier(MPI_COMM_WORLD); 

            // calculate average w
            ++tag; 
            for (int i = 1; i < size; i++) {
                MPI_Send(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
            }
            // figure out the rest part
            for (int i = (size - 1) * block + (size == 1); i < M - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    w[i * N + j] = (u[(i - 1) * N + j] + u[(i + 1) * N + j] + u[i * N + j - 1] + u[i * N + j + 1] ) / 4.0; 
                }
            }
            ++tag; 
            for (int i = 1; i < size; i++) {
                MPI_Recv(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Barrier(MPI_COMM_WORLD); 

            diff = my_diff = 0.0; 
            // find max diff
            double local_max_diff = 0.0; 
            ++tag; 
            for (int i = 1; i < size; i++) {
                MPI_Send(w + (i - 1) * block * N, N * block, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
            }
            // figure out the rest part
            for (int i = (size - 1) * block + (size == 1); i < M - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    if (local_max_diff < fabs(w[i * N + j] - u[i * N + j])) {
                        local_max_diff = fabs(w[i * N + j] - u[i * N + j]); 
                    }
                }
            }
            MPI_Reduce(&local_max_diff, &my_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (diff < my_diff) diff = my_diff; 

            iterations++; 
            if (iterations == iterations_print) {
                printf("  %8d  %f\n", iterations, diff);
                iterations_print = iterations_print * 2; 
            }
            
            int isloop = (epsilon <= diff); 
            ++tag; 
            for (int i = 1; i < size; i++) {
                MPI_Send(&isloop, 1, MPI_INT, i, tag, MPI_COMM_WORLD); 
            }
            MPI_Barrier(MPI_COMM_WORLD); 
        }

        gettimeofday(&end_time, NULL);
        long seconds = end_time.tv_sec - begin_time.tv_sec;
        long microseconds = end_time.tv_usec - begin_time.tv_usec;
        double elapsed_time = seconds + microseconds;

        printf("\n  %8d  %f\n\n", iterations, diff);
        printf("  Error tolerance achieved.\n");
        printf("  running time = %fs\n", elapsed_time);

        // terminate
        printf("\nHEATED_PLATE_MPI:\n" );
        printf("  Normal end of execution.\n" );  
    }
    else {
        // initialize 1
        int block = M / size; 
        double local_mean = 0.0;
        MPI_Recv(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = (rank == 1) ? 1 : (rank - 1) * block; i < rank * block; i++) {
            w[i * N + 0] = w[i * N + N - 1] = 100.0;
            local_mean += w[i * N + 0] + w[i * N + N - 1]; 
        }
        double global_mean = 0.0; 
        MPI_Send(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD); 
        MPI_Reduce(&local_mean, &global_mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // initialize 2
        block = N / size; 
        local_mean = 0.0; 

        MPI_Recv(w + (rank - 1) * block, block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = (rank - 1) * block; i < rank * block; i++) {
            w[i] = 0.0; 
            w[(M - 1) * N + i] = 100.0; 
            local_mean += w[i] + w[(M - 1) * N + i]; 
        }
        MPI_Send(w + (rank - 1) * block, block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD);
        MPI_Reduce(&local_mean, &global_mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // initialize 3
        block = M / size; 
        double mean = 0.0;
        MPI_Recv(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&mean, 1, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        for (int i = (rank == 1) ? 1 : (rank - 1) * block; i < rank * block; i++) {
            for (int j = 0; j < N - 1; j++) {
                w[i * N + j] = mean; 
            }
        }
        MPI_Send(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD); 
        MPI_Barrier(MPI_COMM_WORLD);

        int isloop = 1; 
        while (isloop) {
            // save old
            MPI_Recv(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = (rank - 1) * block; i < rank * block; i++) {
                for (int j = 0; j < N; j++) {
                    u[i * N + j] = w[i * N + j]; 
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Send(u + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            // calculate average w
            MPI_Recv(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = (rank == 1) ? 1 : (rank - 1) * block; i < rank * block; i++) {
                for (int j = 1; j < N - 1; j++) {
                    w[i * N + j] = (u[(i - 1) * N + j] + u[(i + 1) * N + j] + u[i * N + j - 1] + u[i * N + j + 1] ) / 4.0;
                }
            }
            MPI_Send(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            // find max diff
            double local_max_diff = 0.0; 
            MPI_Recv(w + (rank - 1) * block * N, N * block, MPI_DOUBLE, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = (rank == 1) ? 1 : (rank - 1) * block; i < rank * block; i++) {
                for (int j = 1; j < N - 1; j++) {
                    if (local_max_diff < fabs(w[i * N + j] - u[i * N + j])) {
                        local_max_diff = fabs(w[i * N + j] - u[i * N + j]); 
                    }
                }
            }
        
            double global_max_diff = 0.0; 
            MPI_Reduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            MPI_Recv(&isloop, 1, MPI_INT, 0, ++tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    free(w); 
    free(u); 
    
    MPI_Finalize();  
    
    return 0;
}