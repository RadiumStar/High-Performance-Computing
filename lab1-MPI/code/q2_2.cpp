#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>

void print_matrix(int m, int n, double* mat) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%.2f \t", mat[i * n + j]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int m, n, k;

    // read size of input matrix
    // m = 3, n = 2, k = 3;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    int rank, size, block;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    block = m / size;   // calculate 'block' rows in A in each thread
    double* b = new double[n * k];
    double* ans = new double[m * k];   // store answer in each thread

    // Create custom MPI datatype for sending matrix A
    MPI_Datatype blockType;
    MPI_Type_vector(block, n, n * size, MPI_DOUBLE, &blockType);
    MPI_Type_commit(&blockType);

    if (!rank) {    // main thread 
        double* a = new double[m * n];
        double* c = new double[m * k];

        srand((unsigned)time(0));
        // generate random matrix A & B
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = (double)rand() / (double)(RAND_MAX) * 100;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                b[i * k + j] = (double)rand() / (double)(RAND_MAX) * 100;
            }
        }

        // start
        double start = MPI_Wtime();


        // send the whole matrix B
        for (int i = 1; i < size; i++) {
            MPI_Send(b, n * k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        // send the matrix A in 'block'
        for (int i = 1; i < size; i++) {
            MPI_Send(a + (i - 1) * block * n, 1, blockType, i, 1, MPI_COMM_WORLD);
        }

        // calculate the rest of A since m % size may not equal to 0
        for (int i = (size - 1) * block; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double temp = 0;
                for (int t = 0; t < n; t++)
                    temp += a[i * n + t] * b[t * k + j];
                c[i * k + j] = temp;
            }
        }

        for (int t = 1; t < size; t++) {
            MPI_Recv(ans, block * k, MPI_DOUBLE, t, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < block; i++) {
                for (int j = 0; j < k; j++) {
                    c[((t - 1) * block + i) * k + j] = ans[i * k + j];
                }
            }
        }

        double stop = MPI_Wtime();

        // print matrix A, B, and result C
        printf("matrix A: \n");
        print_matrix(m, n, a);
        printf("matrix B: \n");
        print_matrix(n, k, b);
        printf("matrix C: \n");
        print_matrix(m, k, c);

        printf("running time: %lf s\n", stop - start);
        delete[] a;
    }
    else {
        double* buffer = new double[n * block];

        MPI_Recv(b, n * k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(buffer, block, blockType, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < block; i++) {
            for (int j = 0; j < k; j++) {
                double temp = 0;
                for (int t = 0; t < n; t++)
                    temp += buffer[i * n + t] * b[t * k + j];
                ans[i * k + j] = temp;
            }
        }

        MPI_Send(ans, block * k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        delete[] buffer;
        delete[] ans;
    }

    delete[] b;
    MPI_Finalize();
    return 0;
}
