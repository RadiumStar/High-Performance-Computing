# include <stdio.h>
# include <stdlib.h>
# include <sys/time.h>

__global__ void matrix_multiply(double* a, double* b, double* c, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        for (int col = 0; col < k; col++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += a[row * n + i] * b[i * k + col];
            }
            c[row * k + col] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    struct timeval start, end; 
    int N = atoi(argv[1]), F = atoi(argv[2]);
    int stride = atoi(argv[3]);
    int channel = atoi(argv[4]); 
    int kernel_cnt = atoi(argv[5]); 
    int block_size = atoi(argv[6]);
    int padding = (((stride - N + F) % stride + stride) % stride) / 2; 
    int output_size = (N - F + 2 * padding) / stride + 1; 
    printf("-- Parameters Information --\n"); 
    printf("N: %d, F: %d, stride: %d, channel: %d, kernel_cnt: %d, padding: %d\n", N, F, stride, channel, kernel_cnt, padding);

    N += 2 * padding;
    double** input = (double**)malloc(channel * sizeof(double*)); 
    double** output = (double**)malloc(kernel_cnt * sizeof(double*));
    double** kernel = (double**)malloc(kernel_cnt * channel * sizeof(double*));
    for (int i = 0; i < channel; i++) {
        input[i] = (double*)malloc(N * N * sizeof(double));
    }
    for (int i = 0; i < kernel_cnt; i++) {
        output[i] = (double*)malloc(output_size * output_size * sizeof(double));
    }
    for (int i = 0; i < kernel_cnt * channel; i++) {
        kernel[i] = (double*)malloc(F * F * sizeof(double));
    }

    // initialize input and kernel
    int offset = (N - 2 * padding) * (N - 2 * padding); 
    // printf("-- input --\n"); 
    for (int c = 0; c < channel; c++) {
        // printf("channel %d: \n", c);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i < padding || j < padding || N - i <= padding || N - j <= padding) {
                    input[c][i * N + j] = 0; 
                    // printf("%4.0f ", input[c][i * N + j]);
                }
                else {
                    input[c][i * N + j] = (i - padding) * (N - 2 * padding) + j - padding + 1 + c * offset; 
                    // printf("%4.0f ", input[c][i * N + j]);
                }
            } 
            // printf("\n");
        }
        // printf("\n"); 
    }
    for (int d = 0; d < kernel_cnt * channel; d++) {
        for (int c = 0; c < kernel_cnt; c++) {
            for (int i = 0; i < F; i++) {
                for (int j = 0; j < F; j++) {
                    kernel[d * kernel_cnt + c][i * F + j] = 1.0;
                }
            }
        }
    }

    // im2col
    double *a, *b, *c; 
    int m = kernel_cnt; 
    int n = channel * F * F;
    int k = output_size * output_size;
    cudaMallocManaged((void**)&a, m * n * sizeof(double));
    cudaMallocManaged((void**)&b, n * k * sizeof(double));
    cudaMallocManaged((void**)&c, m * k * sizeof(double));
    // a = (double*)malloc(channel * kernel_cnt * F * F * sizeof(double)); 
    // b = (double*)malloc(channel * F * F * output_size * output_size * sizeof(double));
    // c = (double*)malloc(kernel_cnt * output_size * output_size * sizeof(double)); 

    // initialize im2col values 
    gettimeofday(&start, NULL);
    // a: kernel -> col
    // printf("-- kernel to col --\n"); 
    for (int d = 0; d < kernel_cnt; d++) {
        for (int c = 0; c < channel; c++) {
            for (int i = 0; i < F * F; i++) {
                a[(d * kernel_cnt + c) * F * F + i] = kernel[d * kernel_cnt + c][i]; 
            }
        }
    }
    
    // b: input -> matrix
    for (int c = 0; c < channel; c++) {
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < output_size; j++) {
                for (int p = 0; p < F; p++) {
                    for (int q = 0; q < F; q++) {
                        b[(c * F * F + p * F + q) * k + i * output_size + j] = input[c][(i * stride + p) * N + j * stride + q];
                    }
                }
            }
        }
    }

    dim3 blockSize(block_size); 
    dim3 gridSize((n * k + blockSize.x - 1) / blockSize.x);
    matrix_multiply<<<gridSize, blockSize>>>(a, b, c, m, n, k); 

    cudaDeviceSynchronize();
    // printf("-- Output --\n"); 
    // for (int i = 0; i < output_size; i++) {
    //     for (int j = 0; j < output_size; j++) {
    //         printf("%4.0f ", c[i * output_size + j]); 
    //     }
    //     printf("\n"); 
    // }
    gettimeofday(&end, NULL);
    printf("running time: %f ms\n\n\n", (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) / 1e3);


    // free memory
    cudaFree(a); 
    cudaFree(b); 
    cudaFree(c);
    
    for (int i = 0; i < channel; i++) {
        free(input[i]);
    }
    free(input); 
    for (int i = 0; i < kernel_cnt; i++) {
        free(output[i]);
    }
    free(output); 
    for (int i = 0; i < kernel_cnt * channel; i++) {
        free(kernel[i]); 
    }
    free(kernel); 

    return 0; 
}
