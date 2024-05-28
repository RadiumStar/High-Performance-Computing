# include <stdio.h>
# include <stdlib.h>
# include <sys/time.h>

int main(int argc, char* argv[]) {
    struct timeval start, end; 
    int N = atoi(argv[1]), F = atoi(argv[2]);
    int stride = atoi(argv[3]);
    int channel = atoi(argv[4]); 
    int kernel_cnt = atoi(argv[5]); 
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
    for (int d = 0; d < kernel_cnt; d++) {
        for (int c = 0; c < channel; c++) { 
            for (int i = 0; i < F; i++) {
                for (int j = 0; j < F; j++) {
                    kernel[d * kernel_cnt + c][i * F + j] = 1.0;
                }
            }
        }
    }

    // convolution
    gettimeofday(&start, NULL);
    for (int d = 0; d < kernel_cnt; d++) {
        for (int c = 0; c < channel; c++) {
            for (int i = 0; i < output_size; i++) {
                for (int j = 0; j < output_size; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < F; k++) {
                        for (int l = 0; l < F; l++) {
                            sum += input[c][(i * stride + k) * N + j * stride + l] * kernel[d * kernel_cnt + c][k * F + l]; 
                        }
                    }
                    output[d][i * output_size + j] += sum; 
                }
            }
        }
    }
    gettimeofday(&end, NULL);
    // printf("-- output --\n");
    // for (int d = 0; d < kernel_cnt; d++) {
    //     printf("channel %d: \n", d); 
    //     for (int i = 0; i < output_size; i++) {
    //         for (int j = 0; j < output_size; j++) {
    //             printf("%4.0f ", output[d][i * output_size + j]); 
    //         }
    //         printf("\n"); 
    //     }
    // }
    printf("running time: %f ms\n\n\n", (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) / 1e3);

    // free memory
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