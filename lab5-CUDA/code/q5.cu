# include <stdio.h>
# include <stdlib.h>
# include <sys/time.h>
# include <cudnn.h>

const double alpha = 1.0;
const double beta = 0.0;

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
    double* input = (double*)malloc(channel * N * N * sizeof(double)); 
    double* output = (double*)malloc(kernel_cnt * output_size * output_size * sizeof(double));
    double* kernel = (double*)malloc(kernel_cnt * channel * F * F * sizeof(double));

    // initialize input and kernel
    int offset = (N - 2 * padding) * (N - 2 * padding); 
    // printf("-- input --\n"); 
    for (int c = 0; c < channel; c++) {
        // printf("channel %d: \n", c);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i < padding || j < padding || N - i <= padding || N - j <= padding) {
                    input[c * N * N + i * N + j] = 0; 
                    // printf("%4.0f ", input[c * N * N + i * N + j]);
                }
                else {
                    input[c * N * N + i * N + j] = (i - padding) * (N - 2 * padding) + j - padding + 1 + c * offset; 
                    // printf("%4.0f ", input[c * N * N + i * N + j]);
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
                    kernel[(d * kernel_cnt + c) * F * F + i * F + j] = 1.0;
                }
            }
        }
    }

    // convolution
    // cuDNN intialization
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, channel, N, N);

    // kernel
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, kernel_cnt, channel, F, F);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, kernel_cnt, output_size, output_size);

    double* d_input;
    double* d_kernel;
    double* d_output;

    cudaMalloc((void**)&d_input, channel * N * N * sizeof(double));
    cudaMalloc((void**)&d_kernel, kernel_cnt * channel * F * F * sizeof(double));
    cudaMalloc((void**)&d_output, kernel_cnt * output_size * output_size * sizeof(double));

    gettimeofday(&start, NULL);
    cudaMemcpy(d_input, input, channel * N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_cnt * channel * F * F * sizeof(double), cudaMemcpyHostToDevice);

    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, filter_desc, d_kernel, conv_desc, algo, nullptr, 0, &beta, output_desc, d_output);

    cudaMemcpy(output, d_output, kernel_cnt * output_size * output_size * sizeof(double), cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);
    
    printf("running time: %f ms\n\n\n", (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_usec - start.tv_usec) / 1e3);

    // printf("-- output --\n");
    // for (int d = 0; d < kernel_cnt; d++) {
    //     printf("channel %d: \n", d); 
    //     for (int i = 0; i < output_size; i++) {
    //         for (int j = 0; j < output_size; j++) {
    //             printf("%4.0f ", output[d * output_size * output_size + i * output_size + j]); 
    //         }
    //         printf("\n"); 
    //     }
    // }

    // free memory
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroy(cudnn);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(input); 
    free(output); 
    free(kernel); 

    return 0; 
}
