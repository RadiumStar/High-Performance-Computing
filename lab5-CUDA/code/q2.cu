#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

const double alpha = 1.0;
const double beta = 0.0;

int main(int argc, char* argv[]) {
    double *a, *b, *c; 
    size_t n = atoi(argv[1]); 
    size_t size = n * n * sizeof(double);

    cudaMallocManaged((void**)&a, size); 
    cudaMallocManaged((void**)&b, size);
    cudaMallocManaged((void**)&c, size);

    // initialize host matrix a & b
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            *(a + i * n + j) = i + j;
            *(b + i * n + j) = i + j;
        }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a, n, b, n, &beta, c, n);

    cudaDeviceSynchronize();
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cublasDestroy(handle);

    return 0;
}
