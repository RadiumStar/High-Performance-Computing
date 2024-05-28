#include <cstdio>
#include <cstdlib>

__global__ void matrix_multiply(double* a, double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n * n) {
        int row = idx / n; 
        int col = idx % n;

        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    double *a, *b, *c; 
    size_t n = atoi(argv[1]); 
    size_t size = n * n * sizeof(double);
    int block_size = atoi(argv[2]); 

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

    dim3 blockSize(block_size);
    dim3 gridSize((n * n + blockSize.x - 1) / blockSize.x);

    matrix_multiply<<<gridSize, blockSize>>>(a, b, c, n);
    
    cudaDeviceSynchronize();
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
