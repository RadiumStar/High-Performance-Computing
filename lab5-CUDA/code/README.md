# lab5 CUDA Programming

## Files
- `q1.cu`: 任务一 矩阵乘法
- `q2.cu`: 任务二 `cublas` 矩阵乘法
- `q3.c `: 任务三 卷积运算
- `q4.cu`: 任务四 im2col 卷积运算
- `q5.cu`: 任务五 `cudnn` 卷积运算

## Run
```shell
# task 1
nvcc -o q1 q1.cu
./q1 [matrix_size] [cuda_thread_block_size]

# task 2
nvcc -o q2 q2.cu -lcublas
./q2 [matrix_size]

# task 3
gcc -o q3 q3.c
./q3 [N] [F] [stride] [channel] [kernel_cnt]

# task 4
nvcc -o q4 q4.cu 
# set cudablocksize = 1963 on trial
./q4 [N] [F] [stride] [channel] [kernel_cnt] [cuda block size] 

# task 5
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
nvcc -o q5 q5.cu -I/opt/conda/include -L/opt/conda/lib -lcudnn
./q5 [N] [F] [stride] [channel] [kernel_cnt]
```