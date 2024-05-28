# Lab1 HPC

- `q1.cpp`: 点到点通信
- `q2.cpp`: 集合通信
- `q2_2.cpp`: 将点到点通信改造为采用 `mpi_type_create_struct` 聚合 mpi 进程内变量后进行通信
- `matrix_multiply.h`: 矩阵乘法库函数头文件
- `matrix_multiply.cpp`: 矩阵乘法库函数实现文件
- `libmatrix_multiply.so`: 共享函数库，如果你想使用，请在同一目录下放置该文件
    ```cpp
    #include "matrix_multiply"
    ```
    之后采用
    ```shell
    mpicxx test.cpp -o test -I. -L. -Wl,-rpath=. -libmatrix_multiply
    ```
    进行链接编译
- `test.cpp`: 测试共享函数库文件