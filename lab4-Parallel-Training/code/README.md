# Openmp Lab
## Environment 
- Ubuntu 22.04 LTS

## Install

```shell
sudo apt install gcc
sudo apt install gdb
sudo apt install python3
sudo apt install python3-pip
sudo apt install python-is-python3
pip install matplotlib
sudo apt-get install libpthread-stubs0-dev
sudo apt install libomp-dev
```

## Code
- `q1.c`: 第一题fft的并行化程序
- `q2.c`: 第二题heated_plate_openmp改造为mpi的程序
- `myfor.h`: 第三题构建 `parallel_for` 程序头文件
- `myfor.c`: 第三题构建 `parallel_for` 程序实现文件
- `myfor.o`
- `myfor.so`: 第三题的动态链接库
- `massif.out.[pid]`: 是valgrind捕获的内存文件，其中3693150,3168717,2119900,22099分别是线程1,2,4,8时的文件
- `fft_serial`, `fft_serial_test`, `heated_plate_openmp.c` 是题目原文件
