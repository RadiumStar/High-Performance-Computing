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
- `q1.c`: 第一题矩阵乘法程序
- `q2_1.c`, `q2_2.c`: 第二题静态调度和动态调度程序
- `myfor.h`: 第三题构建 `parallel_for` 程序头文件
- `myfor.c`: 第三题构建 `parallel_for` 程序实现文件
- `myfor.o`
- `myfor.so`: 第三题的动态链接库
- `q3.c`: 第三题测试程序
- `visualize.py`: 可视化比较矩阵运行时间的程序
