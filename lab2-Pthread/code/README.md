# Pthread Lab
## Environment 
- Ubuntu 22.04 LTS

## Install

```shell
sudo apt install gcc
sudo apt install gdb
sudo apt install python3
sudo apt install python3-pip
sudo apt install python-is-python3
pip install numpy 
pip install matplotlib
sudo apt-get install libpthread-stubs0-dev
sudo apt-get install htop
```

## Code
- `q1.c`: 第一题矩阵乘法程序
- `q2_0.c`, `q2_1.c`, `q2_2.c`: 第二题数组求和程序
- `q3.c`: 第三题求解一元二次方程程序
- `q4.c`: 第四题monte-carlo方法程序
- `visualize.py`: 可视化第一题程序

## Run
- .c
    ```shell
    gcc -o file file.c -lpthread -lm
    ./file
    ```
- .py
    ```shell
    python file.py
    ```