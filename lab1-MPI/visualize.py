import matplotlib.pyplot as plt
COLORS = ['lightcoral', 'cornflowerblue', 'mediumorchid']

def single_draw(matrix_sizes, cores, times, title):
    # 绘制折线图
    for i in range(len(times)):
        plt.plot(cores, times[i], label=f"Matrix Size {matrix_sizes[i]}", color = COLORS[i])

    # 图表设置
    plt.xlabel('Parallel Size')
    plt.ylabel('Time')
    plt.title(title + ': Runtime vs Parallel Size')
    plt.legend()
    plt.grid(True)
    # 显示图表
    plt.savefig(title + '.png', dpi = 600)
    plt.show()

def compare_draw(matrix_size, cores, times1, times2, title):
    # 绘制折线图
    plt.plot(cores, times1, label=f"Matrix Size {matrix_size} P2P", color = COLORS[0])
    plt.plot(cores, times2, label=f"Matrix Size {matrix_size} Collective", color = COLORS[1])

    # 图表设置
    plt.xlabel('Parallel Size')
    plt.ylabel('Time')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # 显示图表
    plt.savefig(title + '.png', dpi = 600)
    plt.show()

# 数据
matrix_sizes = [512, 1024, 2048]
cores = [1, 2, 3, 4, 5, 6, 7, 8]

# 运行时间数据
times1 = [
    [0.4401 , 0.4244 , 0.2854 , 0.2172 , 0.1800 , 0.1640 , 0.1453 , 0.1296], 
    [4.8088 , 3.9332 , 2.6107 , 2.0097 , 1.6830 , 1.4784 , 1.3135 , 1.2090], 
    [50.4752, 43.4833, 32.1189, 23.0939, 18.2454, 17.5429, 14.3861, 12.1480]
]

times2 = [
    [0.4651, 0.2527, 0.1807, 0.1439, 0.1263, 0.1166, 0.1153, 0.1259],
    [4.8021, 2.9651, 2.2914, 2.1073, 1.8927, 1.6003, 1.4674, 1.3618],
    [51.5778, 29.8343, 22.2448, 18.703, 16.6846, 14.9946, 13.6884, 13.5365]
]


single_draw(matrix_sizes, cores, times1, title = 'p2p_communication')
single_draw(matrix_sizes, cores, times2, title = 'collective_communication')

compare_draw(512, cores, times1[0], times2[0], title = 'P2P_VS_Collective_Communication(512)')
compare_draw(1024, cores, times1[1], times2[1], title = 'P2P_VS_Collective_Communication(1024)')
compare_draw(2048, cores, times1[2], times2[2], title = 'P2P_VS_Collective_Communication(2048)')