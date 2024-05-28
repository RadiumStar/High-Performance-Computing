import matplotlib.pyplot as plt

"""q1"""
# matrix_sizes = [512, 1024, 2048]
# threads = [1, 2, 3, 4, 5, 6, 7, 8]
# colors = ['royalblue', 'mediumpurple', 'coral']

# running_times = [
#     [436.27, 223.72, 157.34, 118.87, 101.47, 87.97, 86.65, 93.48],
#     [5314.85, 2687.31, 1824.58, 1404.21, 1173.49, 1040.76, 924.48, 888.63],
#     [47357.26, 24279.94, 16586.41, 12977.71, 10779.60, 9708.70, 8682.82, 7971.73]
# ]

# fig = plt.figure()

# for i in range(len(matrix_sizes)):
#     plt.plot(threads, running_times[i], marker = 'o', label=f'Matrix Size: {matrix_sizes[i]}', color = colors[i])

# plt.xlabel('Thread Count')
# plt.ylabel('Running Time (ms)')
# plt.title('Matrix Multiplication Performance')
# plt.grid(True)
# plt.legend()

# fig.savefig("q1_matrix_multiply_performance.png", dpi = 600)
# plt.show()

"""q2"""
matrix_sizes = [512, 1024, 2048]
threads = [1, 2, 3, 4, 5, 6, 7, 8]
colors = ['royalblue', 'mediumpurple', 'coral']

running_times1 = [
    [433.82, 232.12, 159.06, 128.40, 98.00, 97.45, 88.69, 89.02], 
    [5202.15, 2659.74, 1837.62, 1626.56, 1217.23, 1077.12, 979.93, 917.97], 
    [51052.33, 29893.75, 17605.85, 13601.12, 11825.53, 14971.73, 13930.29, 13989.29]
]

running_times2 = [
    [432.27, 223.75, 159.73, 119.19, 107.08, 94.07, 97.68, 95.24   ], 
    [5359.54, 2710.29, 1838.84, 1443.26, 1247.00, 1058.71, 956.59, 897.35  ], 
    [52971.11, 25049.47, 18516.37, 13981.71, 11749.86, 10176.60, 10794.72, 10400.18 ]
]

running_times3 = [
    [ 441.82, 233.59, 161.07, 128.73, 105.19, 94.61, 83.52, 97.90], 
    [5473.45, 2990.10, 1932.65, 1488.17, 1282.08, 1234.50, 1429.10, 1579.42], 
    [52384.29, 26170.78, 18146.90, 13902.29, 12364.17, 11292.37, 11233.89, 11554.16]
]


for i in range(3):
    fig = plt.figure()
    plt.plot(threads, running_times1[i], label = "default schedule")
    plt.plot(threads, running_times2[i], label = "static schedule")
    plt.plot(threads, running_times3[i], label = "dynamic schedule")

    plt.xlabel('Thread Count')
    plt.ylabel('Running Time (ms)')
    plt.title('Matrix Multiplication Performance(size:)%d' % (matrix_sizes[i]))
    plt.grid(True)
    plt.legend()

    fig.savefig("q2_matrix_multiply_performance_size%d.png" % (matrix_sizes[i]), dpi = 600)
    plt.show()