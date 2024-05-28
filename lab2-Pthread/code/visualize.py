import matplotlib.pyplot as plt

matrix_sizes = [512, 1024, 2048]
thread_counts = [1, 2, 3, 4, 5, 6, 7, 8]

data = [
    [445.1, 239.9, 211.7, 149.0, 143.9, 121.9, 120.8, 148.4],
    [4706.7, 2614.3, 1970.3, 1582.2, 1295.1, 1295.6, 1153.4, 1110.4],
    [62599.8, 31105.9, 22407.5, 18967.1, 16732.3, 15378.2, 14208.6, 14122.7]
]

colors = ['b', 'g', 'r']

plt.title("Performance vs. Thread Count for Different Matrix Sizes")
plt.xlabel("Thread Count")
plt.ylabel("Execution Time (ms)")

for i in range(len(matrix_sizes)):
    plt.plot(thread_counts, data[i], label=f"Matrix Size {matrix_sizes[i]}", color=colors[i])

plt.legend(loc="upper right")

plt.xticks(thread_counts)
plt.grid()

plt.savefig("performance_for_q1.png", dpi = 800)
plt.show()
