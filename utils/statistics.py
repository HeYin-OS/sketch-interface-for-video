import matplotlib.pyplot as plt
import numpy as np
import taichi as ti


def plot_taichi_data(taichi_data_container: ti.template()):
    np_data = taichi_data_container.to_numpy()
    print("to np")
    data = np_data[np_data != 0]
    print("clean 0")
    bin_width = 0.01
    bins = np.arange(0.1, 0.2 + bin_width, bin_width)
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='green', edgecolor='black')
    print("plot completed")
    # 添加标题和坐标轴标签
    plt.title("Histogram of Weights", fontsize=14)
    plt.xlabel("Value Range", fontsize=12)
    plt.ylabel("Times", fontsize=12)
    plt.xlim(0.0, 0.2)
    print("tags and labels added")
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    print("grid lines added")
    # 保存图表为 JPG 文件
    plt.savefig("histogram_with_0.1_bins.jpg", format="jpg")
    plt.show()
    print("saved")
