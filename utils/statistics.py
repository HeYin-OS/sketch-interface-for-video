import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import utils.logging as log


def plot_taichi_data(taichi_data_container: ti.template()):
    np_data = taichi_data_container.to_numpy()
    log.printLog(0, "converted into numpy array", False)
    data = np_data[np_data != 0]
    log.printLog(0, "cleaned 0 value from container", False)
    bin_width = 0.01
    bins = np.arange(0.1, 0.2 + bin_width, bin_width)
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='green', edgecolor='black')
    # 添加标题和坐标轴标签
    plt.title("Histogram of Weights", fontsize=14)
    plt.xlabel("Value Range", fontsize=12)
    plt.ylabel("Times", fontsize=12)
    plt.xlim(0.0, 0.2)
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    log.printLog(0, "plot completed", False)
    # 保存图表为 JPG 文件
    plt.savefig("histogram_with_0.1_bins.jpg", format="jpg")
    plt.show()
    log.printLog(0, "saved as jpeg", False)
