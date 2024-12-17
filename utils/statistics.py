import matplotlib.pyplot as plt
import numpy as np
import taichi as ti


def plot_taichi_data(taichi_data_container: ti.template()):
    np_data = taichi_data_container.to_numpy()
    data = np_data[np_data != 0]
    bin_width = 0.01
    bins = np.arange(0.1, 0.2 + bin_width, bin_width)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='green', edgecolor='black')
    plt.title("Histogram of Weights", fontsize=14)
    plt.xlabel("Value Range", fontsize=12)
    plt.ylabel("Times", fontsize=12)
    plt.xlim(0.0, 0.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("histogram_with_0.1_bins.jpg", format="jpg")
    plt.show()
