import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from neural_network import create_data


def main():
    # EXAMPLE ON XOR DATA
    x_data, y_data = create_data(n_generated=5000, add_noise=True)
    arr1 = x_data[:, 0]
    arr2 = x_data[:, 1]
    labl = y_data[:, 0]

    color = ['red' if l == 0 else 'green' for l in labl]
    plt.scatter(arr1, arr2, color=color)
    plt.show()


if __name__ == "__main__":
    main()
