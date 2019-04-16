import numpy as np


def get_circle_data():
    # Skip first row because headings
    circle_data = np.loadtxt('C:/Users/tsdev/Desktop/circle_25/input.txt', skiprows=1)
    y_data = circle_data[:, circle_data.shape[1] - 1]
    x_data = circle_data[:, 1:(circle_data.shape[1] - 1)]
    return x_data, y_data


def get_spiral_data():
    # Skip first row because headings
    spiral_data = np.loadtxt('C:/Users/tsdev/Desktop/spiral_25/input.txt', skiprows=1)
    y_data = spiral_data[:, spiral_data.shape[1] - 1]
    x_data = spiral_data[:, 1:(spiral_data.shape[1] - 1)]
    return x_data, y_data
