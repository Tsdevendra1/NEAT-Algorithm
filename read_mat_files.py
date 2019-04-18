import scipy.io as sio
import numpy as np


def get_shm_multi_class_data():
    mat_contents = sio.loadmat('C:/Users/tsdev/Desktop/SHM DATA/4dof_features.mat')
    y_data = mat_contents['labels'][:, 0]
    y_data.shape = (y_data.shape[0])
    x_data = mat_contents['multi_class_feats']

    shuffled_data = np.empty([x_data.shape[0], 11])
    shuffled_data[:, 0:10] = x_data
    shuffled_data[:, 10] = y_data
    # Shuffle data because it was ordered before by class
    np.random.shuffle(shuffled_data)

    x_data = shuffled_data[:, 0:10]
    y_data = shuffled_data[:, 10]
    y_data.shape = (y_data.shape[0], 1)

    y_data_one_hot = np.zeros((y_data.shape[0], 17))
    for row in range(y_data.shape[0]):
        label = int(y_data[row, 0])
        # label - 1 for indexing reasons, for example label = 1 means that the first column (index = 0) is the one with the value one
        y_data_one_hot[row, label - 1] = 1

    # if normalise_x:
    #     # We perform these operations because for this data, the values are too high are negative causing issues during
    #     # optimisation otherwise
    #     x_data = x_data * -1
    #     x_data = x_data / 100

    return x_data, y_data_one_hot


def get_shm_two_class_data(normalise_x=True):
    mat_contents = sio.loadmat('C:/Users/tsdev/Desktop/SHM DATA/4dof_features.mat')
    y_data = mat_contents['labels'][:, 1]
    y_data.shape = (y_data.shape[0])
    x_data = mat_contents['two_class_feats']

    shuffled_data = np.empty([x_data.shape[0], 4])
    shuffled_data[:, 0:3] = x_data
    shuffled_data[:, 3] = y_data
    # Shuffle data because it was ordered before by class
    np.random.shuffle(shuffled_data)

    x_data = shuffled_data[:, 0:3]
    y_data = shuffled_data[:, 3]
    y_data.shape = (y_data.shape[0], 1)

    shuffle_check = y_data[0:400, :]
    unique, counts = np.unique(shuffle_check, return_counts=True)
    shuffle_check_length = len(shuffle_check)
    class_1_percentage = counts[0] / shuffle_check_length * 100
    class_2_percentage = counts[1] / shuffle_check_length * 100

    if normalise_x:
        # We perform these operations because for this data, the values are too high are negative causing issues during
        # optimisation otherwise
        x_data = x_data * -1
        x_data = x_data / 100

    if class_1_percentage < 40 or class_2_percentage < 40:
        raise ValueError('Imbalanced classes due to shuffle, please re-initialise')

    return x_data, y_data


def main():
    x_data, y_data = get_shm_two_class_data()
    assert (y_data.shape[1] == 1)

    x_data_multi, y_data_multi = get_shm_multi_class_data(normalise_x=False)
    print(x_data_multi.shape)


if __name__ == "__main__":
    main()
