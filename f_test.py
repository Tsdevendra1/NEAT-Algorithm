import numpy as np
import scipy.stats as stats
import sklearn.metrics


def calculate_f_score(real_data, prediction):
    predicted_num_1 = prediction.count(1)
    predicted_actual_num_1 = prediction.count(1)
    real_num_1 = real_data.count(1)

    precision = predicted_num_1 / predicted_actual_num_1
    recall = predicted_actual_num_1 / real_num_1

    return 2 * ((precision * recall) / (precision + recall))


def main():
    real_data = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    prediction = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    f1_score_2 = sklearn.metrics.f1_score(real_data, prediction, average='weighted')

    print(f1_score_2)

if __name__ == "__main__":
    main()
