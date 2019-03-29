import numpy as np
import scipy.stats as stats
import sklearn.metrics


def main():
    """
    Simple f score example
    :return:
    """
    real_data = [1, 0, 1, 0, 1, 1, 0, 0, 1]
    prediction = [1, 0, 1, 0, 1, 1, 0, 0, 1]

    f1_score_2 = sklearn.metrics.f1_score(real_data, prediction)

    print(f1_score_2)


if __name__ == "__main__":
    main()
