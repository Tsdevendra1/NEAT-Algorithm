from NEAT import NEAT
from config import Config
from neural_network import create_data
import numpy as np


def main():
    # Keep a consistent seed to make debugging easier TODO: Check if this work's across files
    np.random.seed(1)

    num_data_to_generate = 6250
    training_percentage = 0.8
    training_upper_limit_index = round(num_data_to_generate * training_percentage)

    # Create data
    x_data, y_data = create_data(n_generated=num_data_to_generate)

    # Training data
    x_training = x_data[0:training_upper_limit_index]
    y_training = y_data[0:training_upper_limit_index]

    # Test data
    x_test = x_data[training_upper_limit_index:]
    y_test = y_data[training_upper_limit_index:]

    neat = NEAT(x_training_data=x_training, y_training_data=y_training, x_test_data=x_test, y_test_data=y_test,
                config=Config, fitness_threshold=-0.1)

    neat.run(max_num_generations=10000, use_backprop=True, print_generation_information=True)


if __name__ == "__main__":
    main()
