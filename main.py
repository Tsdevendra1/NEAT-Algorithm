from NEAT import NEAT
from config import Config
from neural_network import create_data
import numpy as np


def main():

    # Keep a consistent seed to make debugging easier TODO: Check if this work's across files
    np.random.seed(1)

    x_data, y_data = create_data(n_generated=5000)

    neat = NEAT(x_training_data=x_data, y_training_data=y_data, config=Config, fitness_threshold=-0.1)

    neat.run(max_num_generations=10000, use_backprop=True, print_generation_information=True)


if __name__ == "__main__":
    main()
