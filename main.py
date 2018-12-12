from NEAT import NEAT
from config import Config
from neural_network import create_data


def main():
    x_data, y_data = create_data(n_generated=5000)

    neat = NEAT(x_training_data=x_data, y_training_data=y_data, config=Config, fitness_threshold=0.01)

    neat.run(max_num_generations=10000, use_backprop=False, print_generation_information=True)


if __name__ == "__main__":
    main()
