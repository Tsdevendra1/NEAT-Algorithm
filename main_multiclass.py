from NEAT_multiclass import NEATMultiClass
from config_multiclass import ConfigMultiClass
from data_storage import get_circle_data
from neural_network import create_data
import numpy as np
from read_mat_files import get_shm_two_class_data, get_shm_multi_class_data


def main():
    np.random.seed(1)

    # Choose which algorithm is running using keys
    algorithm_options = {0: 'xor_full', 1: 'shm_multi_class'}
    algorithm_running = algorithm_options[1]

    if algorithm_running == algorithm_options[0]:
        num_data_to_generate = 6250

        # Create data
        x_data, y_data = create_data(n_generated=num_data_to_generate, add_noise=False, use_one_hot=True)
    elif algorithm_running == algorithm_options[1]:
        # Create data
        x_data, y_data = get_shm_multi_class_data()
        num_data_to_generate = len(x_data)

    # Training data
    training_percentage = 0.8
    training_upper_limit_index = round(num_data_to_generate * training_percentage)
    x_training = x_data[0:training_upper_limit_index]
    y_training = y_data[0:training_upper_limit_index]

    # Test data
    x_test = x_data[training_upper_limit_index:]
    y_test = y_data[training_upper_limit_index:]

    f1_score_threshold = 0.95 if algorithm_running != algorithm_options[1] else None
    fitness_threshold = -0.1 if algorithm_running != algorithm_options[1] else None

    neat = NEATMultiClass(x_training_data=x_training, y_training_data=y_training, x_test_data=x_test,
                          y_test_data=y_test,
                          config=ConfigMultiClass, fitness_threshold=fitness_threshold,
                          f1_score_threshold=f1_score_threshold,
                          algorithm_running=algorithm_running)

    neat.run(max_num_generations=250, use_backprop=True, print_generation_information=True,
             show_population_weight_distribution=False)


if __name__ == "__main__":
    main()
