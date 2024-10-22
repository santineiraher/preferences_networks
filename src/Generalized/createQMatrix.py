from itertools import product
import numpy as np
import generalized_dl1.config as config
from utils.generalized_utils import export_q_matrix_and_vars


def generate_network_types(characteristics):
    all_pairs = list(product(characteristics, repeat=2))
    special_types = [(char, 'NA') for char in characteristics]
    return all_pairs + special_types

def generate_valid_combinations(preference_classes):
    valid_combinations = []
    for i, pref_class in enumerate(preference_classes, 1):
        for network_type in pref_class:
            valid_combinations.append((i, network_type))
    return valid_combinations

def update_matrix(matrix, valid_combinations, preference_classes):
    agent_characteristics = config.agent_characteristics
    for i, (class_i, type_i) in enumerate(valid_combinations):
        if type_i[1] != 'NA':
            continue
        x = type_i[0]
        for j, (class_j, type_j) in enumerate(valid_combinations):
            if type_j[1] != 'NA':
                continue
            z = type_j[0]
            if z not in agent_characteristics:
                continue
            if ((x, z) in preference_classes[class_i - 1] and 
                (z, x) in preference_classes[class_j - 1]):
                matrix[i, j] = 1

def generate_assignment_parameters(valid_combinations):
    assignment_parameters ={}
    alphas= [f"alpha_{i}" for i in range(len(valid_combinations))]
    for i, (class_num,network_type) in enumerate(valid_combinations):
        assignment_parameters[f"alpha_{i}"]=(class_num,network_type)
    return alphas, assignment_parameters

if __name__ == '__main__':
    # Example usage
    agent_characteristics = config.agent_characteristics

    # Define preference classes
    preference_classes = config.preference_classes

    # Generate network types
    all_network_types = generate_network_types(agent_characteristics)

    # Generate valid combinations
    valid_combinations = generate_valid_combinations(preference_classes)

    # Create matrix
    matrix_size = len(valid_combinations)
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Update matrix based on rules
    update_matrix(matrix, valid_combinations, preference_classes)

    # Generate assignment parameters
    alpha, assignment_parameters = generate_assignment_parameters(valid_combinations)

    # Display output for checking
    print("Q Matrix:")
    print(matrix)

    print("\nalpha:")
    print(alpha)

    print("\n Assignment parameters ")
    print(assignment_parameters)

    export_q_matrix_and_vars(matrix,assignment_parameters)