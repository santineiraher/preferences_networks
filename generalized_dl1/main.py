from src_gen.assignment import AssignmentParameters
from src_gen.preferences import PreferenceClass
from src_gen.qp_problem2 import solve_qp
import config
import numpy as np


def generate_valid_combinations(preference_classes):
    valid_combinations = []
    for i, (agent, subsets) in enumerate(preference_classes.items(), 1):
        for network_type in subsets:
            valid_combinations.append((i, network_type))
    return valid_combinations

def create_param_functions(utility_matrix):
    """
    Create parameter functions based on an input utility matrix.
    The utility matrix maps pairs of agent characteristics to utility values.
    """
    def param_function(x_i, x_l):
        return utility_matrix.get((x_i, x_l), 0)  # Default to 0 if not found

    return param_function

def generate_valid_combinations(preference_classes):
    valid_combinations = []
    for i, (agent, subsets) in enumerate(preference_classes.items(), 1):
        for network_type in subsets:
            valid_combinations.append((i, network_type))
    return valid_combinations

def create_param_functions(utility_matrix):
    """
    Create parameter functions based on an input utility matrix.
    The utility matrix maps pairs of agent characteristics to utility values.
    """
    def param_function(x_i, x_l):
        return utility_matrix.get((x_i, x_l), 0)  # Default to 0 if not found

    return param_function

def run_qp_pipeline(typeshares, param_distributions):
    # Step 1: Build Preference Classes with agent_characteristics
    preference_class_obj = PreferenceClass(config.agent_characteristics)
    preference_classes = preference_class_obj.get_all_preference_classes()

    # Step 2: Generate Valid Combinations
    valid_combinations = generate_valid_combinations(preference_classes)

    # Step 3: Define a fictional utility matrix with additional values for alone and self-interaction
    utility_matrix = {
        # Alone (f(x_i, 0))
        ('A1', 0): 0.1,
        ('A2', 0): 0.1,
        ('B1', 0): 0.1,
        ('B2', 0): 0.1,

        # Interaction with same type
        ('A1', 'A1'): 0.8,
        ('A2', 'A2'): 0.7,
        ('B1', 'B1'): 0.9,
        ('B2', 'B2'): 0.75,

        # Interaction with different types
        ('A1', 'A2'): 0.6,
        ('A1', 'B1'): 0.4,
        ('A1', 'B2'): 0.2,
        ('A2', 'A1'): 0.5,
        ('A2', 'B1'): 0.3,
        ('A2', 'B2'): 0.7,
        ('B1', 'A1'): 0.4,
        ('B1', 'A2'): 0.2,
        ('B1', 'B2'): 0.5,
        ('B2', 'A1'): 0.6,
        ('B2', 'A2'): 0.4,
        ('B2', 'B1'): 0.3,
    }

    # Step 4: Create the parameter function using the utility matrix
    param_function = create_param_functions(utility_matrix)

    # Step 5: Solve the QP problem using the `solve_qp` function with a tolerance of 1e-6
    alpha_solution, result = solve_qp(typeshares, valid_combinations, preference_classes, param_function, tolerance=1e-6)

    print("Alpha solution:", alpha_solution)
    print("QP Result:", result)
    return alpha_solution, result

if __name__ == "__main__":
    typeshares = {"A1": 0.5, "A2": 0.3, "B1": 0.1, "B2": 0.1}
    param_distributions = {
        "A1": {"A2": 0.4, "B1": 0.3, "B2": 0.3},
        "A2": {"A1": 0.5, "B1": 0.25, "B2": 0.25}
    }

    alpha_solution,result = run_qp_pipeline(typeshares, param_distributions)

    print("Alpha solution:", alpha_solution)
    print("QP Result:", result)

