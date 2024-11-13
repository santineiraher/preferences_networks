from src.Generalized.assignment import AssignmentParameters
from src.Generalized.preferences import PreferenceClass
from src.Generalized.qp_problem2 import solve_qp
import config
import numpy as np


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
    for (i, types) in enumerate(preference_classes):
        for network_type in types:
            valid_combinations.append((types, network_type))
    return valid_combinations


def create_param_functions(utility_matrix):
    """
    Create parameter functions based on an input utility matrix.
    The utility matrix maps pairs of agent characteristics to utility values.
    """

    def param_function(x_i, x_l):
        return utility_matrix.get((x_i, x_l), 0)  # Default to 0 if not found

    return param_function


def create_agent_types(agent_characteristics):
    agent_types = []

    # Add all (x,0) types
    for x in agent_characteristics:
        agent_types.append((x, 0))

    # Add all (x,y) types where x,y are agent characteristics
    for x in agent_characteristics:
        for y in agent_characteristics:
            agent_types.append((x, y))

    return agent_types


def get_preference_classes(agent_characteristics):
    preference_classes = []

    # For each characteristic x_i
    for x_i in agent_characteristics:
        # Get all possible subsets S (including empty set)
        subsets = itertools.chain.from_iterable(
            itertools.combinations(agent_characteristics, r)
            for r in range(0, len(agent_characteristics) + 1)
        )

        # For each subset S, create the preference class
        for S in subsets:
            # Just append the preference class list: [(x_i, 0)] + [(x_i, x_j) for x_j in S]
            preference_classes.append([(x_i, 0)] + [(x_i, x_j) for x_j in S])

    return preference_classes


def run_qp_pipeline(typeshares, param_distributions):
    # Step 1: Build Preference Classes with agent_characteristics

    all_types = create_agent_types(config.agent_characteristics)
    preference_classes = get_preference_classes(config.agent_characteristics)

    # Step 2: Generate Valid Combinations
    valid_combinations = generate_valid_combinations(preference_classes)

    # Step 3: Define a fictional utility matrix with additional values for alone and self-interaction


    utility_matrix = {
        # Alone (f(x_i, 0))
        (('A', 1), 0): 0.1,
        (('A', 2), 0): 0.1,
        (('B', 1), 0): 0.1,
        (('B', 2), 0): 0.1,
        # Interaction with same type
        (('A', 1), ('A', 1)): 0.8,
        (('A', 2), ('A', 2)): 0.7,
        (('B', 1), ('B', 1)): 0.9,
        (('B', 2), ('B', 2)): 0.75,
        # Interaction with different types
        (('A', 1), ('A', 2)): 0.6,
        (('A', 1), ('B', 1)): 0.4,
        (('A', 1), ('B', 2)): 0.2,
        (('A', 2), ('A', 1)): 0.5,
        (('A', 2), ('B', 1)): 0.3,
        (('A', 2), ('B', 2)): 0.7,
        (('B', 1), ('A', 1)): 0.4,
        (('B', 1), ('A', 2)): 0.2,
        (('B', 1), ('B', 2)): 0.5,
        (('B', 2), ('A', 1)): 0.6,
        (('B', 2), ('A', 2)): 0.4,
        (('B', 2), ('B', 1)): 0.3,
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
    cond_distributions = {
        (('A', 1), 0): 0,
        (('A', 1), ('A', 1)): 0.1,
        (('A', 1), ('A', 2)): 0.4,
        (('A', 1), ('B', 1)): 0.3,
        (('A', 1), ('B', 2)): 0.2,
        (('A', 2), 0): 0,
        (('A', 2), ('A', 1)): 0.1,
        (('A', 2), ('A', 2)): 0.4,
        (('A', 2), ('B', 1)): 0.3,
        (('A', 2), ('B', 2)): 0.2,
        (('B', 1), 0): 0,
        (('B', 1), ('A', 1)): 0.1,
        (('B', 1), ('A', 2)): 0.4,
        (('B', 1), ('B', 1)): 0.3,
        (('B', 1), ('B', 2)): 0.2,
        (('B', 2), 0): 0,
        (('B', 2), ('A', 1)): 0.1,
        (('B', 2), ('A', 2)): 0.4,
        (('B', 2), ('B', 1)): 0.3,
        (('B', 2), ('B', 2)): 0.2,
    }

    alpha_solution, result = run_qp_pipeline(typeshares, cond_distributions)

    print("Alpha solution:", alpha_solution)
    print("QP Result:", result)