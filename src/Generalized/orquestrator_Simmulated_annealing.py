import numpy as np
import itertools
import datetime
import time
import warnings
import os
import concurrent.futures
import pandas as pd
import config
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Import the required functions from the QP module
from src.Generalized.qp_problem_new import (solve_qp, simulate_stable_preferences, create_q_matrix,
                                            create_box_constraints, create_preference_class_constraints,
                                            map_types_to_preference_classes, create_index_grid,
                                            create_type_share_constraints)


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


def expand_utility_matrix(template):
    """
    Expand a 10-parameter template into a complete utility matrix.

    Mapping rules:
      - Alone: For group A, both (('A',1),0) and (('A',2),0) use template key (('A','X'),0).
               For group B, both (('B',1),0) and (('B',2),0) use template key (('B','X'),0).

      - Within-group interactions for group A:
          * Same characteristic: use key (('A','X'), ('A','X'))
          * Different characteristics: use key (('A','X'), ('A','Y'))
          Similarly for group B with keys (('B','X'), ('B','X')) and (('B','X'), ('B','Y')).

      - Cross-group interactions:
          When an agent from group A interacts with one from group B:
              - If A is 1 and B is 1: use key (('A','X'), ('B','X'))
              - If A is 1 and B is 2: use key (('A','X'), ('B','Y'))
              - If A is 2 and B is 1: use key (('B','X'), ('A','Y'))
              - If A is 2 and B is 2: use key (('B','X'), ('A','X'))
          When the order is reversed (B then A), swap the keys:
              - If B is 1 and A is 1: use key (('B','X'), ('A','X'))
              - If B is 1 and A is 2: use key (('B','X'), ('A','Y'))
              - If B is 2 and A is 1: use key (('A','X'), ('B','Y'))
              - If B is 2 and A is 2: use key (('A','X'), ('B','X'))
    """
    expanded_matrix = {}

    agents_A = [('A', 1), ('A', 2)]
    agents_B = [('B', 1), ('B', 2)]

    # Alone cases
    for agent in agents_A:
        expanded_matrix[(agent, 0)] = template.get((('A', 'X'), 0))
    for agent in agents_B:
        expanded_matrix[(agent, 0)] = template.get((('B', 'X'), 0))

    # Within-group interactions for group A
    for a1 in agents_A:
        for a2 in agents_A:
            if a1 == a2:
                expanded_matrix[(a1, a2)] = template.get((('A', 'X'), ('A', 'X')))
            else:
                expanded_matrix[(a1, a2)] = template.get((('A', 'X'), ('A', 'Y')))

    # Within-group interactions for group B
    for b1 in agents_B:
        for b2 in agents_B:
            if b1 == b2:
                expanded_matrix[(b1, b2)] = template.get((('B', 'X'), ('B', 'X')))
            else:
                expanded_matrix[(b1, b2)] = template.get((('B', 'X'), ('B', 'Y')))

    # Cross-group interactions: A (first) with B (second)
    for a in agents_A:
        for b in agents_B:
            if a[1] == 1 and b[1] == 1:
                key = (('A', 'X'), ('B', 'X'))
            elif a[1] == 1 and b[1] == 2:
                key = (('A', 'X'), ('B', 'Y'))
            elif a[1] == 2 and b[1] == 1:
                key = (('B', 'X'), ('A', 'Y'))
            elif a[1] == 2 and b[1] == 2:
                key = (('B', 'X'), ('A', 'X'))
            expanded_matrix[(a, b)] = template.get(key)

    # Cross-group interactions: B (first) with A (second)
    for b in agents_B:
        for a in agents_A:
            if b[1] == 1 and a[1] == 1:
                key = (('B', 'X'), ('A', 'X'))
            elif b[1] == 1 and a[1] == 2:
                key = (('B', 'X'), ('A', 'Y'))
            elif b[1] == 2 and a[1] == 1:
                key = (('A', 'X'), ('B', 'Y'))
            elif b[1] == 2 and a[1] == 2:
                key = (('A', 'X'), ('B', 'X'))
            expanded_matrix[(b, a)] = template.get(key)

    return expanded_matrix


def evaluate_solution(template, agent_characteristics, cond_distributions, solver="scipy"):
    """Evaluate a solution using the existing QP solver."""
    try:
        utility_matrix = expand_utility_matrix(template)
        all_types = create_agent_types(agent_characteristics)
        preference_classes = get_preference_classes(agent_characteristics)
        valid_combinations = generate_valid_combinations(preference_classes)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            _, result = solve_qp(
                all_types,
                utility_matrix,
                valid_combinations,
                preference_classes,
                cond_distributions,
                agent_characteristics,
                tolerance=1e-6,
                pack=solver
            )
        return result if result is not None else float('inf')
    except Exception as e:
        return float('inf')


def format_time(seconds):
    """Convert seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


def generate_initial_solution():
    """Generate a random initial solution within reasonable bounds."""
    return {
        (('A', 'X'), 0): np.random.uniform(0, 0.3),
        (('B', 'X'), 0): np.random.uniform(0, 0.3),
        (('A', 'X'), ('A', 'X')): np.random.uniform(0.5, 1.0),
        (('B', 'X'), ('B', 'X')): np.random.uniform(0.5, 1.0),
        (('A', 'X'), ('A', 'Y')): np.random.uniform(0.2, 0.8),
        (('B', 'X'), ('B', 'Y')): np.random.uniform(0.2, 0.8),
        (('A', 'X'), ('B', 'X')): np.random.uniform(0.1, 0.7),
        (('B', 'X'), ('A', 'X')): np.random.uniform(0.1, 0.7),
        (('A', 'X'), ('B', 'Y')): np.random.uniform(0.1, 0.7),
        (('B', 'X'), ('A', 'Y')): np.random.uniform(0.1, 0.7)
    }


def generate_neighbor(current_solution, temperature):
    """
    Generate a neighboring solution using adaptive exploration strategies based on temperature.

    Args:
        current_solution: Current solution dictionary
        temperature: Current temperature (0-1 range)

    Returns:
        Dictionary containing the new neighboring solution
    """
    # Define exploration strategies and their temperature-dependent probabilities
    if temperature > 0.7:
        # High temperature: prioritize exploration
        mode_probs = [0.2, 0.3, 0.3, 0.2]  # [small, large, targeted, reset]
    elif temperature > 0.4:
        # Medium temperature: mix of exploration and exploitation
        mode_probs = [0.4, 0.3, 0.2, 0.1]
    elif temperature > 0.1:
        # Lower temperature: more refinement
        mode_probs = [0.6, 0.2, 0.1, 0.1]
    else:
        # Very low temperature: mostly refinement
        mode_probs = [0.8, 0.1, 0.1, 0.0]

    exploration_mode = np.random.choice(
        ["small_perturbation", "large_perturbation", "targeted_jump", "complete_reset"],
        p=mode_probs
    )

    # Make a copy of the current solution
    neighbor = current_solution.copy()

    # Group parameters by type for targeted perturbations
    alone_params = [k for k in neighbor.keys() if k[1] == 0]
    same_type_params = [k for k in neighbor.keys() if isinstance(k[1], tuple) and k[0][0] == k[1][0]]
    cross_type_params = [k for k in neighbor.keys() if isinstance(k[1], tuple) and k[0][0] != k[1][0]]

    # Apply the selected exploration strategy
    if exploration_mode == "small_perturbation":
        # Small changes to many parameters
        perturbation_scale = 0.1 * temperature

        # Perturb a subset of parameters with small changes
        for param in neighbor.keys():
            if np.random.random() < 0.7:  # 70% chance to perturb each parameter
                perturbation = np.random.normal(0, perturbation_scale)

                if param in alone_params:
                    neighbor[param] = np.clip(neighbor[param] + perturbation, 0.01, 0.3)
                elif param in same_type_params:
                    neighbor[param] = np.clip(neighbor[param] + perturbation, 0.5, 1.0)
                else:  # cross_type_params
                    neighbor[param] = np.clip(neighbor[param] + perturbation, 0.1, 0.7)

    elif exploration_mode == "large_perturbation":
        # Larger changes to fewer parameters
        perturbation_scale = 0.3

        # Select a random subset of parameters to perturb (30%)
        all_params = list(neighbor.keys())
        num_to_perturb = max(1, int(len(all_params) * 0.3))
        # Use Python's random module instead of NumPy for selecting tuple keys
        import random
        params_to_perturb = random.sample(all_params, num_to_perturb)

        for param in params_to_perturb:
            # Apply larger perturbations
            perturbation = np.random.normal(0, perturbation_scale)

            if param in alone_params:
                neighbor[param] = np.clip(neighbor[param] + perturbation, 0.01, 0.3)
            elif param in same_type_params:
                neighbor[param] = np.clip(neighbor[param] + perturbation, 0.5, 1.0)
            else:  # cross_type_params
                neighbor[param] = np.clip(neighbor[param] + perturbation, 0.1, 0.7)

    elif exploration_mode == "targeted_jump":
        # Target a specific group of parameters
        target_group = np.random.choice(["alone", "same_type", "cross_type"])

        if target_group == "alone":
            # Significant changes to alone utilities
            for param in alone_params:
                # Either make it very small or relatively large
                if np.random.random() < 0.5:
                    neighbor[param] = np.random.uniform(0.01, 0.1)
                else:
                    neighbor[param] = np.random.uniform(0.1, 0.3)

        elif target_group == "same_type":
            # Target same-type interactions
            for param in same_type_params:
                # Either make it smaller or larger
                if np.random.random() < 0.3:
                    neighbor[param] = np.random.uniform(0.5, 0.7)
                else:
                    neighbor[param] = np.random.uniform(0.7, 1.0)

        elif target_group == "cross_type":
            # Target cross-group interactions
            for param in cross_type_params:
                # Randomize completely within bounds
                neighbor[param] = np.random.uniform(0.1, 0.7)

    elif exploration_mode == "complete_reset":
        # Generate a completely new solution
        # But possibly preserve some aspects of the current solution

        # Create a fresh solution
        new_solution = generate_initial_solution()

        # With 30% probability, keep some of the original solution's characteristics
        if np.random.random() < 0.3:
            # Choose a category to preserve
            preserve_category = np.random.choice(["alone", "same_type", "cross_type"])

            if preserve_category == "alone":
                for param in alone_params:
                    new_solution[param] = neighbor[param]
            elif preserve_category == "same_type":
                for param in same_type_params:
                    new_solution[param] = neighbor[param]
            elif preserve_category == "cross_type":
                for param in cross_type_params:
                    new_solution[param] = neighbor[param]

        return new_solution

    return neighbor


def run_simulated_annealing(cond_distributions, agent_characteristics,
                            initial_temp=1.0, final_temp=0.01, cooling_rate=0.97,
                            iterations_per_temp=50, solver="scipy", major=""):
    """
    Run enhanced simulated annealing to find optimal parameters.

    Args:
        cond_distributions: Dictionary of conditional distributions
        agent_characteristics: List of agent characteristics
        initial_temp: Initial temperature (default: 1.0)
        final_temp: Final temperature (default: 0.01)
        cooling_rate: Cooling rate (default: 0.97)
        iterations_per_temp: Iterations per temperature (default: 50)
        solver: QP solver to use (default: "scipy")
        major: Major identifier for output filename

    Returns:
        DataFrame with results
    """
    # Setup output directory and logging
    output_dir = config.GENERAL_PARAMETER_PATH
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize
    current_solution = generate_initial_solution()
    current_energy = evaluate_solution(current_solution, agent_characteristics,
                                       cond_distributions, solver)
    best_solution = current_solution.copy()
    best_energy = current_energy

    # Store all accepted solutions
    solutions_data = []

    # Setup for restart mechanism
    iterations_since_improvement = 0
    restart_threshold = 200  # Restart after 200 iterations without improvement

    # Setup for early stopping detection
    energy_history = []
    plateau_window = 50  # Window size for plateau detection
    plateau_threshold = 1e-6  # Threshold for std dev to detect plateau

    temperature = initial_temp
    iteration = 0
    start_time = time.time()

    print(f"\nStarting Enhanced Simulated Annealing search:")
    print(f"Initial temperature: {initial_temp}")
    print(f"Cooling rate: {cooling_rate}")
    print(f"Iterations per temperature: {iterations_per_temp}")
    print("=" * 50)

    while temperature > final_temp:
        for i in range(iterations_per_temp):
            iteration += 1

            # Generate and evaluate neighbor
            neighbor = generate_neighbor(current_solution, temperature)
            neighbor_energy = evaluate_solution(neighbor, agent_characteristics,
                                                cond_distributions, solver)

            # Decide whether to accept neighbor
            delta_e = neighbor_energy - current_energy
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy

                # Store the accepted solution
                solution_dict = {
                    'iteration': iteration,
                    'temperature': temperature,
                    'result': current_energy,
                    'major': major,
                    'timestamp': timestamp
                }

                # Store parameters with their full tuple structure
                ordered_params = [
                    (('A', 'X'), 0),
                    (('B', 'X'), 0),
                    (('A', 'X'), ('A', 'X')),
                    (('B', 'X'), ('B', 'X')),
                    (('A', 'X'), ('A', 'Y')),
                    (('B', 'X'), ('B', 'Y')),
                    (('A', 'X'), ('B', 'X')),
                    (('B', 'X'), ('A', 'X')),
                    (('A', 'X'), ('B', 'Y')),
                    (('B', 'X'), ('A', 'Y'))
                ]
                for param in ordered_params:
                    solution_dict[str(param)] = current_solution[param]

                solutions_data.append(solution_dict)

                # Update best solution if necessary
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = current_solution.copy()
                    iterations_since_improvement = 0
                else:
                    iterations_since_improvement += 1

            else:
                # Even if we don't accept the solution, increment iterations since improvement
                iterations_since_improvement += 1

            # Track energy history for plateau detection
            energy_history.append(current_energy)
            if len(energy_history) > plateau_window:
                energy_history.pop(0)  # Keep only the most recent values

            # Check for restart conditions

            # 1. Too many iterations without improvement
            if iterations_since_improvement > restart_threshold:
                print(f"Restarting after {iterations_since_improvement} iterations without improvement")
                # Restart from best solution with some perturbation
                current_solution = generate_neighbor(best_solution, 0.5)  # Medium perturbation
                current_energy = evaluate_solution(current_solution, agent_characteristics,
                                                   cond_distributions, solver)
                iterations_since_improvement = 0
                energy_history = []  # Reset energy history

            # 2. Detected plateau (energy not changing significantly)
            elif len(energy_history) == plateau_window:
                energy_std = np.std(energy_history)
                if energy_std < plateau_threshold:
                    print(f"Detected plateau at iteration {iteration}, performing restart")
                    # Generate completely new solution
                    current_solution = generate_initial_solution()
                    current_energy = evaluate_solution(current_solution, agent_characteristics,
                                                       cond_distributions, solver)
                    iterations_since_improvement = 0
                    energy_history = []  # Reset energy history

            # Progress reporting
            if iteration % 50 == 0 or iteration % iterations_per_temp == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration: {iteration}")
                print(f"Temperature: {temperature:.6f}")
                print(f"Current Energy: {current_energy:.6f}")
                print(f"Best Energy: {best_energy:.6f}")
                print(f"Time elapsed: {format_time(elapsed_time)}")
                print(f"Iterations since improvement: {iterations_since_improvement}")
                print("-" * 50)

        # Cool down
        temperature *= cooling_rate

    # Create DataFrame from results
    results_df = pd.DataFrame(solutions_data)

    # Save results
    output_path = os.path.join(output_dir, f"{timestamp}_sa_results_{major}.csv")
    results_df.to_csv(output_path, index=False)

    print("\nEnhanced Simulated Annealing completed.")
    print(f"Total iterations: {iteration}")
    print(f"Best energy found: {best_energy:.6f}")
    print(f"Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    # WHEN RUNNING FROM A VM
    # csv_path = "/home/santiagoneirahernandez/preferences_networks/data/Datasets/Type_shares/Observed_type_shares_non_zeros_generalized.csv"
    csv_path = os.path.join(config.TYPE_SHARES_FOLDER_PATH_GEN,
                            "Observed_type_shares_non_zeros_generalized.csv")
    df = pd.read_csv(csv_path)
    # print(df.head(2))
    df = df[df['major'] == "Economía"]
    # print(df.head(2))
    df = df[df['term'] == 201610]
    # print(df)
    # print(type(df.iloc[0,0]))
    cond_distributions = {
        (('A', 1), 0): df.iloc[0, 0],
        (('A', 1), ('A', 1)): df.iloc[0, 1],
        (('A', 1), ('A', 2)): df.iloc[0, 2],
        (('A', 1), ('B', 1)): df.iloc[0, 3],
        (('A', 1), ('B', 2)): df.iloc[0, 4],
        (('A', 2), 0): df.iloc[0, 5],
        (('A', 2), ('A', 1)): df.iloc[0, 6],
        (('A', 2), ('A', 2)): df.iloc[0, 7],
        (('A', 2), ('B', 1)): df.iloc[0, 8],
        (('A', 2), ('B', 2)): df.iloc[0, 9],
        (('B', 1), 0): df.iloc[0, 10],
        (('B', 1), ('A', 1)): df.iloc[0, 11],
        (('B', 1), ('A', 2)): df.iloc[0, 12],
        (('B', 1), ('B', 1)): df.iloc[0, 13],
        (('B', 1), ('B', 2)): df.iloc[0, 14],
        (('B', 2), 0): df.iloc[0, 15],
        (('B', 2), ('A', 1)): df.iloc[0, 16],
        (('B', 2), ('A', 2)): df.iloc[0, 17],
        (('B', 2), ('B', 1)): df.iloc[0, 18],
        (('B', 2), ('B', 2)): df.iloc[0, 19],
    }

    results_df = run_simulated_annealing(
        cond_distributions=cond_distributions,
        agent_characteristics=config.agent_characteristics,
        initial_temp=1.0,
        final_temp=0.001,
        cooling_rate=0.97,
        iterations_per_temp=50,
        solver="scipy",
        major="Economics_201610"
    )


