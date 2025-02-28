import numpy as np
import pandas as pd
import os
import time
import datetime
import warnings
import concurrent.futures
import itertools
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import config

# Import functions from existing modules
from src.Generalized.qp_problem_new import solve_qp
from orquestrator_Simmulated_annealing import (
    expand_utility_matrix, create_agent_types, get_preference_classes,
    generate_valid_combinations, evaluate_solution, format_time
)


def load_utility_templates(result_file_path, tolerance=1e-2, max_per_temp=5):
    """
    Load utility templates from a simulated annealing results file,
    filtering by tolerance and balancing across temperature values.

    Args:
        result_file_path: Path to the SA results file
        tolerance: Maximum result value to include (lower is better)
        max_per_temp: Maximum number of templates to include per temperature value

    Returns:
        List of tuples (template_dict, iteration) containing utility templates
        and their iteration number
    """
    # Read the results file
    df = pd.read_csv(result_file_path)

    # Filter by tolerance
    filtered_df = df[df['result'] < tolerance]

    if filtered_df.empty:
        print(f"No solutions found with result < {tolerance}")
        return []

    # Group by temperature
    grouped_by_temp = filtered_df.groupby('temperature')

    # Select balanced samples across temperatures
    balanced_samples = []

    for temp, group in grouped_by_temp:
        # Sort by result (lower is better) and take the best max_per_temp samples
        best_samples = group.sort_values('result').head(max_per_temp)
        balanced_samples.append(best_samples)

    # Combine all selected samples
    balanced_df = pd.concat(balanced_samples)

    print(f"Selected {len(balanced_df)} balanced templates across {len(grouped_by_temp)} temperature values")

    # Extract the utility templates with iteration as ID
    templates = []
    for _, row in balanced_df.iterrows():
        template = {}
        # Get iteration as ID
        iteration = row['iteration']

        # Extract utility values from columns that start with (('
        for col in row.index:
            if col.startswith("(('"):
                # Parse the string representation of the tuple key
                key = eval(col)
                template[key] = row[col]

        templates.append((template, iteration))

    return templates


def create_distribution_from_values(values, metaparams):
    """
    Create conditional distribution from flat values using metaparameters
    for symmetry constraints.

    Args:
        values: List of values for the distribution
        metaparams: Dictionary with N_A_1, N_A_2, N_B_1, N_B_2, N values

    Returns:
        Dictionary mapping agent type pairs to distribution values
    """
    # Extract metaparameters
    N_A_1 = metaparams['N_A_1']
    N_A_2 = metaparams['N_A_2']
    N_B_1 = metaparams['N_B_1']
    N_B_2 = metaparams['N_B_2']

    # Create base distribution with unrestricted values
    cond_dist = {}

    # Alone cases - no restrictions
    cond_dist[(('A', 1), 0)] = values[0]
    cond_dist[(('A', 2), 0)] = values[1]
    cond_dist[(('B', 1), 0)] = values[2]
    cond_dist[(('B', 2), 0)] = values[3]

    # Same type interactions - no restrictions
    cond_dist[(('A', 1), ('A', 1))] = values[4]
    cond_dist[(('A', 2), ('A', 2))] = values[5]
    cond_dist[(('B', 1), ('B', 1))] = values[6]
    cond_dist[(('B', 2), ('B', 2))] = values[7]

    # Different type interactions with symmetry constraints
    # For within group A
    cond_dist[(('A', 1), ('A', 2))] = values[8]
    cond_dist[(('A', 2), ('A', 1))] = values[8] * N_A_1 / N_A_2

    # For within group B
    cond_dist[(('B', 1), ('B', 2))] = values[9]
    cond_dist[(('B', 2), ('B', 1))] = values[9] * N_B_1 / N_B_2

    # For cross-group interactions
    cond_dist[(('A', 1), ('B', 1))] = values[10]
    cond_dist[(('B', 1), ('A', 1))] = values[10] * N_A_1 / N_B_1

    cond_dist[(('A', 1), ('B', 2))] = values[11]
    cond_dist[(('B', 2), ('A', 1))] = values[11] * N_A_1 / N_B_2

    cond_dist[(('A', 2), ('B', 1))] = values[12]
    cond_dist[(('B', 1), ('A', 2))] = values[12] * N_A_2 / N_B_1

    cond_dist[(('A', 2), ('B', 2))] = values[13]
    cond_dist[(('B', 2), ('A', 2))] = values[13] * N_A_2 / N_B_2

    return cond_dist


def generate_initial_distribution():
    """
    Generate a random initial distribution within reasonable bounds.

    Returns:
        List of values for the distribution
    """
    # Generate values for:
    # 4 alone cases, 4 same-type interactions, and 6 unique different-type interactions
    values = []

    # Alone cases (small probabilities)
    for _ in range(4):
        values.append(np.random.uniform(0, 0.2))

    # Same-type interactions (larger probabilities)
    for _ in range(4):
        values.append(np.random.uniform(0.5, 0.9))

    # Different-type interactions (moderate probabilities)
    for _ in range(6):
        values.append(np.random.uniform(0.2, 0.6))

    return values


def generate_distribution_neighbor(current_values, temperature):
    """
    Generate a neighboring distribution by perturbing the current values.

    Args:
        current_values: Current distribution values
        temperature: Current temperature for SA

    Returns:
        New distribution values
    """
    neighbor = current_values.copy()

    # Select random number of parameters to perturb
    num_params = max(1, int(len(current_values) * temperature))
    indices = np.random.choice(len(current_values), num_params, replace=False)

    for idx in indices:
        current_value = neighbor[idx]
        # Perturbation size based on temperature
        perturbation = np.random.normal(0, temperature * 0.2)

        # Apply perturbation with appropriate bounds based on parameter type
        if idx < 4:  # Alone cases
            new_value = np.clip(current_value + perturbation, 0, 0.3)
        elif idx < 8:  # Same-type interactions
            new_value = np.clip(current_value + perturbation, 0.3, 0.95)
        else:  # Different-type interactions
            new_value = np.clip(current_value + perturbation, 0.1, 0.8)

        neighbor[idx] = new_value

    return neighbor


def evaluate_distribution(values, utility_matrix, metaparams, agent_characteristics, solver="scipy"):
    """
    Evaluate a distribution using the QP solver.

    Args:
        values: Distribution values
        utility_matrix: Dictionary of utility values
        metaparams: Dictionary with population parameters
        agent_characteristics: List of agent characteristics
        solver: QP solver to use

    Returns:
        Result value (lower is better)
    """
    try:
        # Create conditional distribution
        cond_distributions = create_distribution_from_values(values, metaparams)

        # Standard QP setup
        all_types = create_agent_types(agent_characteristics)
        preference_classes = get_preference_classes(agent_characteristics)
        valid_combinations = generate_valid_combinations(preference_classes)

        # Solve QP with the given utility matrix and our candidate distribution
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
        print(f"Error in evaluate_distribution: {str(e)}")
        return float('inf')


def run_counterfactual_analysis(utility_template, iteration_id, metaparams,
                                initial_temp=1.0, final_temp=0.01, cooling_rate=0.95,
                                iterations_per_temp=3, solver="scipy", major=""):
    """
    Run counterfactual analysis to find distributions consistent with given utilities.

    Args:
        utility_template: Dictionary mapping type pairs to utility values
        iteration_id: Iteration ID of the original solution
        metaparams: Dictionary with population parameters
        initial_temp: Initial temperature for SA
        final_temp: Final temperature for SA
        cooling_rate: Cooling rate for SA
        iterations_per_temp: Iterations per temperature
        solver: QP solver to use
        major: Major identifier for output

    Returns:
        DataFrame with results
    """
    # Setup output directory and logging
    output_dir = config.GENERAL_COUNTERFACTUAL_PATH
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Expand utility template to full matrix
    utility_matrix = expand_utility_matrix(utility_template)

    # Initialize
    current_values = generate_initial_distribution()
    current_energy = evaluate_distribution(
        current_values,
        utility_matrix,
        metaparams,
        config.agent_characteristics,
        solver
    )

    best_values = current_values.copy()
    best_energy = current_energy

    # Store all accepted solutions
    solutions_data = []

    temperature = initial_temp
    iteration = 0
    start_time = time.time()

    print(f"\nStarting Counterfactual Analysis:")
    print(f"Initial temperature: {initial_temp}")
    print(f"Cooling rate: {cooling_rate}")
    print("=" * 50)

    while temperature > final_temp:
        for i in range(iterations_per_temp):
            iteration += 1

            # Generate and evaluate neighbor
            neighbor_values = generate_distribution_neighbor(current_values, temperature)
            neighbor_energy = evaluate_distribution(
                neighbor_values,
                utility_matrix,
                metaparams,
                config.agent_characteristics,
                solver
            )

            # Decide whether to accept neighbor
            delta_e = neighbor_energy - current_energy
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temperature):
                current_values = neighbor_values
                current_energy = neighbor_energy

                # Create distribution from current values
                current_dist = create_distribution_from_values(current_values, metaparams)

                # Store the accepted solution
                solution_dict = {
                    'iteration': iteration,
                    'temperature': temperature,
                    'result': current_energy,
                    'major': major,
                    'timestamp': timestamp,
                    'N_A_1': metaparams['N_A_1'],
                    'N_A_2': metaparams['N_A_2'],
                    'N_B_1': metaparams['N_B_1'],
                    'N_B_2': metaparams['N_B_2'],
                    'N': metaparams['N'],
                    'original_iteration': iteration_id
                }

                # Add the distribution values
                for k, v in current_dist.items():
                    solution_dict[str(k)] = v

                solutions_data.append(solution_dict)

                # Update best solution if necessary
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_values = current_values.copy()

            # Progress reporting
            if iteration % 50 == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration: {iteration}")
                print(f"Temperature: {temperature:.6f}")
                print(f"Current Energy: {current_energy:.6f}")
                print(f"Best Energy: {best_energy:.6f}")
                print(f"Time elapsed: {format_time(elapsed_time)}")
                print("-" * 50)

        # Cool down
        temperature *= cooling_rate

    # Create DataFrame from results
    if solutions_data:
        results_df = pd.DataFrame(solutions_data)

        # Save results
        output_path = os.path.join(output_dir, f"{timestamp}_counterfactual_results_{major}_factual.csv")
        results_df.to_csv(output_path, index=False)

        print("\nCounterfactual Analysis completed.")
        print(f"Total iterations: {iteration}")
        print(f"Best energy found: {best_energy:.6f}")
        print(f"Results saved to: {output_path}")

        return results_df
    else:
        print("\nNo valid solutions found in counterfactual analysis.")
        return pd.DataFrame()


def process_template(args):
    """
    Process a single utility template for parallel execution.

    Args:
        args: Tuple containing (template_index, template_tuple, metaparams, iterations, major)

    Returns:
        Result DataFrame
    """
    template_index, template_tuple, metaparams, iterations, major = args
    template, iteration_id = template_tuple

    print(f"Processing template {template_index} (original_iteration: {iteration_id})...")

    results_df = run_counterfactual_analysis(
        utility_template=template,
        iteration_id=iteration_id,
        metaparams=metaparams,
        initial_temp=1.0,
        final_temp=0.01,
        cooling_rate=0.95,
        iterations_per_temp=iterations,
        solver="scipy",
        major=f"{major}_template{template_index}"
    )

    return results_df


def run_batch_counterfactual_analysis(result_file_path, metaparams,
                                      tolerance=1e-2, max_per_temp=5,
                                      iterations_per_temp=10, max_workers=None):
    """
    Run counterfactual analysis on multiple utility templates in parallel.

    Args:
        result_file_path: Path to the SA results file
        metaparams: Dictionary with population parameters
        tolerance: Maximum result value to include
        max_per_temp: Maximum number of templates to include per temperature value
        iterations_per_temp: Iterations per temperature for SA
        max_workers: Maximum number of workers for parallel processing

    Returns:
        List of result DataFrames
    """
    # Extract major from the file path
    file_name = os.path.basename(result_file_path)
    major = file_name.split('_')[-1].split('.')[0]  # Extract major from the filename

    # Load templates with temperature balancing
    templates = load_utility_templates(result_file_path, tolerance, max_per_temp)

    if not templates:
        return []

    # Process templates in parallel
    results = []

    if max_workers is None or max_workers > 1:
        # Parallel processing
        task_args = [
            (i, template, metaparams, iterations_per_temp, major)
            for i, template in enumerate(templates)
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_results = [executor.submit(process_template, arg) for arg in task_args]

            for future in concurrent.futures.as_completed(future_results):
                df = future.result()
                if not df.empty:
                    results.append(df)
    else:
        # Sequential processing
        for i, template in enumerate(templates):
            df = run_counterfactual_analysis(
                utility_template=template,
                metaparams=metaparams,
                initial_temp=1.0,
                final_temp=0.01,
                cooling_rate=0.95,
                iterations_per_temp=iterations_per_temp,
                solver="scipy",
                major=f"{major}_template{i}"
            )
            if not df.empty:
                results.append(df)

    return results


if __name__ == "__main__":
    # Example usage
    result_file_path = os.path.join(config.GENERAL_PARAMETER_PATH, "20250224_123144_sa_results_Economics_201610.csv")

    # Set metaparameters based on the population
    metaparams = {
        'N_A_1': 44,  # Number of type A1 agents
        'N_A_2': 22,  # Number of type A2 agents
        'N_B_1': 9,  # Number of type B1 agents
        'N_B_2': 6,  # Number of type B2 agents
        'N': 81  # Total population
    }

    # Run batch analysis with temperature balancing
    results = run_batch_counterfactual_analysis(
        result_file_path=result_file_path,
        metaparams=metaparams,
        tolerance=1e-2,
        max_per_temp=3,  # Take up to 3 solutions per temperature value
        iterations_per_temp=3,
        max_workers=None  # Use all available cores
    )

    print(f"Completed analysis with {len(results)} successful template runs")