import numpy as np
import pandas as pd
import os
import time
import datetime
import warnings
import concurrent.futures
import itertools
import math
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

    # Select only the first row from each temperature group
    balanced_samples = []

    for temp, group in grouped_by_temp:
        # Sort by result (lower is better) and take only the first row
        best_sample = group.sort_values('result').head(1)
        balanced_samples.append(best_sample)

    # Combine all selected samples
    balanced_df = pd.concat(balanced_samples)

    ##print(f"Selected {len(balanced_df)} balanced templates across {len(grouped_by_temp)} temperature values")
    #balanced_df=balanced_df.sort_values('result')
    #balanced_df=balanced_df.head(1)
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


def create_balanced_distribution(raw_values, metaparams):
    """
    Creates a valid distribution through iterative normalization.
    Enforces three key constraints:
    1. All values between 0 and 1
    2. Probabilities sum to 1 for each first item
    3. Measure conditions are preserved

    Args:
        raw_values: List of raw probability values
        metaparams: Dictionary with N_A_1, N_A_2, N_B_1, N_B_2, N values

    Returns:
        Dictionary mapping agent type pairs to distribution values
    """
    import numpy as np

    # Define first items
    first_items = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]

    # Define all pairs for measure conditions
    pairs = [
        (('A', 1), ('A', 2)),
        (('B', 1), ('B', 2)),
        (('A', 1), ('B', 1)),
        (('A', 1), ('B', 2)),
        (('A', 2), ('B', 1)),
        (('A', 2), ('B', 2))
    ]

    # Initialize distribution
    dist = {}

    # Initialize with minimum positive values
    for first in first_items:
        dist[(first, 0)] = 0.01  # Alone
        dist[(first, first)] = 0.01  # Same-type

    for first, second in pairs:
        dist[(first, second)] = 0.01
        dist[(second, first)] = 0.01

    # Use raw values to set initial weights while ensuring positivity
    # Alone values (indices 0-3)
    for i, first in enumerate(first_items):
        if i < len(raw_values):
            dist[(first, 0)] = max(0.01, raw_values[i])

    # Same-type interactions (indices 4-7)
    for i, first in enumerate(first_items):
        idx = i + 4
        if idx < len(raw_values):
            dist[(first, first)] = max(0.01, raw_values[idx])

    # Different-type interactions (indices 8-19)
    for i, (first, second) in enumerate(pairs):
        idx = i + 8
        if idx < len(raw_values):
            dist[(first, second)] = max(0.01, raw_values[idx])

    # Multiple iterations of constraint enforcement
    for iteration in range(30):
        # Step 1: Enforce measure conditions
        for first, second in pairs:
            if (first, second) in dist and (second, first) in dist:
                N_first = metaparams[f'N_{first[0]}_{first[1]}']
                N_second = metaparams[f'N_{second[0]}_{second[1]}']

                # Calculate the average value that satisfies the measure condition
                product1 = dist[(first, second)] * N_first
                product2 = dist[(second, first)] * N_second
                avg_product = (product1 + product2) / 2

                # Update both values to satisfy the measure condition
                dist[(first, second)] = avg_product / N_first
                dist[(second, first)] = avg_product / N_second

        # Step 2: Ensure all values are between 0 and 1
        for k in dist:
            dist[k] = max(0.01, min(0.99, dist[k]))

        # Step 3: Normalize to ensure sum-to-1 for each first item
        for first in first_items:
            first_keys = [k for k in dist.keys() if k[0] == first]
            total = sum(dist[k] for k in first_keys)

            if total > 0:  # Avoid division by zero
                for k in first_keys:
                    dist[k] /= total

    # Final measure condition check and adjustment
    for first, second in pairs:
        if (first, second) in dist and (second, first) in dist:
            N_first = metaparams[f'N_{first[0]}_{first[1]}']
            N_second = metaparams[f'N_{second[0]}_{second[1]}']

            # Calculate the difference in the measure condition
            left = dist[(first, second)] * N_first
            right = dist[(second, first)] * N_second

            # If there's still a significant difference, make one final adjustment
            if abs(left - right) > 1e-5:
                avg = (left + right) / 2
                dist[(first, second)] = avg / N_first
                dist[(second, first)] = avg / N_second

                # Re-normalize for the affected first items
                for item in [first, second]:
                    first_keys = [k for k in dist.keys() if k[0] == item]
                    total = sum(dist[k] for k in first_keys)
                    if total > 0:
                        for k in first_keys:
                            dist[k] /= total

    return dist

def create_distribution_from_values(values, metaparams):
    """
    Create conditional distribution from flat values using metaparameters
    with three constraints:
    1. All values between 0 and 1
    2. Probabilities sum to 1 for each first item
    3. Measure conditions are preserved (e.g., ('A',2),('A',1) = N_A_1/N_A_2 * ('A',1),('A',2))

    Args:
        values: List of values for the distribution
        metaparams: Dictionary with N_A_1, N_A_2, N_B_1, N_B_2, N values

    Returns:
        Dictionary mapping agent type pairs to distribution values
    """
    # Use the balanced distribution function that preserves measure conditions
    # while ensuring sum-to-1 for each first item
    return create_balanced_distribution(values, metaparams)


def generate_initial_distribution():
    """
    Generate a random initial distribution with appropriate structure.

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


def generate_distribution_neighbor(current_values, temperature, metaparams):
    """
    Generate a more diverse neighboring distribution while preserving constraints.
    Uses adaptive perturbation sizes and occasional jumps to explore the solution space.

    Args:
        current_values: Current list of distribution values
        temperature: Current temperature for SA (0-1 range)
        metaparams: Dictionary with population parameters

    Returns:
        New list of distribution values that maintain constraints
    """
    import numpy as np

    # Define exploration strategy based on temperature
    exploration_modes = [
        "small_perturbation",  # Small changes to current solution
        "large_perturbation",  # Larger changes to current solution
        "targeted_jump",  # Focus changes on specific interaction types
        "complete_reset"  # Generate an entirely new solution
    ]

    # Probability distribution for exploration modes (changes with temperature)
    # At high temperatures: more resets and jumps
    # At low temperatures: more small perturbations
    if temperature > 0.7:
        mode_probs = [0.2, 0.3, 0.3, 0.2]  # High temperature: explore widely
    elif temperature > 0.4:
        mode_probs = [0.4, 0.3, 0.2, 0.1]  # Medium temperature: mix of exploration
    elif temperature > 0.1:
        mode_probs = [0.6, 0.2, 0.1, 0.1]  # Lower temperature: refine more
    else:
        mode_probs = [0.8, 0.1, 0.1, 0.0]  # Very low temperature: mostly refine

    # Select exploration mode
    exploration_mode = np.random.choice(exploration_modes, p=mode_probs)

    # First items and interaction pairs for reference
    first_items = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
    pairs = [
        (('A', 1), ('A', 2)),
        (('B', 1), ('B', 2)),
        (('A', 1), ('B', 1)),
        (('A', 1), ('B', 2)),
        (('A', 2), ('B', 1)),
        (('A', 2), ('B', 2))
    ]

    # Indices for different parameter types
    alone_indices = list(range(4))  # 0-3: Alone cases
    same_type_indices = list(range(4, 8))  # 4-7: Same-type interactions
    different_type_indices = list(range(8, 14))  # 8-13: Different-type interactions

    # Create a copy of current values as starting point
    new_values = current_values.copy()

    # Apply the selected exploration strategy
    if exploration_mode == "small_perturbation":
        # Small changes to all parameters
        perturbation_scale = 0.1 * temperature

        # Perturb the independent parameters (alone and same-type)
        for idx in alone_indices + same_type_indices:
            if np.random.random() < 0.7:  # Perturb with 70% probability
                perturbation = np.random.normal(0, perturbation_scale)

                if idx in alone_indices:
                    new_values[idx] = np.clip(new_values[idx] + perturbation, 0.01, 0.5)
                else:  # same-type indices
                    new_values[idx] = np.clip(new_values[idx] + perturbation, 0.1, 0.9)

    elif exploration_mode == "large_perturbation":
        # Larger changes to fewer parameters
        perturbation_scale = 0.3 * (1 + temperature)

        # Select a subset of parameters to perturb
        num_to_perturb = max(1, int(len(alone_indices + same_type_indices) * 0.5))
        indices_to_perturb = np.random.choice(
            alone_indices + same_type_indices,
            num_to_perturb,
            replace=False
        )

        for idx in indices_to_perturb:
            perturbation = np.random.normal(0, perturbation_scale)

            if idx in alone_indices:
                new_values[idx] = np.clip(new_values[idx] + perturbation, 0.01, 0.5)
            else:  # same-type indices
                new_values[idx] = np.clip(new_values[idx] + perturbation, 0.1, 0.9)

    elif exploration_mode == "targeted_jump":
        # Focus on one category and make significant changes
        target_category = np.random.choice(["alone", "same_type"])

        if target_category == "alone":
            # Significantly change alone probabilities
            for idx in alone_indices:
                # Either make it very small or relatively large
                if np.random.random() < 0.5:
                    new_values[idx] = np.random.uniform(0.01, 0.05)
                else:
                    new_values[idx] = np.random.uniform(0.2, 0.5)

            # Make compensating changes to same-type (to help with sum-to-1)
            for idx in same_type_indices:
                # Adjust in the opposite direction
                if new_values[idx - 4] > 0.2:  # If alone is large
                    new_values[idx] = np.random.uniform(0.1, 0.5)  # Make same-type smaller
                else:  # If alone is small
                    new_values[idx] = np.random.uniform(0.5, 0.9)  # Make same-type larger

        else:  # target_category == "same_type"
            # Significantly change same-type probabilities
            for idx in same_type_indices:
                # Either make it smaller or larger
                if np.random.random() < 0.5:
                    new_values[idx] = np.random.uniform(0.1, 0.3)
                else:
                    new_values[idx] = np.random.uniform(0.6, 0.9)

            # Make compensating changes to alone values
            for idx in alone_indices:
                # Adjust in the opposite direction
                if new_values[idx + 4] > 0.5:  # If same-type is large
                    new_values[idx] = np.random.uniform(0.01, 0.2)  # Make alone smaller
                else:  # If same-type is small
                    new_values[idx] = np.random.uniform(0.2, 0.5)  # Make alone larger

    elif exploration_mode == "complete_reset":
        # Generate completely new values
        # Alone values: typically smaller
        for i in alone_indices:
            new_values[i] = np.random.uniform(0.01, 0.5)

        # Same-type interactions: typically larger
        for i in same_type_indices:
            new_values[i] = np.random.uniform(0.1, 0.9)

        # Different-type interactions: directly set and will be rebalanced
        for i in different_type_indices:
            new_values[i] = np.random.uniform(0.05, 0.7)

    # Create a balanced distribution to enforce all constraints
    balanced_dist = create_balanced_distribution(new_values, metaparams)

    # Convert back to a list of values in the same order
    result = []

    # Alone values
    for first in first_items:
        result.append(balanced_dist[(first, 0)])

    # Same-type interactions
    for first in first_items:
        result.append(balanced_dist[(first, first)])

    # Different-type interactions (forward direction)
    for first, second in pairs:
        result.append(balanced_dist[(first, second)])

    return result


def evaluate_distribution(values, utility_matrix, metaparams, agent_characteristics,
                          solver="scipy"):
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
        # Create conditional distribution with constraints
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
                tolerance=1e-2,
                pack=solver
            )

        return result if result is not None else float('inf')
    except Exception as e:
        print(f"Error in evaluate_distribution: {str(e)}")
        return float('inf')


def validate_distribution(dist, metaparams):
    """
    Validate that the distribution meets the required constraints:
    1. All values between 0 and 1
    2. For each first item, probabilities sum to 1
    3. Measure conditions are preserved

    Args:
        dist: Dictionary mapping agent type pairs to distribution values
        metaparams: Dictionary with population parameters

    Returns:
        Boolean indicating whether the distribution is valid
        and a message describing any issues
    """
    # Check all values are between 0 and 1
    for k, v in dist.items():
        if v < 0 or v > 1:
            return False, f"Value for {k} is {v}, which is outside [0,1]"

    # Group by first item and check sum = 1
    first_items = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
    for first in first_items:
        sum_probs = sum(v for k, v in dist.items() if k[0] == first)
        if abs(sum_probs - 1.0) > 1e-6:  # Allow small floating point errors
            return False, f"Sum of probabilities for {first} is {sum_probs}, not 1.0"

    # Check measure conditions using multiplication instead of division
    # A1-A2 relation
    if (('A', 1), ('A', 2)) in dist and (('A', 2), ('A', 1)) in dist:
        left_side = dist[(('A', 2), ('A', 1))] * metaparams['N_A_2']
        right_side = dist[(('A', 1), ('A', 2))] * metaparams['N_A_1']
        if abs(left_side - right_side) > 1e-4:  # Allow small errors
            return False, f"A2-A1 * N_A_2 = {left_side}, A1-A2 * N_A_1 = {right_side}, should be equal"

    # B1-B2 relation
    if (('B', 1), ('B', 2)) in dist and (('B', 2), ('B', 1)) in dist:
        left_side = dist[(('B', 2), ('B', 1))] * metaparams['N_B_2']
        right_side = dist[(('B', 1), ('B', 2))] * metaparams['N_B_1']
        if abs(left_side - right_side) > 1e-4:
            return False, f"B2-B1 * N_B_2 = {left_side}, B1-B2 * N_B_1 = {right_side}, should be equal"

    # A1-B1 relation
    if (('A', 1), ('B', 1)) in dist and (('B', 1), ('A', 1)) in dist:
        left_side = dist[(('B', 1), ('A', 1))] * metaparams['N_B_1']
        right_side = dist[(('A', 1), ('B', 1))] * metaparams['N_A_1']
        if abs(left_side - right_side) > 1e-4:
            return False, f"B1-A1 * N_B_1 = {left_side}, A1-B1 * N_A_1 = {right_side}, should be equal"

    # A1-B2 relation
    if (('A', 1), ('B', 2)) in dist and (('B', 2), ('A', 1)) in dist:
        left_side = dist[(('B', 2), ('A', 1))] * metaparams['N_B_2']
        right_side = dist[(('A', 1), ('B', 2))] * metaparams['N_A_1']
        if abs(left_side - right_side) > 1e-4:
            return False, f"B2-A1 * N_B_2 = {left_side}, A1-B2 * N_A_1 = {right_side}, should be equal"

    # A2-B1 relation
    if (('A', 2), ('B', 1)) in dist and (('B', 1), ('A', 2)) in dist:
        left_side = dist[(('B', 1), ('A', 2))] * metaparams['N_B_1']
        right_side = dist[(('A', 2), ('B', 1))] * metaparams['N_A_2']
        if abs(left_side - right_side) > 1e-4:
            return False, f"B1-A2 * N_B_1 = {left_side}, A2-B1 * N_A_2 = {right_side}, should be equal"

    # A2-B2 relation
    if (('A', 2), ('B', 2)) in dist and (('B', 2), ('A', 2)) in dist:
        left_side = dist[(('B', 2), ('A', 2))] * metaparams['N_B_2']
        right_side = dist[(('A', 2), ('B', 2))] * metaparams['N_A_2']
        if abs(left_side - right_side) > 1e-4:
            return False, f"B2-A2 * N_B_2 = {left_side}, A2-B2 * N_A_2 = {right_side}, should be equal"

    return True, "Distribution is valid"


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
    output_dir = config.MEDICINE_COUNTERFACTUAL_PATH
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
            neighbor_values = generate_distribution_neighbor(current_values, temperature,metaparams)
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

                # Validate the distribution
                is_valid, validation_message = validate_distribution(current_dist, metaparams)
                if not is_valid:
                    print(f"Warning: Invalid distribution at iteration {iteration}: {validation_message}")
                    continue

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
        output_path = os.path.join(output_dir, f"{timestamp}_counterfactual_results_{major}_counterfactual_equaldistz.csv")
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
        cooling_rate=0.85,
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
                utility_template=template[0],
                iteration_id=template[1],
                metaparams=metaparams,
                initial_temp=1.0,
                final_temp=0.01,
                cooling_rate=0.8,
                iterations_per_temp=iterations_per_temp,
                solver="scipy",
                major=f"{major}_template{i}"
            )
            if not df.empty:
                results.append(df)

    return results


if __name__ == "__main__":
    # Example usage
    result_file_path = os.path.join(config.GENERAL_PARAMETER_PATH, "20250225_120801_sa_results_Medicine_201610.csv")

    # Set metaparameters based on the population
    metaparams = {
        'N_A_1': 18 ,  # Number of type A1 agents
        'N_A_2': 18,  # Number of type A2 agents
        'N_B_1': 11,  # Number of type B1 agents
        'N_B_2': 12,  # Number of type B2 agents
        'N': 59  # Total population
    }

    # Run batch analysis with temperature balancing
    results = run_batch_counterfactual_analysis(
        result_file_path=result_file_path,
        metaparams=metaparams,
        tolerance=1e-3,
        max_per_temp=1,  # Take up to 3 solutions per temperature value
        iterations_per_temp=6,
        max_workers=None  # Use all available cores
    )

    print(f"Completed analysis with {len(results)} successful template runs")