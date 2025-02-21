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


def create_template_from_uniform(uniform_array):
    """
    Convert uniform array values into template matrix format.
    """
    template_matrix = {}
    types = ['A', 'B']

    # Helper to get value from uniform array
    def get_uniform_value(i, j):
        return float(uniform_array[i % uniform_array.shape[0], j % uniform_array.shape[1]])

    # Create alone cases
    for i, t in enumerate(types):
        template_matrix[((t, 'X'), 0)] = get_uniform_value(i, 0)

    # Create interaction cases
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            if t1 == t2:
                template_matrix[((t1, 'X'), (t2, 'X'))] = get_uniform_value(i, j)
            else:
                template_matrix[((t1, 'X'), (t2, 'Y'))] = get_uniform_value(i, j)
                template_matrix[((t2, 'X'), (t1, 'Y'))] = get_uniform_value(j, i)

    return template_matrix


def process_row_to_cond_distributions(row):
    """
    Convert a row from the CSV into conditional distributions format.
    """
    cond_dist = {}
    types = ['A', 'B']
    rooms = [1, 2]

    # Process mu values from the row
    for t1, r1 in itertools.product(types, rooms):
        # Alone case
        key = f"mu_(('{t1}', {r1}), 0)"
        cond_dist[((t1, r1), 0)] = float(row[key])

        # Interaction cases
        for t2, r2 in itertools.product(types, rooms):
            key = f"mu_(('{t1}', {r1}), ('{t2}', {r2}))"
            cond_dist[((t1, r1), (t2, r2))] = float(row[key])

    return cond_dist


def run_qp_pipeline(cond_distributions, utility_matrix):
    """
    Run QP pipeline with given conditional distributions and utility matrix
    """
    all_types = create_agent_types(config.agent_characteristics)
    preference_classes = get_preference_classes(config.agent_characteristics)
    valid_combinations = generate_valid_combinations(preference_classes)

    try:
        alpha_solution, result = solve_qp(all_types, utility_matrix, valid_combinations,
                                          preference_classes, cond_distributions,
                                          tolerance=1e-6, pack="scipy")
        return alpha_solution, result
    except Exception as e:
        print(f"QP solver failed: {str(e)}")
        return None, None


def run_analysis_pipeline():
    """
    Main function to run the analysis over all CSV rows
    """
    # Setup paths
    output_dir = config.GENERAL_PARAMETER_PATH
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    csv_path = os.path.join(config.TYPE_SHARES_FOLDER_PATH_GEN,
                            "Observed_type_shares_non_zeros_generalized.csv")
    df = pd.read_csv(csv_path)

    # Create uniform utility array (10-dimensional)
    uniform_array = Big_loops.create_uniform_utility_array(size_dim=6)



    # Process each row
    for idx, row in df.iterrows():
        term = row['term']
        major = row['major']
        print(f"Processing term: {term}, major: {major}")

        # Get conditional distributions from row
        cond_distributions = process_row_to_cond_distributions(row)

        # Iterate through the 10-dimensional array
        found_solution = False
        for indices in itertools.product(range(6), repeat=10):
            # Extract values for this combination
            values = uniform_array[indices]

            # Create template matrix from these values
            template_matrix = create_template_from_values(values)

            # Expand the template matrix
            utility_matrix = Big_loops.expand_utility_matrix(template_matrix)

            # Try to solve QP problem
            alpha_solution, result = run_qp_pipeline(cond_distributions, utility_matrix)

            # If solved successfully
            if result is not None and result < 1e-4:
                output_file = os.path.join(output_dir, f"pref_{term}_{major}.csv")

                # Save results
                results_dict = {
                    'term': term,
                    'major': major,
                    'result': result,
                    'utility_values': values.tolist(),
                    ##
                }

                pd.DataFrame([results_dict]).to_csv(output_file, index=False)
                print(f"Found solution for term {term}, major {major}")
                found_solution = True
                break

        if not found_solution:
            print(f"No solution found for term {term}, major {major}")


def create_agent_types(agent_characteristics):
    """Create all possible agent types from characteristics."""
    agent_types = []
    for x in agent_characteristics:
        agent_types.append((x, 0))
    for x in agent_characteristics:
        for y in agent_characteristics:
            agent_types.append((x, y))
    return agent_types


def get_preference_classes(agent_characteristics):
    """Generate all possible preference classes."""
    preference_classes = []
    for x_i in agent_characteristics:
        subsets = itertools.chain.from_iterable(
            itertools.combinations(agent_characteristics, r)
            for r in range(0, len(agent_characteristics) + 1)
        )
        for S in subsets:
            preference_classes.append([(x_i, 0)] + [(x_i, x_j) for x_j in S])
    return preference_classes


def generate_valid_combinations(preference_classes):
    """Generate valid combinations of preference classes and types."""
    valid_combinations = []
    for (i, types) in enumerate(preference_classes):
        for network_type in types:
            valid_combinations.append((types, network_type))
    return valid_combinations


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


def generate_parameter_grid(n_points=5):
    """Generate grid of utility parameters with 10 independent dimensions."""
    param_ranges = {
        'alone_A': np.linspace(0, 1.0, n_points),  # for (('A','X'), 0)
        'alone_B': np.linspace(0, 1.0, n_points),  # for (('B','X'), 0)
        'same_A': np.linspace(0, 1.0, n_points),  # for (('A','X'), ('A','X'))
        'same_B': np.linspace(0, 1.0, n_points),  # for (('B','X'), ('B','X'))
        'diff_A': np.linspace(0, 1.0, n_points),  # for (('A','X'), ('A','Y'))
        'diff_B': np.linspace(0, 1.0, n_points),  # for (('B','X'), ('B','Y'))
        'cross_AXBX': np.linspace(0, 1.0, n_points),  # for (('A','X'), ('B','X'))
        'cross_BXAX': np.linspace(0, 1.0, n_points),  # for (('B','X'), ('A','X'))
        'cross_AXBY': np.linspace(0, 1.0, n_points),  # for (('A','X'), ('B','Y'))
        'cross_BXAY': np.linspace(0, 1.0, n_points)  # for (('B','X'), ('A','Y'))
    }

    sorted_keys = sorted(param_ranges.keys())
    combinations = itertools.product(*(param_ranges[key] for key in sorted_keys))

    template_matrices = []
    for combo in combinations:
        template = {
            (('A', 'X'), 0): combo[sorted_keys.index('alone_A')],
            (('B', 'X'), 0): combo[sorted_keys.index('alone_B')],
            (('A', 'X'), ('A', 'X')): combo[sorted_keys.index('same_A')],
            (('B', 'X'), ('B', 'X')): combo[sorted_keys.index('same_B')],
            (('A', 'X'), ('A', 'Y')): combo[sorted_keys.index('diff_A')],
            (('B', 'X'), ('B', 'Y')): combo[sorted_keys.index('diff_B')],
            (('A', 'X'), ('B', 'X')): combo[sorted_keys.index('cross_AXBX')],
            (('B', 'X'), ('A', 'X')): combo[sorted_keys.index('cross_BXAX')],
            (('A', 'X'), ('B', 'Y')): combo[sorted_keys.index('cross_AXBY')],
            (('B', 'X'), ('A', 'Y')): combo[sorted_keys.index('cross_BXAY')]
        }
        template_matrices.append(template)

    return template_matrices


def test_single_parameter_set(agent_characteristics, utility_matrix, cond_distributions, solver="scipy"):
    """Test single set of utility parameters using the specified solver."""
    all_types = create_agent_types(agent_characteristics)
    preference_classes = get_preference_classes(agent_characteristics)
    valid_combinations = generate_valid_combinations(preference_classes)

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('ignore')
        # Pass the solver parameter here to solve_qp
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
    return result, w


def process_template(args):
    """
    Worker function to process a single parameter template.

    Args:
        args: tuple containing (index, template, agent_characteristics, cond_distributions, tolerance, solver)

    Returns:
        A tuple: (index, template, result, combined_warnings, error_message)
    """
    index, template, agent_characteristics, cond_distributions, tolerance, solver = args
    try:
        utility_matrix = expand_utility_matrix(template)
        output_capture = StringIO()
        with redirect_stdout(output_capture), redirect_stderr(output_capture):
            result, caught_warnings = test_single_parameter_set(agent_characteristics, utility_matrix,
                                                                cond_distributions, solver)
        captured_output = output_capture.getvalue()

        combined_warnings = []
        if captured_output:
            for line in captured_output.splitlines():
                if "Optimization failed" in line:
                    combined_warnings.append(line)
        if caught_warnings:
            for warn in caught_warnings:
                combined_warnings.append(str(warn.message))
        return (index, template, result, combined_warnings, None)
    except Exception as e:
        return (index, template, None, None, str(e))


def estimate_runtime(n_combinations, sample_size=5, cond_distributions=None, max_workers=None, solver="scipy"):
    """
    Estimate total runtime based on sample runs, adjusted for parallel execution.
    """
    if cond_distributions is None:
        raise ValueError("cond_distributions must be provided")

    agent_characteristics = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
    template = {
        (('A', 'X'), 0): 0.1,
        (('B', 'X'), 0): 0.1,
        (('A', 'X'), ('A', 'X')): 0.8,
        (('B', 'X'), ('B', 'X')): 0.9,
        (('A', 'X'), ('A', 'Y')): 0.6,
        (('B', 'X'), ('B', 'Y')): 0.5,
        (('A', 'X'), ('B', 'X')): 0.4,
        (('B', 'X'), ('A', 'X')): 0.4,
        (('A', 'X'), ('B', 'Y')): 0.2,
        (('B', 'X'), ('A', 'Y')): 0.2,
    }
    utility_matrix = expand_utility_matrix(template)

    times = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        for _ in range(sample_size):
            start = time.time()
            test_single_parameter_set(agent_characteristics, utility_matrix, cond_distributions, solver=solver)
            times.append(time.time() - start)
    avg_time = np.mean(times)
    effective_workers = max_workers if max_workers is not None else os.cpu_count()
    total_estimated = (avg_time * n_combinations) / effective_workers
    return total_estimated


def format_time(seconds):
    """Convert seconds to human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


def run_parameter_search(cond_distributions, n_points=3, tolerance=1,
                         max_workers=None, solver="scipy", major=""):
    """Run parameter search in parallel and record results in a DataFrame."""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = config.GENERAL_PARAMETER_PATH
    os.makedirs(output_dir, exist_ok=True)

    # Define agent characteristics and generate parameter templates
    agent_characteristics = config.agent_characteristics
    templates = generate_parameter_grid(n_points)
    total_count = len(templates)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        estimated_time = estimate_runtime(total_count, cond_distributions=cond_distributions, max_workers=max_workers)

    effective_workers = max_workers if max_workers is not None else os.cpu_count()

    print(f"\nEstimated total runtime: {format_time(estimated_time)}")
    print(f"Testing {total_count} parameter combinations using {effective_workers} workers")
    print("=" * 50)

    # Create list to store results
    results_data = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings_path = os.path.join(output_dir, f"{timestamp}_warnings.log")

    consistent_count = 0
    task_args = [(i, template, agent_characteristics, cond_distributions, tolerance, solver)
                 for i, template in enumerate(templates, 1)]

    start_time = time.time()
    completed_tasks = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor, \
            open(warnings_path, "w") as warn_f:

        futures = {executor.submit(process_template, arg): arg[0] for arg in task_args}
        for future in concurrent.futures.as_completed(futures):
            completed_tasks += 1
            index, template, result, combined_warnings, error_message = future.result()

            if error_message is not None:
                warn_f.write(f"Error processing combination {index}: {error_message}\n\n")
            else:
                if combined_warnings and len(combined_warnings) > 0:
                    warn_f.write(f"\nWarnings for parameter set {index}:\n")
                    for warning_msg in combined_warnings:
                        warn_f.write(f"{warning_msg}\n")
                if result is not None and result < tolerance:
                    consistent_count += 1
                    # Create a dictionary for this result
                    result_dict = {
                        'combination_id': index,
                        'result': result,
                        'major': major,
                        'timestamp': timestamp
                    }
                    # Add all template parameters with cleaned up column names
                    for k, v in template.items():
                        # Extract components from the tuple key
                        first_type, first_char = k[0]  # e.g., ('A', 'X')
                        second_comp = k[1]  # either 0 or another tuple like ('B', 'X')

                        if second_comp == 0:
                            # Handle alone case
                            param_name = f"utility_{first_type}_alone"
                        else:
                            # Handle interaction case
                            second_type, second_char = second_comp
                            param_name = f"utility_{first_type}_{second_type}_interaction"

                        result_dict[param_name] = v
                    results_data.append(result_dict)

            if completed_tasks % max(1, total_count // 100) == 0 or completed_tasks == total_count:
                elapsed_time = time.time() - start_time
                progress = completed_tasks / total_count
                estimated_remaining = (elapsed_time / progress) - elapsed_time
                print(
                    f"Progress: Tested {completed_tasks} out of {total_count} parameter combinations ({progress:.6%})")
                print(f"Using {effective_workers} workers")
                print(f"Time elapsed: {format_time(elapsed_time)}")
                print(f"Estimated remaining: {format_time(estimated_remaining)}")
                print(f"Consistent parameters found so far: {consistent_count}")
                print("-" * 50)

    # Create DataFrame from results
    results_df = pd.DataFrame(results_data)

    # Save DataFrame to CSV
    output_path = os.path.join(output_dir, f"{timestamp}_parameter_search_results_{major}.csv")
    results_df.to_csv(output_path, index=False)

    print("\nParameter search completed.")
    print(f"Total consistent parameter sets found: {consistent_count}")
    print(f"Results saved to: {output_path}")
    print(f"Warnings saved to: {warnings_path}")

    return results_df










if __name__ == "__main__":
    # WHEN RUNNING FROM A VM
    csv_path = "/home/santiagoneirahernandez/preferences_networks/data/Datasets/Type_shares/Observed_type_shares_non_zeros_generalized.csv"
    #csv_path = os.path.join(config.TYPE_SHARES_FOLDER_PATH_GEN,
    #                        "Observed_type_shares_non_zeros_generalized.csv")
    df = pd.read_csv(csv_path)
    #print(df.head(2))
    df=df[df['major']=="Economía"]
    #print(df.head(2))
    df=df[df['term']==201610]
    #print(df)
    #print(type(df.iloc[0,0]))
    cond_distributions = {
        (('A', 1), 0): df.iloc[0,0],
        (('A', 1), ('A', 1)): df.iloc[0,1],
        (('A', 1), ('A', 2)): df.iloc[0,2],
        (('A', 1), ('B', 1)): df.iloc[0,3],
        (('A', 1), ('B', 2)): df.iloc[0,4],
        (('A', 2), 0): df.iloc[0,5],
        (('A', 2), ('A', 1)): df.iloc[0,6],
        (('A', 2), ('A', 2)): df.iloc[0,7],
        (('A', 2), ('B', 1)): df.iloc[0,8],
        (('A', 2), ('B', 2)): df.iloc[0,9],
        (('B', 1), 0): df.iloc[0,10],
        (('B', 1), ('A', 1)): df.iloc[0,11],
        (('B', 1), ('A', 2)): df.iloc[0,12],
        (('B', 1), ('B', 1)): df.iloc[0,13],
        (('B', 1), ('B', 2)): df.iloc[0,14],
        (('B', 2), 0): df.iloc[0,15],
        (('B', 2), ('A', 1)): df.iloc[0,16],
        (('B', 2), ('A', 2)): df.iloc[0,17],
        (('B', 2), ('B', 1)): df.iloc[0,18],
        (('B',2), ('B', 2)): df.iloc[0,19],
    }
    run_parameter_search(cond_distributions, n_points=4, max_workers=None, solver="scipy",major="Economics_201610")
