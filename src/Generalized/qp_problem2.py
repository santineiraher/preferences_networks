import cvxpy as cp
import numpy as np
import config


def update_matrix(matrix, valid_combinations, preference_classes):
    """
    Update the Q matrix based on valid combinations and preference classes.
    """
    agent_characteristics = config.agent_characteristics
    for i, (class_i, type_i) in enumerate(valid_combinations):
        if type_i[1] != 0:
            continue
        x = type_i[0]

        for j, (class_j, type_j) in enumerate(valid_combinations):
            if type_j[1] != 0:
                continue
            z = type_j[0]

            if ((x, z) in class_i and
                    (z, x) in class_j):
                matrix[i, j] = 1  # Set matrix value to 1 if preference conditions are met

    return matrix


def create_q_matrix(valid_combinations, preference_classes):
    """
    Create and update the Q matrix for the QP problem based on valid combinations and preference classes.
    """
    size = len(valid_combinations)
    Q = np.zeros((size, size))  # Initialize Q matrix with zeros

    # Update the matrix using the provided logic
    Qupdated = update_matrix(Q, valid_combinations, preference_classes)

    return Qupdated


def create_box_constraints(x):
    """
    Create 0 <= x[i] <= 1 constraints for all i

    Args:
        x: cvxpy Variable
    Returns:
        list of constraints
    """
    n = x.size
    constraints = []
    for i in range(n):
        constraints.append(x[i] >= 0)
        constraints.append(x[i] <= 1)
    return constraints


def create_preference_class_constraints(preference_classes, valid_combinations, x):
    """
    For each preference class, sum of x[i] where i corresponds to that class = 1

    Args:
        preference_classes: list of preference classes
        valid_combinations: list of (preference_class, type) tuples
        x: cvxpy Variable
    Returns:
        list of constraints
    """
    constraints = []

    for pc in preference_classes:
        # Find indices where valid_combination has preference class pc
        indices = [i for i, v in enumerate(valid_combinations) if v[0] == pc]
        # Sum of x[i] for these indices should be 1
        constraints.append(sum(x[i] for i in indices) == 1)

    return constraints


def map_types_to_preference_classes(all_types, preference_classes):
    """
    Creates dictionary mapping each type to the preference classes it belongs to

    Args:
        all_types: list of agent types
        preference_classes: list of preference classes (each class is a list of types)

    Returns:
        dict: maps each type to list of preference classes containing it
    """
    type_to_prefs = {}

    # Initialize dictionary with empty list for each type
    for type_i in all_types:
        type_to_prefs[type_i] = []

    # For each preference class
    for pref_class in preference_classes:
        # For each type in that preference class
        for type_i in pref_class:
            # If this type exists in all_types, add this preference class to its list
            if type_i in all_types:
                type_to_prefs[type_i].append(pref_class)

    return type_to_prefs


def create_index_grid(valid_combinations):
    """
    Creates dictionary mapping (agent_type, pref_class) to its index in valid_combinations

    Args:
        valid_combinations: list of (pref_class, type) tuples

    Returns:
        dict: maps (type, pref_class) to index in valid_combinations if such combination exists
    """
    grid = {}

    # For each valid combination
    for i, (pref_class, agent_type) in enumerate(valid_combinations):
        # Add entry mapping (agent_type, pref_class) to index i
        grid[(agent_type, frozenset(pref_class))] = i

    return grid


def create_type_share_constraints(x, all_types, prefs_agent_type, x_grid, cond_mass, cond_type_shares):
    """
    Create constraints ensuring type shares match observed population shares

    Args:
        x: cvxpy Variable
        all_types: list of agent types
        prefs_agent_type: dict mapping agent type to list of relevant preference classes
        x_grid: dict mapping (agent_type, pref_class) to index in x
        cond_mass: dict mapping preference class to its conditional mass (float 0-1)
        cond_type_shares: dict mapping agent type to its observed share in population

    Returns:
        list of constraints (one per agent type)
    """
    constraints = []

    for agent_type in all_types:
        # Get all preference classes involving this type
        relevant_prefs = prefs_agent_type[agent_type]

        # Build sum: Σ x[grid(type,pref)] * cond_mass[pref] * pref[0][0]
        type_sum = sum(
            x[x_grid[(agent_type, frozenset(pref_class))]] *
            cond_mass[frozenset(pref_class)]
            for pref_class in relevant_prefs
        )

        # Add constraint: sum = observed share
        constraints.append(type_sum == cond_type_shares[agent_type])

    return constraints


# Example usage:


def simulate_stable_preferences(base_utilities, agent_chars, preference_classes, num_sims):
    """
    Simulate stability of preference classes with separate stability for each characteristic

    Args:
        base_utilities: dict mapping agent types to their base utility
        agent_chars: list of possible agent characteristics
        preference_classes: list of preference classes (each is a set of agent types)
        num_sims: number of simulations to run

    Returns:
        dict mapping preference classes to their fraction of times being stable
    """
    # Initialize counts for each preference class
    stability_counts = {frozenset(pc): 0 for pc in preference_classes}

    # Run simulations
    for _ in range(num_sims):
        # In each simulation, find stable preference classes for each characteristic
        sim_stable_classes = set()

        # For each characteristic X, find its stable preference class
        for X in agent_chars:
            # Get simple type utility for this characteristic
            simple_type = (X, 0)
            simple_utility = base_utilities[simple_type]

            # Find all complex types (X,Y) that are stable
            stable_types = {simple_type}  # simple type is always stable

            # Check each complex type with first characteristic X
            complex_types = {t for t in base_utilities
                             if t[0] == X and t[1] != 0}

            for complex_type in complex_types:
                # Get random perturbation for this complex type
                epsilon = np.random.uniform(-1, 0)
                perturbed_utility = base_utilities[complex_type] + epsilon

                # Check if complex type is stable
                if perturbed_utility > simple_utility:
                    stable_types.add(complex_type)

            # Find preference class that matches these stable types exactly
            for pc in preference_classes:
                if set(pc) == stable_types:
                    sim_stable_classes.add(frozenset(pc))

        # Add count for each stable preference class in this simulation
        for pc in sim_stable_classes:
            stability_counts[pc] += 1

    # Convert counts to fractions
    stability_fractions = {
        pc: count / num_sims
        for pc, count in stability_counts.items()
    }

    return stability_fractions


def make_psd(Q):
    """
    Make matrix positive semidefinite using eigenvalue clipping

    Args:
        Q: input symmetric matrix

    Returns:
        Modified positive semidefinite matrix
    """
    # First ensure symmetry
    Q = (Q + Q.T) / 2

    # Get eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Q)

    # Print diagnostic information
    print(f"Original eigenvalues range: [{min(eigvals):.10f}, {max(eigvals):.10f}]")
    print(f"Number of negative eigenvalues: {sum(eigvals < 0)}")

    # Clip negative eigenvalues to small positive number
    eigvals = np.maximum(eigvals, 1e-10)

    # Reconstruct matrix
    Q_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Print modification info
    diff = np.max(np.abs(Q_psd - Q))
    print(f"Maximum absolute change in any entry: {diff:.10f}")

    return Q_psd



def solve_qp(typeshares, valid_combinations, preference_classes, param_function, tolerance=1e-6):
    """
    Solve the QP problem with tolerance.
    typeshares: A list of typeshares for each agent class.
    valid_combinations: A list of valid (class, network type) combinations.
    preference_classes: The preference classes to use for matrix updates.
    param_function: A function that takes (x_i, x_l) and returns utility values.
    tolerance: A small value used for tolerances in the constraints (default: 1e-6).
    Returns the solution to the QP problem.
    """
    # Step 1: Create the Q matrix
    Q = create_q_matrix(valid_combinations, preference_classes)

    # Step 2: Set up constraints

    n = len(valid_combinations)
    alpha = cp.Variable(n)
    constraints = []
    box_constraints = create_box_constraints(alpha)
    pref_constraints = create_preference_class_constraints(preference_classes, valid_combinations, alpha)

    prefs_agent_type = map_types_to_preference_classes(all_types, preference_classes)
    x_grid = create_index_grid(valid_combinations)
    num_sims = 10000
    distribution = simulate_stable_preferences(utility_matrix, config.agent_characteristics, preference_classes,
                                               num_sims)

    type_share_constraints = create_type_share_constraints(alpha, all_types, prefs_agent_type, x_grid, distribution,
                                                           cond_distributions)

    constraints.extend(box_constraints)
    constraints.extend(pref_constraints)
    constraints.extend(type_share_constraints)

    Qpsd = make_psd(Q)

    # Define objective function: minimize alpha'Qalpha
    objective = cp.Minimize(cp.quad_form(alpha, Qpsd))

    # Create and solve the problem
    prob = cp.Problem(objective, constraints)

    # Return the optimized values of alpha
    return alpha.value, result