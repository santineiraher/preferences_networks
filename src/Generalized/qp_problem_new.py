import cvxpy as cp
from scipy.optimize import minimize
import numpy as np


def update_matrix(matrix, valid_combinations, preference_classes):
    """
    Update the Q matrix based on valid combinations and preference classes.

    Args:
        matrix: Initial zero matrix to be updated
        valid_combinations: List of (preference_class, type) tuples
        preference_classes: List of preference classes for matrix updates

    Returns:
        Updated matrix with 1s where preference conditions are met
    """
    # For each valid combination (i,j), check if the preferences are compatible
    for i, (class_i, type_i) in enumerate(valid_combinations):
        if type_i[1] != 0:  # Skip if not a base type
            continue
        x = type_i[0]
        for j, (class_j, type_j) in enumerate(valid_combinations):
            if type_j[1] != 0:  # Skip if not a base type
                continue
            z = type_j[0]
            # Set matrix value to 1 if mutual preferences exist
            if ((x, z) in class_i and (z, x) in class_j):
                matrix[i, j] = 1
    return matrix


def create_q_matrix(valid_combinations, preference_classes):
    """
    Create and update the Q matrix for the QP problem.

    Args:
        valid_combinations: List of valid (preference_class, type) tuples
        preference_classes: List of preference classes for matrix updates

    Returns:
        Completed Q matrix for the quadratic program
    """
    size = len(valid_combinations)
    Q = np.zeros((size, size))  # Initialize Q matrix with zeros
    return update_matrix(Q, valid_combinations, preference_classes)


def create_box_constraints(x):
    """
    Create box constraints ensuring 0 ≤ x[i] ≤ 1 for all i.

    Args:
        x: cvxpy Variable

    Returns:
        List of constraints enforcing value bounds
    """
    n = x.size
    constraints = []
    for i in range(n):
        constraints.append(x[i] >= 0)
        constraints.append(x[i] <= 1)
    return constraints


def create_preference_class_constraints(preference_classes, valid_combinations, x):
    """
    Create constraints ensuring sum of x[i] = 1 for each preference class.

    Args:
        preference_classes: List of preference classes
        valid_combinations: List of (preference_class, type) tuples
        x: cvxpy Variable

    Returns:
        List of constraints ensuring probability completeness
    """
    constraints = []
    for pc in preference_classes:
        indices = [i for i, v in enumerate(valid_combinations) if v[0] == pc]
        # The sum of x[i] for these indices should equal 1
        constraints.append(sum(x[i] for i in indices) == 1)
    return constraints


def map_types_to_preference_classes(all_types, preference_classes):
    """
    Create mapping from each type to its preference classes.

    Args:
        all_types: List of all possible agent types
        preference_classes: List of preference classes

    Returns:
        Dictionary mapping each type to list of preference classes containing it
    """
    type_to_prefs = {type_i: [] for type_i in all_types}
    for pref_class in preference_classes:
        for type_i in pref_class:
            if type_i in all_types:
                type_to_prefs[type_i].append(pref_class)
    return type_to_prefs


def create_index_grid(valid_combinations):
    """
    Create mapping from (type, preference_class) to index.

    Args:
        valid_combinations: List of (preference_class, type) tuples

    Returns:
        Dictionary mapping (type, frozen_pref_class) to index
    """
    return {(agent_type, frozenset(pref_class)): i
            for i, (pref_class, agent_type) in enumerate(valid_combinations)}


def create_type_share_constraints(x, all_types, prefs_agent_type, x_grid, cond_mass, cond_type_shares):
    """
    Create constraints matching observed type shares.

    Args:
        x: cvxpy Variable
        all_types: List of all possible types
        prefs_agent_type: Dictionary mapping types to preference classes
        x_grid: Dictionary mapping (type, preference_class) to index
        cond_mass: Dictionary mapping preference classes to masses
        cond_type_shares: Dictionary mapping types to observed shares

    Returns:
        List of constraints matching type shares
    """
    constraints = []
    for agent_type in all_types:
        relevant_prefs = prefs_agent_type[agent_type]
        type_sum = sum(
            x[x_grid[(agent_type, frozenset(pref_class))]] * cond_mass[frozenset(pref_class)]
            for pref_class in relevant_prefs
        )
        constraints.append(type_sum == cond_type_shares[agent_type])
    return constraints


def simulate_stable_preferences(base_utilities, agent_chars, preference_classes, num_sims):
    """
    Simulate preference class stability with random perturbations.

    Args:
        base_utilities: Dictionary mapping types to base utility values
        agent_chars: List of possible agent characteristics
        preference_classes: List of preference classes
        num_sims: Number of simulation runs

    Returns:
        Dictionary mapping preference classes to stability frequencies
    """
    stability_counts = {frozenset(pc): 0 for pc in preference_classes}
    for _ in range(num_sims):
        sim_stable_classes = set()
        for X in agent_chars:
            simple_type = (X, 0)
            simple_utility = base_utilities[simple_type]
            stable_types = {simple_type}  # simple type is always stable
            complex_types = {t for t in base_utilities if t[0] == X and t[1] != 0}
            for complex_type in complex_types:
                epsilon = np.random.uniform(-1, 0)
                perturbed_utility = base_utilities[complex_type] + epsilon
                if perturbed_utility > simple_utility:
                    stable_types.add(complex_type)
            for pc in preference_classes:
                if set(pc) == stable_types:
                    sim_stable_classes.add(frozenset(pc))
        for pc in sim_stable_classes:
            stability_counts[pc] += 1
    return {pc: count / num_sims for pc, count in stability_counts.items()}


def make_psd(Q):
    """
    Make matrix positive semidefinite via eigenvalue clipping.

    Args:
        Q: Input symmetric matrix

    Returns:
        Modified positive semidefinite matrix
    """
    Q = (Q + Q.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.maximum(eigvals, 1e-10)
    Q_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return Q_psd


def solve_qp(all_types, utility_matrix, valid_combinations, preference_classes,
             cond_distributions, agent_characteristics, tolerance=1e-6, pack="scipy"):
    """
    Solve the quadratic programming problem.

    Args:
        all_types: List of all possible agent types
        utility_matrix: Dictionary mapping type pairs to utilities
        valid_combinations: List of valid (preference_class, type) pairs
        preference_classes: List of preference classes
        cond_distributions: Dictionary of conditional type distributions
        agent_characteristics: List of agent characteristics
        tolerance: Numerical tolerance for optimization (default: 1e-6)
        pack: Solver package to use (default: "scipy"; alternatively "cvxpy")

    Returns:
        Tuple of (solution vector, objective value)
    """
    # Step 1: Create the Q matrix
    Q = create_q_matrix(valid_combinations, preference_classes)
    n = len(valid_combinations)

    if pack == "cvxpy":
        # CVXPY approach
        alpha = cp.Variable(n)
        # Create constraints using helper functions
        box_constraints = create_box_constraints(alpha)
        pref_constraints = create_preference_class_constraints(preference_classes, valid_combinations, alpha)
        prefs_agent_type = map_types_to_preference_classes(all_types, preference_classes)
        x_grid = create_index_grid(valid_combinations)
        # Simulate to get the distribution (cond_mass)
        distribution = simulate_stable_preferences(utility_matrix, agent_characteristics, preference_classes, 10000)
        type_share_constraints = create_type_share_constraints(alpha, all_types, prefs_agent_type, x_grid, distribution,
                                                               cond_distributions)
        constraints = []
        constraints.extend(box_constraints)
        constraints.extend(pref_constraints)
        constraints.extend(type_share_constraints)
        # Make Q positive semidefinite
        Qpsd = make_psd(Q)
        # Define and solve the CVXPY problem
        objective = cp.Minimize(cp.quad_form(alpha, Qpsd))
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        if prob.status not in ["infeasible", "unbounded"]:
            print(f"Optimal value: {prob.value}")
            print("Optimal alpha values:", alpha.value)
        else:
            print(f"Problem status: {prob.status}")
        return alpha.value, result

    elif pack == "scipy":
        # SciPy approach
        def objective(x):
            return x @ Q @ x

        constraints = []
        for pc in preference_classes:
            indices = [i for i, v in enumerate(valid_combinations) if v[0] == pc]

            def make_pref_constraint(indices):
                return lambda x: sum(x[i] for i in indices) - 1

            constraints.append({
                'type': 'eq',
                'fun': make_pref_constraint(indices)
            })

        prefs_agent_type = map_types_to_preference_classes(all_types, preference_classes)
        x_grid = create_index_grid(valid_combinations)
        distribution = simulate_stable_preferences(utility_matrix, agent_characteristics, preference_classes, 10000)
        for agent_type in all_types:
            relevant_prefs = [pc for pc in preference_classes if agent_type in pc]

            def make_type_constraint(agent_type, relevant_prefs):
                return lambda x: (
                        sum(x[x_grid[(agent_type, frozenset(pref_class))]] * distribution[frozenset(pref_class)]
                            for pref_class in relevant_prefs) - cond_distributions[agent_type]
                )

            constraints.append({
                'type': 'eq',
                'fun': make_type_constraint(agent_type, relevant_prefs)
            })

        bounds = [(0, 1) for _ in range(n)]
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'ftol': 1e-9, 'maxiter': 1000})
        if result.success:
            print(f"Optimization successful! Found minimum at f(x) = {result.fun}")
        else:
            print(f"Optimization failed: {result.message}")
        return result.x, result.fun

    else:
        raise ValueError("Unsupported pack. Please choose 'cvxpy' or 'scipy'.")
