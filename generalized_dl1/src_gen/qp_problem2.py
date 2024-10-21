import cvxpy as cp
import numpy as np
import config

def update_matrix(matrix, valid_combinations, preference_classes):
    """
    Update the Q matrix based on valid combinations and preference classes.
    """
    agent_characteristics = config.agent_characteristics
    for i, (class_i, type_i) in enumerate(valid_combinations):
        if len(type_i) < 2 or type_i[1] != 'NA':
            continue
        x = type_i[0]

        for j, (class_j, type_j) in enumerate(valid_combinations):
            if len(type_j) < 2 or type_j[1] != 'NA':
                continue
            z = type_j[0]

            if z not in [ac[0] for ac in agent_characteristics]:
                continue

            if ((x, z) in preference_classes[class_i - 1] and
                (z, x) in preference_classes[class_j - 1]):
                matrix[i, j] = 1  # Set matrix value to 1 if preference conditions are met

def create_q_matrix(valid_combinations, preference_classes):
    """
    Create and update the Q matrix for the QP problem based on valid combinations and preference classes.
    """
    size = len(valid_combinations)
    Q = np.zeros((size, size))  # Initialize Q matrix with zeros

    # Update the matrix using the provided logic
    update_matrix(Q, valid_combinations, preference_classes)

    return Q

def construct_probabilities(preference_classes, param_function):
    """
    Construct the probability vector P based on the preference classes and utility function f(x_i, x_l).
    param_function: A single function that takes (x_i, x_l) and returns utility values from the utility matrix.
    """
    P = []
    for (x_i, S) in preference_classes.keys():
        prob = 1
        for x_l in S:
            prob *= param_function(x_i, x_l)  # f(x_i, x_l)
        for x_k in set(config.agent_characteristics) - set(S):
            prob *= (1 - param_function(x_i, x_k))  # 1 - f(x_i, x_k)
        P.append(prob)
    return np.array(P)

def construct_type_shares(typeshares, valid_combinations):
    """
    Construct the adjusted type-share vector \\tilde \Pi based on the type-shares and valid combinations.
    """
    Pi = np.zeros(len(valid_combinations))
    for i, (class_i, _) in enumerate(valid_combinations):
        Pi[i] = typeshares.get(class_i, 0)
    return Pi

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

    # Step 2: Define the assignment parameters variable (alpha)
    alpha = cp.Variable(len(valid_combinations))  # Vector for assignment parameters

    # Step 3: Construct the vectors P (probabilities) and \tilde \Pi (adjusted type-shares)
    P = construct_probabilities(preference_classes, param_function)
    Pi = construct_type_shares(typeshares, valid_combinations)

    # Step 4: Reshape alpha into the matrix A
    # A should have dimensions (len(preference_classes), len(valid_combinations))
    # Reshape alpha into the matrix A to match the structure
    A = cp.reshape(alpha, (len(preference_classes), len(valid_combinations)))

    # Step 5: Define the objective function
    objective = cp.Minimize((1/2) * cp.quad_form(alpha, Q))

    # Step 6: Set up constraints
    constraints = []

    # Constraint 1: Sum of assignment parameters for each agent class must be close to 1 (with tolerance)
    for class_i in range(1, len(valid_combinations) + 1):
        relevant_indices = [i for i, (ci, _) in enumerate(valid_combinations) if ci == class_i]
        constraints.append(cp.abs(cp.sum(alpha[relevant_indices]) - 1) <= tolerance)

    # Constraint 2: Stability condition (alpha values must be zero for invalid combinations, with tolerance)
    for i, (class_i, type_i) in enumerate(valid_combinations):
        if len(type_i) < 2 or type_i[1] != 'NA':
            constraints.append(cp.abs(alpha[i]) <= tolerance)

    # Constraint 3: A^T P = \tilde \Pi
    constraints.append(cp.abs(A.T @ P - Pi) <= tolerance)

    # Constraint 4: Completeness condition: A \times 1 = 1
    constraints.append(cp.sum(A, axis=1) == 1)

    # Step 7: Solve the QP problem
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # Return the optimized values of alpha
    return alpha.value, result
