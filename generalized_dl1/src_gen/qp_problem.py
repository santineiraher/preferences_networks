from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, minimize, value
import numpy as np
from generalized_dl1.utils.generalized_utils import ensure_output_dir_exists
import generalized_dl1.config as config
import os
import json
import pandas as pd

def load_q_matrix():
    """Load the Q matrix from file."""
    matrix_file = f"{config.RESULTS_DIR}/q_matrix.npy"
    return np.load(matrix_file)


def load_assignment_parameters():
    """Load the assignment parameters from JSON file and restructure them."""
    vars_file = os.path.join(config.RESULTS_DIR, "assignment_parameters.json")

    with open(vars_file, "r") as f:
        assignment_parameters = json.load(f)

    # Restructure the dictionary using the convention: "alpha", "preference_class", "type"
    structured_parameters = []
    for alpha, (preference_class, type_info) in assignment_parameters.items():
        structured_parameters.append({
            "alpha": alpha,
            "preference_class": preference_class,
            "type": type_info
        })
        # Convert the structured list to a pandas DataFrame
    df = pd.DataFrame(structured_parameters)

        # Define the CSV file path
    csv_file = os.path.join(config.RESULTS_DIR, "assignment_parameters.csv")

        # Export the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

    print(f"Assignment parameters saved as CSV to: {csv_file}")

    return structured_parameters

def create_qp_problem():
    # Load the Q matrix
    Q = load_q_matrix()
    n = Q.shape[0]  # Number of assignment parameters (decision variables)

    # Create a Pyomo concrete model
    model = ConcreteModel()

    # Define the decision variables (alpha_0, alpha_1, ..., alpha_n-1)
    model.alpha = Var(range(n), domain=NonNegativeReals, bounds=(0, 1))  # Non-negative real values between 0 and 1

    # Define the objective function: 1/2 * alpha^T Q alpha
    def objective_rule(model):
        return 0.5 * sum(model.alpha[i] * Q[i, j] * model.alpha[j] for i in range(n) for j in range(n))

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Load assignment parameters and create a dictionary of class to type mappings
    structured_parameters = load_assignment_parameters()

    # Create a dictionary to group alphas by class
    class_to_alphas = {}
    for i, param in enumerate(structured_parameters):
        class_num = param["preference_class"]
        if class_num not in class_to_alphas:
            class_to_alphas[class_num] = []
        class_to_alphas[class_num].append(i)  # Add index of alpha corresponding to this class

    # Add class-based summation constraints (sum of alphas in each class must be 1)
    def class_summation_constraint(model, class_num):
        return sum(model.alpha[i] for i in class_to_alphas[class_num]) == 1

    # Add 36 constraints, one for each class (from 1 to 36)
    model.class_constraints = Constraint(range(1, 37), rule=class_summation_constraint)

    return model


def display_objective_equation(model, Q):
    """Display the symbolic equation for the objective function, showing only non-null terms."""
    print("\nObjective Function (Symbolic Equation):")

    expr = model.objective.expr
    non_zero_terms = []

    # Loop over the terms of the expression and filter out zero terms
    for i in range(len(model.alpha)):
        for j in range(len(model.alpha)):
            coef = model.alpha[i] * Q[i, j] * model.alpha[j]
            if value(Q[i, j]) != 0:
                non_zero_terms.append(f"0.5 * (alpha_{i} * {Q[i, j]} * alpha_{j})")

    # Join and display only non-zero terms
    if non_zero_terms:
        print(" + ".join(non_zero_terms))
    else:
        print("The objective function contains only zero terms.")


def display_quadratic_terms(model, Q):
    """Display only the quadratic terms of the objective function where Q[i, i] is non-zero."""
    print("\nQuadratic Terms (Diagonal Elements of Q):")

    non_zero_terms = []

    # Loop over the diagonal elements of Q (i.e., where i == j)
    for i in range(len(model.alpha)):
        coef = model.alpha[i] * Q[i, i] * model.alpha[i]
        if value(Q[i, i]) != 0:
            non_zero_terms.append(f"0.5 * (alpha_{i} * {Q[i, i]} * alpha_{i})")

    # Join and display only non-zero quadratic terms
    if non_zero_terms:
        print(" + ".join(non_zero_terms))
    else:
        print("There are no non-zero quadratic terms.")

def solve_qp_problem():
    # Load Q matrix
    Q = load_q_matrix()

    # Create the QP model
    model = create_qp_problem()

    # Display the quadratic terms in the objective function
    display_quadratic_terms(model, Q)

    # Use the SCIP solver
    solver = SolverFactory('scip')  # Use SCIP as the solver

    # Solve the problem
    results = solver.solve(model, tee=True)

    # Extract the optimized alpha values
    optimal_alphas = [value(model.alpha[i]) for i in range(len(model.alpha))]

    # Print the optimal solution
    print("\nOptimal Alpha Values:")
    for i, alpha_value in enumerate(optimal_alphas):
        print(f"alpha_{i}: {alpha_value}")



if __name__ == "__main__":
    ensure_output_dir_exists()  # Ensure output directory exists
    solve_qp_problem()  # Solve the QP problem
