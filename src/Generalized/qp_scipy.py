from scipy.optimize import minimize
import numpy as np


def solve_qp_with_scipy(Q, valid_combinations, preference_classes, all_types, x_grid, distribution, cond_distributions):
    """
    Solve QP using scipy's minimize with SLSQP method
    """
    n = len(valid_combinations)

    # Objective function
    def objective(x):
        return x @ Q @ x

    # Constraints
    constraints = []

    # 1. Preference class sum constraints
    for pc in preference_classes:
        indices = [i for i, v in enumerate(valid_combinations) if v[0] == pc]

        def make_pref_constraint(indices):
            return lambda x: sum(x[i] for i in indices) - 1

        constraints.append({
            'type': 'eq',
            'fun': make_pref_constraint(indices)
        })

    # 2. Type share constraints
    for agent_type in all_types:
        relevant_prefs = [pc for pc in preference_classes if agent_type in pc]

        def make_type_constraint(agent_type, relevant_prefs):
            return lambda x: (
                    sum(x[x_grid[(agent_type, frozenset(pref_class))]] *
                        distribution[frozenset(pref_class)]
                        for pref_class in relevant_prefs) -
                    cond_distributions[agent_type]
            )

        constraints.append({
            'type': 'eq',
            'fun': make_type_constraint(agent_type, relevant_prefs)
        })

    # Box constraints
    bounds = [(0, 1) for _ in range(n)]

    # Initial guess (uniform)
    x0 = np.ones(n) / n

    # Solve
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    if result.success:
        print(f"Optimization successful!")
        print(f"Found minimum at f(x) = {result.fun}")
        print(f"Number of iterations: {result.nit}")
    else:
        print(f"Optimization failed: {result.message}")

    return result.x, result.fun


# Use it:
solution, objective_value = solve_qp_with_scipy(
    Q,
    valid_combinations,
    preference_classes,
    all_types,
    x_grid,
    distribution,
    cond_distributions
)

print("\nSolution:")
print("x =", solution)
print("Objective value =", objective_value)