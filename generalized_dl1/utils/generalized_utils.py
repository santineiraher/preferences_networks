import os
import numpy as np
import generalized_dl1.config as config
import json


def ensure_output_dir_exists():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def export_q_matrix_and_vars(matrix, assignment_parameters):
    ensure_output_dir_exists()

    # Save Q matrix
    matrix_file = os.path.join(config.RESULTS_DIR, "q_matrix.npy")
    np.save(matrix_file, matrix)

    # Save assignment parameters as a dictionary (in JSON format)
    vars_file = os.path.join(config.RESULTS_DIR, "assignment_parameters.json")
    with open(vars_file, "w") as f:
        json.dump(assignment_parameters, f, indent=4)

    print(f"Q matrix saved to: {matrix_file}")
    print(f"Assignment parameters (as dictionary) saved to: {vars_file}")
