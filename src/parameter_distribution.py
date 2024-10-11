import numpy as np
import pandas as pd
from utils.helper_functions import eq_D1L1  # Placeholder for the eq_D1L1 function
import os
import config

class ParameterDistribution:
    def __init__(self, input_csv="Datasets_type_shares/Observed_type_shares_non_zeros.csv", output_dir="Datasets_param_dist"):
        self.non_zeros = pd.read_csv(input_csv)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_distribution(self):
        gridpts = np.arange(0, 1.01, 0.01)
        grid_shape = (len(gridpts),) * 4

        for idx, row in self.non_zeros.iterrows():
            term = row['term']
            major = row['major']
            print(f"Processing base {term} {major}")

            N_B = row['N_B']
            N_W = row['N_W']
            N = N_B + N_W

            if N_B == 0 or N_W == 0:
                print("Current composition doesn't have low-income students")
                continue  # Skip this iteration

            mu_B = N_B / N
            mu_W = N_W / N
            ratio_BW = mu_B / mu_W
            ratio_WB = mu_W / mu_B

            # Get type shares and check feasibility
            type_shares = row.iloc[:6].values
            S_grid_WB = ratio_BW * type_shares[2]

            # Initialize grids
            p_BB_G, p_BW_G, p_WB_G, p_WW_G = np.meshgrid(gridpts, gridpts, gridpts, gridpts, indexing='ij')

            # Feasible type shares grid
            Q_grid = eq_D1L1(    s_BB=type_shares[1], s_BW=type_shares[2], s_WB=type_shares[4], s_WW=type_shares[5],
            p_BB=p_BB_G, p_BW=p_BW_G, p_WB=p_WB_G, p_WW=p_WW_G,
            ratio_BW=ratio_BW, ratio_WB=ratio_WB)

            # Find equilibrium indices
            true_indices = np.argwhere(Q_grid)

            if true_indices.size > 0:
                equilibria = np.column_stack((
                    p_BB_G[true_indices[:, 0], true_indices[:, 1], true_indices[:, 2], true_indices[:, 3]],
                    p_BW_G[true_indices[:, 0], true_indices[:, 1], true_indices[:, 2], true_indices[:, 3]],
                    p_WB_G[true_indices[:, 0], true_indices[:, 1], true_indices[:, 2], true_indices[:, 3]],
                    p_WW_G[true_indices[:, 0], true_indices[:, 1], true_indices[:, 2], true_indices[:, 3]]
                ))

                # Save to CSV
                file_name = f"{self.output_dir}/paramdist_{term}_{major}.csv"
                pd.DataFrame(equilibria, columns=['p_BB_G', 'p_BW_G', 'p_WB_G', 'p_WW_G']).to_csv(file_name, index=False)
                print(f"Saved {len(equilibria)} rows to {file_name}")
            else:
                print(f"No equilibria found for {term} {major}")

if __name__ == "__main__":
    param_dist = ParameterDistribution()
    param_dist.run_distribution()
