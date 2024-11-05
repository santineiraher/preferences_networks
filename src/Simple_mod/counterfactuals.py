import os
import re
import numpy as np
import pandas as pd
from utils.helper_functions_simple import eq_D1L1  # Use the already refactored eq_D1L1 function
import config  # Import configuration settings


# Helper function to trim filenames
def trim_up_to_paramdist(x):
    return re.sub(r'.*paramdist', '', x)


# Helper function to extract term
def extract_between_first_second_underscore(string):
    parts = string.split("_")
    if len(parts) >= 3:
        return parts[1]
    return None


# Helper function to extract major
def extract_between_second_underscore_and_first_dot(string):
    parts = string.split("_")
    if len(parts) >= 3:
        return parts[2].split(".")[0]
    return None


class Counterfactuals:
    def __init__(self, folder_path=config.PARAM_DIST_FOLDER_PATH, output_dir=config.COUNTERFACTUALS_FOLDER_PATH):
        self.non_zeros = pd.read_csv(config.TYPE_SHARES_FOLDER_PATH+"/Observed_type_shares_non_zeros.csv")
        self.folder_path = folder_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_counterfactuals(self):
        np.random.seed(42)
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        gridpts = np.linspace(0, 1, 26)

        for file in files:
            params = pd.read_csv(file)
            filei = trim_up_to_paramdist(file)
            print(f"Processing file: {filei}")

            term = extract_between_first_second_underscore(filei)
            term_i = int(term) if term else None
            major_i = extract_between_second_underscore_and_first_dot(filei)

            row = self.non_zeros[(self.non_zeros['term'] == term_i) & (self.non_zeros['major'] == major_i)]
            if row.empty:
                print(f"No matching rows for term {term_i} and major {major_i}")
                continue

            N_B = row.iloc[0]['N_B']
            N_W = row.iloc[0]['N_W']
            N = N_B + N_W
            N_B_list = ["No counterfactual", "+50%", "+100%", "+150%"]

            # Initialize grid
            P_grid_BB, P_grid_BW, P_grid_WB, P_grid_WW = np.meshgrid(gridpts, gridpts, gridpts, gridpts, indexing='ij')

            initial = pd.DataFrame(columns=[
                'sBB', 'sBW', 'sWB', 'sWW', 'pBB', 'pBW', 'pWB', 'pWW', 'counterfactual', 'N_B', 'N_W'
            ])

            sample_size = len(params) if len(params) < 350 else max(350, int(0.1 * len(params)))
            sampled_indices = np.random.choice(len(params), size=sample_size, replace=False)

            for j in range(4):
                N_Bi = int(N_B * (0.5 + 0.5 * (j + 1)))
                N_Wi = N - N_Bi
                if N_Bi >= N:
                    print("Counterfactual value exceeds total number of individuals")
                    continue

                mu_B = N_Bi / N
                mu_W = N_Wi / N
                ratio_BW = mu_B / mu_W
                ratio_WB = mu_W / mu_B
                S_grid_WB = ratio_BW * P_grid_BW

                print(f"Processing counterfactual: {N_B_list[j]} with N_B: {N_Bi} and ratio BW: {ratio_BW}")

                for i in sampled_indices:
                    Q_grid_aux = eq_D1L1(
                        P_grid_BB, P_grid_BW, S_grid_WB, P_grid_WW,
                        params.iloc[i, 0], params.iloc[i, 1], params.iloc[i, 2], params.iloc[i, 3],ratio_BW,ratio_WB
                    )

                    if np.sum(Q_grid_aux) == 0:
                        continue

                    eq_index = np.argwhere(Q_grid_aux)

                    aux = pd.DataFrame({
                        'sBB': P_grid_BB[eq_index[:, 0], eq_index[:, 1], eq_index[:, 2], eq_index[:, 3]],
                        'sBW': P_grid_BW[eq_index[:, 0], eq_index[:, 1], eq_index[:, 2], eq_index[:, 3]],
                        'sWB': S_grid_WB[eq_index[:, 0], eq_index[:, 1], eq_index[:, 2], eq_index[:, 3]],
                        'sWW': P_grid_WW[eq_index[:, 0], eq_index[:, 1], eq_index[:, 2], eq_index[:, 3]],
                        'pBB': params.iloc[i, 0],
                        'pBW': params.iloc[i, 1],
                        'pWB': params.iloc[i, 2],
                        'pWW': params.iloc[i, 3],
                        'counterfactual': N_B_list[j],
                        'N_B': N_Bi,
                        'N_W': N_Wi
                    })

                    # Add to main data frame
                    initial = pd.concat([initial, aux], ignore_index=True).drop_duplicates()

            # Save the results
            file_name = f"{self.output_dir}/counter_{term}_{major_i}.csv"
            initial.to_csv(file_name, index=False)
            print(f"Results saved to {file_name}")


if __name__ == "__main__":
    counterfactuals = Counterfactuals()
    counterfactuals.run_counterfactuals()
