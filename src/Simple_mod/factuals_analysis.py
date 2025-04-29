import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils.helper_functions_simple import calculate_fractions, calculate_links, calculate_links_dataframe
import config
from matplotlib.patches import Patch


def extract_term_major(filename):
    term_match = re.search(r'counter_(\d{6})_', filename)
    term = term_match.group(1) if term_match else None

    major_match = re.search(r'counter_\d{6}_(.+)\.csv$', filename)
    major = major_match.group(1) if major_match else None

    return term, major


class FactualAnalysis:
    def __init__(self, folder_path=config.COUNTERFACTUALS_FOLDER_PATH, output_dir=config.FACTUAL_ANALYSIS_DIR):
        self.non_zeros = pd.read_csv(config.TYPE_SHARES_FOLDER_PATH + "/Observed_type_shares_non_zeros.csv")
        self.folder_path = folder_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Store results for CSV and Excel
        self.results = pd.DataFrame(columns=[
            "Major", "semester", "mean_no counterfactual", "mean_50%", "mean_100%", "mean_150%",
            "diff_50%", "pval_50%", "tag_50%", "stderr_diff_50%",
            "diff_100%", "pval_100%", "tag_100%", "stderr_diff_100%",
            "diff_150%", "pval_150%", "tag_150%", "stderr_diff_150%"
        ])

    def run_analysis(self):
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]

        color_palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
        N_B_list = ["No counterfactual", "+50%", "+100%", "+150%"]

        for file in files:
            params = pd.read_csv(file)
            params.columns = ["sBB", "sBW", "sWB", "sWW", "pBB", "pBW", "pWB", "pWW", "counterfactual", "N_B", "N_A"]
            term, major = extract_term_major(file)
            term = int(term) if term else None
            print(f"Processing term {term} and major {major}")

            zeroes = self.non_zeros[(self.non_zeros['term'] == term) & (self.non_zeros['major'] == major)]

            if zeroes.empty:
                print(f"No matching rows for term {term} and major {major}")
                continue

            cross_actual = calculate_links_dataframe(zeroes, zeroes['N_B'].iloc[0], zeroes['N_W'].iloc[0])

            if not cross_actual.empty and 'cross' in cross_actual.columns:
                cross_real = cross_actual['cross'].iloc[0]
            else:
                print(f"Error: 'cross' column is missing or cross_actual is empty for term {term} and major {major}")
                cross_real = np.nan

            print(f"Cross-real: {cross_real}")

            # Process each counterfactual
            initial = pd.DataFrame(columns=['counterfactual', 'cross'])
            means = {}

            for counter in N_B_list:
                if counter not in params['counterfactual'].values:
                    print(f"Counterfactual {counter} not found in data")
                    means[counter] = np.nan
                    continue

                sub_param = params[params['counterfactual'] == counter]
                N_B = sub_param['N_B'].iloc[0]
                N_A = sub_param['N_A'].iloc[0]
                sub_param = calculate_fractions(sub_param, N_B, N_A)

                if 'cross' in sub_param.columns:
                    to_append = sub_param[['counterfactual', 'cross']]
                    initial = pd.concat([initial, to_append], ignore_index=True)
                    means[counter] = sub_param['cross'].mean()
                else:
                    print(f"Warning: 'cross' column is missing for {counter} in term {term}, major {major}")
                    means[counter] = np.nan

            # Ensure all counterfactuals are present by appending missing ones
            for counter in N_B_list:
                if counter not in initial['counterfactual'].values:
                    initial = pd.concat([initial, pd.DataFrame({'counterfactual': [counter], 'cross': [np.nan]})],
                                        ignore_index=True)

            # Filter out NaN values from the DataFrame before plotting
            initial_clean = initial.dropna(subset=['cross'])

            if not initial_clean.empty:
                plt.figure(figsize=(10, 6))

                # Plot the kdeplot only if valid data exists
                sns.kdeplot(data=initial_clean, x='cross', hue='counterfactual', fill=True, common_norm=False,
                            palette=color_palette, alpha=0.5, bw_adjust=3)

                # Add a vertical line for the observed cross value
                plt.axvline(x=cross_real, color='red', linestyle='--')
                plt.annotate('Observed Cross', xy=(cross_real, 0), xytext=(cross_real + 0.02, 0.02), color='red')

                # Ensure the x-axis starts from 0 (since values are positive)
                plt.xlim(left=0)

                # Add a legend manually if Seaborn fails to generate one
                handles, labels = plt.gca().get_legend_handles_labels()
                if not handles:
                    print("No legend was automatically generated, adding manually.")
                    for i, label in enumerate(initial_clean['counterfactual'].unique()):
                        plt.plot([], [], color=color_palette[i], label=label)

                plt.legend(loc='upper right', title='Counterfactual Values')
                plt.title(f"Densities for {major} in semester {term}")
                plt.xlabel('Cross')
                plt.ylabel('Density')
                plt.tight_layout()
                plot_filename = f"{self.output_dir}/kdensity_{term}_{major}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"Results saved to {plot_filename}")
            else:
                print(f"No valid data to plot for term {term} and major {major}")

            # Perform t-tests against "No counterfactual"
            ttest_results = {}
            for counter in N_B_list[1:]:
                if not pd.isna(means[counter]) and not pd.isna(means["No counterfactual"]):
                    group1 = initial_clean[initial_clean['counterfactual'] == counter]['cross']
                    group2 = initial_clean[initial_clean['counterfactual'] == "No counterfactual"]['cross']

                    # Calculate t-statistic and p-value
                    t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit', equal_var=False)

                    # Calculate standard error of difference
                    # Formula: sqrt(s₁²/n₁ + s₂²/n₂) where s is std dev and n is sample size
                    var1 = group1.var(ddof=1)  # Sample variance for group 1
                    var2 = group2.var(ddof=1)  # Sample variance for group 2
                    n1 = len(group1)
                    n2 = len(group2)

                    stderr_diff = np.sqrt((var1 / n1) + (var2 / n2)) if n1 > 0 and n2 > 0 else np.nan

                    ttest_results[counter] = {
                        "diff": means[counter] - means["No counterfactual"],
                        "pval": p_val,
                        "tag": "Significant" if p_val < 0.05 else "Not significant",
                        "stderr_diff": stderr_diff
                    }
                else:
                    ttest_results[counter] = {
                        "diff": np.nan,
                        "pval": np.nan,
                        "tag": "NA",
                        "stderr_diff": np.nan
                    }

            # Append the results to the DataFrame
            self.results = pd.concat([self.results, pd.DataFrame({
                "Major": [major],
                "semester": [term],
                "mean_no counterfactual": [means["No counterfactual"]],
                "mean_50%": [means["+50%"]],
                "mean_100%": [means["+100%"]],
                "mean_150%": [means["+150%"]],
                "diff_50%": [ttest_results["+50%"]["diff"]],
                "pval_50%": [ttest_results["+50%"]["pval"]],
                "tag_50%": [ttest_results["+50%"]["tag"]],
                "stderr_diff_50%": [ttest_results["+50%"]["stderr_diff"]],
                "diff_100%": [ttest_results["+100%"]["diff"]],
                "pval_100%": [ttest_results["+100%"]["pval"]],
                "tag_100%": [ttest_results["+100%"]["tag"]],
                "stderr_diff_100%": [ttest_results["+100%"]["stderr_diff"]],
                "diff_150%": [ttest_results["+150%"]["diff"]],
                "pval_150%": [ttest_results["+150%"]["pval"]],
                "tag_150%": [ttest_results["+150%"]["tag"]],
                "stderr_diff_150%": [ttest_results["+150%"]["stderr_diff"]]
            })], ignore_index=True)

        # Save results to CSV and Excel files
        csv_filename = os.path.join(self.output_dir, "analysis_results.csv")
        excel_filename = os.path.join(self.output_dir, "analysis_results.xlsx")
        self.results.to_csv(csv_filename, index=False)
        self.results.to_excel(excel_filename, index=False)
        print(f"Results saved to {csv_filename} and {excel_filename}")


if __name__ == "__main__":
    factual_analysis = FactualAnalysis()
    factual_analysis.run_analysis()