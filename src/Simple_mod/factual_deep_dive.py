import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils.helper_functions_simple import calculate_fractions, calculate_links,calculate_links_dataframe
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
            if term != 201610:
                continue
            if major not in ["Economía", "Medicina"]:
                continue

            zeroes = self.non_zeros[(self.non_zeros['term'] == term) & (self.non_zeros['major'] == major)]

            if zeroes.empty:
                print(f"No matching rows for term {term} and major {major}")
                continue

            cross_actual = calculate_links_dataframe(zeroes, zeroes['N_B'].iloc[0], zeroes['N_W'].iloc[0])

            if not cross_actual.empty and 'NBB' in cross_actual.columns and 'total' in cross_actual.columns:
                cross_real = cross_actual['NBB'].iloc[0]
                total_real = cross_actual['total'].iloc[0]
            else:
                print(f"Error: Required columns are missing or cross_actual is empty for term {term} and major {major}")
                cross_real = np.nan
                total_real = np.nan

            print(f"Cross-real: {cross_real}, Total-real: {total_real}")

            # Process each counterfactual
            initial = pd.DataFrame(columns=['counterfactual', 'NBB', 'total'])

            for counter in N_B_list:
                if counter not in params['counterfactual'].values:
                    print(f"Counterfactual {counter} not found in data")
                    continue

                sub_param = params[params['counterfactual'] == counter]
                N_B = sub_param['N_B'].iloc[0]
                N_A = sub_param['N_A'].iloc[0]
                sub_param = calculate_links(sub_param, N_B, N_A)

                if 'NBB' in sub_param.columns and 'total' in sub_param.columns:
                    to_append = sub_param[['counterfactual', 'NBB', 'total']]
                    initial = pd.concat([initial, to_append], ignore_index=True)
                else:
                    print(f"Warning: Required columns are missing for {counter} in term {term}, major {major}")

            # Ensure all counterfactuals are present by appending missing ones
            for counter in N_B_list:
                if counter not in initial['counterfactual'].values:
                    initial = pd.concat(
                        [initial, pd.DataFrame({'counterfactual': [counter], 'NBB': [np.nan], 'total': [np.nan]})],
                        ignore_index=True)

            # Create two subplots in one figure
            self.create_double_plot(initial, cross_real, total_real, term, major, color_palette, N_B_list)

    def create_double_plot(self, data, cross_real, total_real, term, major, color_palette, N_B_list):
        """Creates two KDE plots in a single figure - one for NBB and one for total."""

        # Filter out NaN values from the DataFrame before plotting
        data_clean_nbb = data.dropna(subset=['NBB'])
        data_clean_total = data.dropna(subset=['total'])

        if data_clean_nbb.empty and data_clean_total.empty:
            print(f"No valid data to plot for term {term} and major {major}")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: NBB (cross-links)
        if not data_clean_nbb.empty:
            sns.kdeplot(data=data_clean_nbb, x='NBB', hue='counterfactual', fill=True,
                        common_norm=False, palette=color_palette, alpha=0.5,
                        bw_adjust=3, ax=ax1)

            # Add vertical line for observed value
            if not np.isnan(cross_real):
                ax1.axvline(x=cross_real, color='red', linestyle='--')
                ax1.annotate('Observed Number of cross-links',
                             xy=(cross_real, 0),
                             xytext=(cross_real + 0.5, ax1.get_ylim()[1] * 0.7),
                             color='red')

            ax1.set_xlim(left=0)  # Ensure x-axis starts from 0
            ax1.set_title(f"Observed Number of cross-links\n{major} in semester {term}")
            ax1.set_xlabel('NBB (cross-links)')
            ax1.set_ylabel('Density')

        # Plot 2: Total links
        if not data_clean_total.empty:
            sns.kdeplot(data=data_clean_total, x='total', hue='counterfactual', fill=True,
                        common_norm=False, palette=color_palette, alpha=0.5,
                        bw_adjust=3, ax=ax2)

            # Add vertical line for observed value
            if not np.isnan(total_real):
                ax2.axvline(x=total_real, color='red', linestyle='--')
                ax2.annotate('Observed Number of total links',
                             xy=(total_real, 0),
                             xytext=(total_real + 0.5, ax2.get_ylim()[1] * 0.7),
                             color='red')

            ax2.set_xlim(left=0)  # Ensure x-axis starts from 0
            ax2.set_title(f"Observed Number of total links\n{major} in semester {term}")
            ax2.set_xlabel('Total links')
            ax2.set_ylabel('Density')

        # Fix legend issues if needed
        for ax in [ax1, ax2]:
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                print("No legend was automatically generated, adding manually.")
                for i, label in enumerate(N_B_list):
                    ax.plot([], [], color=color_palette[i], label=label)

            ax.legend(loc='upper right', title='Counterfactual Values')

        plt.tight_layout()
        plot_filename = f"{self.output_dir}/double_plot_{term}_{major}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Double plot saved to {plot_filename}")


if __name__ == "__main__":
    factual_analysis = FactualAnalysis()
    factual_analysis.run_analysis()