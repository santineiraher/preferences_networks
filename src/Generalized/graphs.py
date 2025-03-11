import os
import glob
from xxsubtype import bench

import pandas as pd
import numpy as np
import ast

import config
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cross_linkedness_kde(results_df, output_file=None, benchmark_value=1):
    """
    Create a kernel density plot of cross-linkedness values,
    with hue based on 'counterfactual' column.

    Args:
        results_df: DataFrame containing results
        output_file: Path to save the plot (optional)
    """
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Create KDE plot with 'counterfactual' as hue
    if 'counterfactual' in results_df.columns:
        ax = sns.kdeplot(
            data=results_df,
            x='cross_linkedness',
            hue='counterfactual',
            fill=True,
            common_norm=False,
            alpha=0.6,
            linewidth=2
        )
    else:
        # If counterfactual column doesn't exist, add a default value
        print("Warning: 'counterfactual' column not found. Using source_file for hue.")
        if 'source_file' in results_df.columns:
            # Use file name as hue
            ax = sns.kdeplot(
                data=results_df,
                x='cross_linkedness',
                hue='source_file',
                fill=True,
                common_norm=False,
                alpha=0.6,
                linewidth=2
            )
        else:
            # No hue
            ax = sns.kdeplot(
                data=results_df,
                x='cross_linkedness',
                fill=True,
                alpha=0.6,
                linewidth=2
            )

    # Set plot labels and title
    plt.title('Kernel Density Estimation of Cross-Linkedness Values', fontsize=16)
    plt.xlabel('Cross-Linkedness Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add vertical line at x=1 (baseline for normalized cross-linkedness)
    plt.axvline(x=benchmark_value, color='red', linestyle='--', alpha=0.7, label='Random Benchmark')

    # Save the plot if output file is specified
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"KDE plot saved to {output_file}")

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    type_shares =config.TYPE_SHARES_FOLDER_PATH_GEN
    results_path = config.RESULTS_DIR_GEN

    csv_path=os.path.join(type_shares, "Observed_type_shares_non_zeros_generalized.csv")
    file_pattern = os.path.join(results_path, "*_medicine_*")
    files = glob.glob(file_pattern)
    # Process each file
    filtered_dfs = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}")

        # Read the file
        df = pd.read_csv(file_path)
        filtered_dfs.append(df)
        print(f"  Selected {len(df)} rows from {file_name}")


    # Combine all filtered dataframes
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    economics_cs= 0.21818181818181812
    medicine_cs =0.32627765064836
    output_file = os.path.join(results_path,"kdensity_medicine.png")

    plot_cross_linkedness_kde(combined_df, output_file=output_file, benchmark_value=medicine_cs)
