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
        Uses green for "Factual" and blue for "Balancing of classrooms" categories.

        Args:
            results_df: DataFrame containing results
            output_file: Path to save the plot (optional)
            benchmark_value: Value to use for the benchmark line (default=1)
        """
        # Set the style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        # Define custom palette based on categories
        hue_column = None

        if 'counterfactual' in results_df.columns:
            hue_column = 'counterfactual'
        elif 'source_file' in results_df.columns:
            hue_column = 'source_file'
            print("Warning: 'counterfactual' column not found. Using source_file for hue.")

        if hue_column:
            # Get unique categories
            categories = results_df[hue_column].unique()

            # Create custom color palette with all categories
            # Get default seaborn color palette with enough colors
            default_colors = sns.color_palette(n_colors=len(categories))

            # Create complete palette dictionary
            palette = {}
            for i, cat in enumerate(categories):
                if "Factual" in str(cat):
                    palette[cat] = "green"
                elif "Balancing of classrooms" in str(cat):
                    palette[cat] = "dodgerblue"  # Lighter blue color
                else:
                    # Assign a color from the default palette
                    palette[cat] = default_colors[i]

            # Create KDE plot with custom palette
            ax = sns.kdeplot(
                data=results_df,
                x='cross_linkedness',
                hue=hue_column,
                fill=True,
                common_norm=False,
                alpha=0.7,  # Increased alpha for better visibility
                linewidth=2,
                palette=palette  # Apply custom palette
            )
        else:
            # No hue column available
            ax = sns.kdeplot(
                data=results_df,
                x='cross_linkedness',
                fill=True,
                alpha=0.6,
                linewidth=2
            )
            print("Warning: Neither 'counterfactual' nor 'source_file' column found. No hue applied.")

        # Set plot labels and title
        plt.title('Kernel Density Estimation of Cross-Linkedness Values', fontsize=16)
        plt.xlabel('Cross-Linkedness Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)

        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add vertical line at benchmark value
        plt.axvline(x=benchmark_value, color='red', linestyle='--', alpha=0.7, label='Random Benchmark')

        # Add legend with title (ensure it's visible)
        if hue_column:
            plt.legend(title=hue_column.capitalize(), loc='best', frameon=True, framealpha=1.0)
            # Make sure legend handles and labels are correct
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, title=hue_column.capitalize(), loc='best', frameon=True)

        # Save the plot if output file is specified
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"KDE plot saved to {output_file}")

        # Show the plot
        plt.tight_layout()
        plt.show()

    if __name__ == "__main__":

        type_shares = config.TYPE_SHARES_FOLDER_PATH_GEN
        results_path = config.RESULTS_DIR_GEN

        csv_path = os.path.join(type_shares, "Observed_type_shares_non_zeros_generalized.csv")
        file_pattern = os.path.join(results_path, "*economics*_2.csv")
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
        economics_cs = 0.21818181818181812
        medicine_cs = 0.32627765064836
        output_file = os.path.join(results_path, "kdensity_economics_2.png")

        plot_cross_linkedness_kde(combined_df, output_file=output_file, benchmark_value=medicine_cs)

if __name__ == "__main__":

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
        Uses green for "Factual" and blue for "Balancing of classrooms" categories.

        Args:
            results_df: DataFrame containing results
            output_file: Path to save the plot (optional)
            benchmark_value: Value to use for the benchmark line (default=1)
        """
        # Set the style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        # Define custom palette based on categories
        hue_column = None

        if 'counterfactual' in results_df.columns:
            hue_column = 'counterfactual'
        elif 'source_file' in results_df.columns:
            hue_column = 'source_file'
            print("Warning: 'counterfactual' column not found. Using source_file for hue.")

        if hue_column:
            # Get unique categories
            categories = results_df[hue_column].unique()

            # Create custom color palette with all categories
            # Get default seaborn color palette with enough colors
            default_colors = sns.color_palette(n_colors=len(categories))

            # Create complete palette dictionary
            palette = {}
            for i, cat in enumerate(categories):
                if "Factual" in str(cat):
                    palette[cat] = "green"
                elif "Balancing of classrooms" in str(cat):
                    palette[cat] = "dodgerblue"  # Lighter blue color
                else:
                    # Assign a color from the default palette
                    palette[cat] = default_colors[i]

            # Create KDE plot with custom palette
            ax = sns.kdeplot(
                data=results_df,
                x='cross_linkedness',
                hue=hue_column,
                fill=True,
                common_norm=False,
                alpha=0.7,  # Increased alpha for better visibility
                linewidth=2,
                palette=palette  # Apply custom palette
            )
        else:
            # No hue column available
            ax = sns.kdeplot(
                data=results_df,
                x='cross_linkedness',
                fill=True,
                alpha=0.6,
                linewidth=2
            )
            print("Warning: Neither 'counterfactual' nor 'source_file' column found. No hue applied.")

        # Set plot labels and title
        plt.title('Kernel Density Estimation of Cross-Linkedness Values', fontsize=16)
        plt.xlabel('Cross-Linkedness Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)

        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add benchmark line with label (will appear in legend)
        benchmark_line = plt.axvline(x=benchmark_value, color='red', linestyle='--', alpha=0.7,
                                     label='Random Benchmark')

        # No need for separate legend code - handled in the KDE sections

        # Save the plot if output file is specified
        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"KDE plot saved to {output_file}")

        # Show the plot
        plt.tight_layout()
        plt.show()


    if __name__ == "__main__":

        type_shares = config.TYPE_SHARES_FOLDER_PATH_GEN
        results_path = config.RESULTS_DIR_GEN

        csv_path = os.path.join(type_shares, "Observed_type_shares_non_zeros_generalized.csv")
        file_pattern = os.path.join(results_path, "*economics*_3.csv")
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
        economics_cs = 0.21818181818181812
        medicine_cs = 0.32627765064836
        output_file = os.path.join(results_path, "kdensity_economics_3.png")

        plot_cross_linkedness_kde(combined_df, output_file=output_file, benchmark_value=economics_cs)