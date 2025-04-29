import os
import glob
import pandas as pd
import numpy as np
import ast
import config

np.random.seed(20)

def load_and_filter_files(directory_path, result_threshold=1e-3,counter_word= "*_factual.csv"):
    """
    Load and filter all _factual.csv files in the given directory.

    Args:
        directory_path: Path to the directory containing result files
        result_threshold: Maximum result value to include (lower is better)

    Returns:
        DataFrame with combined filtered results
    """
    # Find all files ending with _factual.csv in the directory
    file_pattern = os.path.join(directory_path, counter_word)
    files = glob.glob(file_pattern)

    if not files:
        print(f"No _factual.csv files found in {directory_path}")
        return pd.DataFrame()

    print(f"Found {len(files)} factual result files")

    # Process each file
    filtered_dfs = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}")

        # Read the file
        df = pd.read_csv(file_path)

        # Filter by result threshold
        df = df[df['result'] < result_threshold]

        if df.empty:
            print(f"  No results found with result < {result_threshold} in {file_name}")
            continue

        # Sample one row per temperature
        df = df.groupby('temperature', group_keys=False).apply(
            lambda x: x.sample(min(2, len(x)))
        )

        # Append to filtered dataframes
        filtered_dfs.append(df)
        print(f"  Selected {len(df)} rows from {file_name}")

    if not filtered_dfs:
        print("No data found meeting the criteria")
        return pd.DataFrame()

    # Combine all filtered dataframes
    combined_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f"Combined dataframe has {len(combined_df)} rows")

    return combined_df


def calculate_cross_linkedness(row):
    """
    Calculate cross-linkedness for a single row of data.

    Cross-linkedness = (Number of individuals in mixed relationships /
                       Total number of individuals in any relationship) /
                       ((N_A_1 + N_A_2) * (N_B_1 + N_B_2) / comb(N, 2))

    Args:
        row: A row from the dataframe

    Returns:
        Calculated cross-linkedness value
    """
    # Get population parameters
    N_A_1 = row['N_A_1']
    N_A_2 = row['N_A_2']
    N_B_1 = row['N_B_1']
    N_B_2 = row['N_B_2']
    N = row['N']  # Total population

    # Dictionary to map first component to population count
    pop_counts = {
        ('A', 1): N_A_1,
        ('A', 2): N_A_2,
        ('B', 1): N_B_1,
        ('B', 2): N_B_2
    }

    # Mixed relationship combinations
    mixed_keys = [
        "(('A', 1), ('B', 1))",
        "(('A', 1), ('B', 2))",
        "(('A', 2), ('B', 1))",
        "(('A', 2), ('B', 2))"
    ]

    # All non-alone relationship combinations (within-group and mixed)
    all_relation_keys = [
                            "(('A', 1), ('A', 1))",
                            "(('A', 1), ('A', 2))",
                            "(('A', 2), ('A', 1))",
                            "(('A', 2), ('A', 2))",
                            "(('B', 1), ('B', 1))",
                            "(('B', 1), ('B', 2))",
                            "(('B', 2), ('B', 1))",
                            "(('B', 2), ('B', 2))"
                        ] + mixed_keys

    # Calculate individuals in mixed relationships
    mixed_individuals = 0
    for key in mixed_keys:
        if key in row:
            # Parse the tuple from the string representation
            tuple_key = ast.literal_eval(key)
            first_component = tuple_key[0]
            # Multiply by population count of first component
            individuals = row[key] * pop_counts[first_component]
            mixed_individuals += individuals

    # Calculate total individuals in any relationship (not alone)
    total_individuals = 0
    for key in all_relation_keys:
        if key in row:
            # Parse the tuple from the string representation
            tuple_key = ast.literal_eval(key)
            first_component = tuple_key[0]
            # Multiply by population count of first component
            individuals = row[key] * pop_counts[first_component]
            total_individuals += individuals

    # Calculate baseline cross-linkedness
    if total_individuals > 0:
        base_cross_linkedness = mixed_individuals / total_individuals
    else:
        base_cross_linkedness = 0

    # Calculate normalization factor: (N_A_1 + N_A_2) * (N_B_1 + N_B_2) / comb(N, 2)
    N_A = N_A_1 + N_A_2
    N_B = N_B_1 + N_B_2

    # Binomial coefficient calculation: comb(N, 2) = N * (N-1) / 2
    comb_N_2 = N * (N - 1) / 2

    # Calculate normalization factor
    if comb_N_2 > 0:
        normalization_factor = (N_A * N_B) / comb_N_2
    else:
        normalization_factor = 1.0  # Default to 1 if undefined

    # Calculate normalized cross-linkedness
    normalized_cross_linkedness = base_cross_linkedness / normalization_factor

    return normalized_cross_linkedness



def analyze_results(directory_path, result_threshold=1e-3, output_file=None,counter_word="*_factual.csv",counter_tag="factual"):
    """
    Analyze results from factual files and calculate cross-linkedness.

    Args:
        directory_path: Path to the directory containing result files
        result_threshold: Maximum result value to include
        output_file: Path to save the output CSV (optional)

    Returns:
        DataFrame with analysis results
    """
    # Load and filter data
    combined_df = load_and_filter_files(directory_path, result_threshold,counter_word)

    if combined_df.empty:
        return combined_df

    # Calculate cross-linkedness for each row
    combined_df['cross_linkedness'] = combined_df.apply(calculate_cross_linkedness, axis=1)
    combined_df['counterfactual'] = counter_tag

    # Save to output file if specified
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"Analysis results saved to {output_file}")

    return combined_df


if __name__ == "__main__":

    economics_path = config.GENERAL_COUNTERFACTUAL_PATH_3
    medicine_path = config.MEDICINE_COUNTERFACTUAL_PATH_3

    results_path= config.RESULTS_DIR_GEN
    os.makedirs(results_path, exist_ok=True)

    # Set output file path
    output_file = os.path.join(results_path, "cross_linkedness_analysis_economics_counter_equit_3.csv")

    # Run the analysis
    results = analyze_results(
        directory_path=economics_path,
        result_threshold=1e-3,
        output_file=output_file,
        counter_word="*_equit.csv",
        counter_tag="Incremented proportion of Low - Income"
    )

    # Print overall statistics
    if not results.empty:
        print("\nOverall cross-linkedness statistics:")
        print(f"Mean: {results['cross_linkedness'].mean():.4f}")
        print(f"Median: {results['cross_linkedness'].median():.4f}")
        print(f"Min: {results['cross_linkedness'].min():.4f}")
        print(f"Max: {results['cross_linkedness'].max():.4f}")
        print(f"Std Dev: {results['cross_linkedness'].std():.4f}")


