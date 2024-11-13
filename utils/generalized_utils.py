import os
import numpy as np
import config
import json
import pandas as pd
import warnings
from scipy.optimize import minimize
from collections import defaultdict
from itertools import combinations


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


def share_typez_gen(df, min_num, major, term, folder_class_path):
    # Load the room assignment file
    room_assignment_file = os.path.join(folder_class_path, f"room_assignments_{term}_{major}.csv")

    try:
        room_df = pd.read_csv(room_assignment_file)
    except FileNotFoundError:
        warnings.warn(
            f"Warning: The room assignment file '{room_assignment_file}' does not exist. Proceeding without it.")
        return pd.DataFrame()  # early returning

    room_df['Room']=room_df['Room'].astype(pd.Int64Dtype())
    room_df['Room']+=1


    # Ensure relevant columns exist
    if 'Identifier' not in room_df.columns or 'Room' not in room_df.columns:
        raise ValueError("Room assignment file must contain 'Identifier' and 'Room' columns")

    df = df.dropna()

    # Group the data by carnet1 and carnet2 and summarize the relevant columns
    grouped = df.groupby(['carnet1', 'carnet2'], as_index=False).agg({
        'freq_interaction': 'sum',
        'lowinc1': 'max',
        'midinc1': 'max',
        'highinc1': 'max',
        'lowinc2': 'max',
        'midinc2': 'max',
        'highinc2': 'max'
    })

    # Rename columns to match the original R version
    grouped.columns = ["carnet1", "carnet2", "freq_interaction", "lowinc1", "midinc1", "highinc1", "lowinc2", "midinc2",
                       "highinc2"]

    # Add income_1 and income_2 columns
    grouped['income_1'] = np.where(grouped['lowinc1'] == 1, 1, 0)
    grouped['income_2'] = np.where(grouped['lowinc2'] == 1, 1, 0)

    # Prepare network data
    network_data = grouped[['carnet1', 'carnet2', 'freq_interaction', 'income_1', 'income_2']]

    # Get unique IDs for carnet1 and carnet2
    unique_ids1 = grouped[['carnet1', 'income_1']].drop_duplicates()
    unique_ids2 = grouped[['carnet2', 'income_2']].drop_duplicates().rename(
        columns={'carnet2': 'carnet1', 'income_2': 'income_1'})
    unique_ids = pd.concat([unique_ids1, unique_ids2]).drop_duplicates()

    # Construct the best friends network
    network_data_check = grouped[['carnet1', 'carnet2', 'freq_interaction']].sort_values(by='freq_interaction',
                                                                                         ascending=False)
    best_friends_df = pd.DataFrame()

    while not network_data_check.empty:
        best_friends_df = pd.concat([best_friends_df, network_data_check.iloc[[0]]])
        current_carnet1 = network_data_check.iloc[0]['carnet1']
        current_carnet2 = network_data_check.iloc[0]['carnet2']
        network_data_check = network_data_check[
            ~network_data_check['carnet1'].isin([current_carnet1, current_carnet2]) &
            ~network_data_check['carnet2'].isin([current_carnet1, current_carnet2])
            ]

    # Debugging: Print columns of best_friends_df before filtering


    # Check if 'freq_interaction' exists before filtering
    if 'freq_interaction' not in best_friends_df.columns:
        print("Error: 'freq_interaction' column is missing.")
        return pd.DataFrame()  # Return an empty DataFrame if column is missing

    # Filter out spurious relationships with freq_interaction < min_num
    best_friends_df = best_friends_df.loc[best_friends_df['freq_interaction'] > min_num]

    # Merge with unique IDs to recover income information
    best_friends = best_friends_df.merge(unique_ids1, on='carnet1', how='left')
    best_friends = best_friends.merge(unique_ids2.rename(columns={'carnet1': 'carnet2', 'income_1': 'income_2'}),
                                      on='carnet2', how='left')

    # Identify individuals who don't have any interactions
    non_excluded = pd.concat([best_friends['carnet1'], best_friends['carnet2']]).unique()
    lone = unique_ids.loc[~unique_ids['carnet1'].isin(non_excluded)].copy()
    lone['carnet2'] = pd.NA
    lone['freq_interaction'] = pd.NA
    lone['income_2'] = pd.NA

    # Duplicate the best_friends data for both directions of interaction
    best_friends2 = best_friends.rename(
        columns={'carnet1': 'carnet2', 'carnet2': 'carnet1', 'income_1': 'income_2', 'income_2': 'income_1'})



    best_friends_f = pd.concat([best_friends, best_friends2, lone])

    best_friends_f['carnet1'] = best_friends_f['carnet1'].astype(pd.Int64Dtype())
    best_friends_f['carnet2'] = best_friends_f['carnet2'].astype(pd.Int64Dtype())
    best_friends_f['income_1']=best_friends_f['income_1'].astype(pd.Int64Dtype())
    best_friends_f['income_2']=best_friends_f['income_2'].astype(pd.Int64Dtype())



    best_friends_f = best_friends_f.merge(room_df[['Identifier', 'Room']], left_on='carnet1', right_on='Identifier', how='left').rename(
        columns={'Room': 'room1'})
    best_friends_f = best_friends_f.merge(room_df[['Identifier', 'Room']], left_on='carnet2', right_on='Identifier', how='left').rename(
        columns={'Room': 'room2'})

    best_friends_f=best_friends_f.drop(columns=['Identifier_x','Identifier_y'])
    best_friends_f['room1'] = best_friends_f['room1'].astype(pd.Int64Dtype())
    best_friends_f['room2'] = best_friends_f['room2'].astype(pd.Int64Dtype())

    total_counts = defaultdict(int)
    # Calculate conditional distributions
    cond_distributions = {}
    for _, row in best_friends_f.iterrows():
        income1, income2 = row['income_1'], row['income_2']
        room1, room2 = row['room1'], row['room2']

        # Convert income and room to 'A'/'B' notation
        income_label1 = 'B' if income1 == 1 else 'A'

        # Handle the case where income2 or room2 is NaN (indicating the individual is alone)
        if pd.isna(income2) or pd.isna(room2):
            # The individual is alone
            cond_distributions[((income_label1, room1), 0)] = cond_distributions.get(((income_label1, room1), 0), 0) + 1
            total_counts[(income_label1, room1)] += 1
        else:
            # Convert income and room to 'A'/'B' notation for the second individual
            income_label2 = 'B' if income2 == 1 else 'A'
            # Record the interaction
            cond_distributions[((income_label1, room1), (income_label2, room2))] = cond_distributions.get(
                ((income_label1, room1), (income_label2, room2)), 0) + 1
            total_counts[(income_label1, room1)] += 1
            total_counts[(income_label2, room2)] += 1

    # Normalize distributions to get proportions
    total_pairs = sum(cond_distributions.values())
    if total_pairs > 0:
        cond_distributions = {k: v / total_pairs for k, v in cond_distributions.items()}


    # Calculate the sum for each group (based on the first part of the key)
    group_sums = defaultdict(float)
    for (first_key, second_key), value in cond_distributions.items():
        group_sums[first_key] += value

    # Normalize values within each group
    cond_distributions = {
        (first_key, second_key): (value / group_sums[first_key]) if group_sums[first_key] > 0 else 0
        for (first_key, second_key), value in cond_distributions.items()
    }

    # Corrected N_values calculation for all combinations of income_label and room


    total_individuals = sum(total_counts.values())
    # Calculate the overall sum to normalize the values into proportions
    overall_total = total_individuals if total_individuals > 0 else 1  # Avoid division by zero
    N_values = {
        f"N_{income_label}_{room}": (total_counts[(income_label, room)] / overall_total)
        for income_label in ['A', 'B']
        for room in [1, 2]  # Room levels now include 1 and 2 as specified
    }



    # Calculate total unique individuals
    unique_carnet1 = best_friends_f['carnet1'].dropna().unique()
    unique_carnet2 = best_friends_f['carnet2'].dropna().unique()
    total_individuals = len(set(unique_carnet1).union(set(unique_carnet2)))

    # Prepare the result DataFrame dynamically based on cond_distributions keys
    cond_distribution_columns = {f"mu_{key}": [value] for key, value in cond_distributions.items()}
    # Combine all data into the final DataFrame
    result_df = pd.DataFrame({
        **cond_distribution_columns,  # Conditional distributions as columns
        **N_values,  # Counts of income and room combinations
        'total_individuals': [total_individuals],  # Total unique individuals
        'term': [term],
        'major': [major]
    })

    return result_df

