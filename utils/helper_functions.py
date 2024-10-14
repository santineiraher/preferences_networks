import pandas as pd
import config
import numpy as np


def data_construction(df, term, program_title="", total=False):
    # Filter the dataframe based on term and first_term1/first_term2 columns
    df_sub = df[(df['term_action'] == term) &
                (df['first_term1'] == term) &
                (df['first_term2'] == term)]

    # If total is False, further filter by program_title
    if not total:
        df_sub = df_sub[(df_sub['program_title1'] == program_title) &
                        (df_sub['program_title2'] == program_title)]

    return df_sub



def share_typez(df, min_num, major, term):
    # Remove rows with missing values
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
    grouped.columns = ["carnet1", "carnet2", "freq_interaction", "lowinc1", "midinc1", "highinc1", "lowinc2", "midinc2", "highinc2"]

    # Add income_1 and income_2 columns
    grouped['income_1'] = np.where(grouped['lowinc1'] == 1, 1, 0)
    grouped['income_2'] = np.where(grouped['lowinc2'] == 1, 1, 0)

    # Prepare network data
    network_data = grouped[['carnet1', 'carnet2', 'freq_interaction', 'income_1', 'income_2']]

    # Get unique IDs for carnet1 and carnet2
    unique_ids1 = grouped[['carnet1', 'income_1']].drop_duplicates()
    unique_ids2 = grouped[['carnet2', 'income_2']].drop_duplicates().rename(columns={'carnet2': 'carnet1', 'income_2': 'income_1'})
    unique_ids = pd.concat([unique_ids1, unique_ids2]).drop_duplicates()

    # Construct the best friends network
    network_data_check = grouped[['carnet1', 'carnet2', 'freq_interaction']].sort_values(by='freq_interaction', ascending=False)
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
    print(f"Columns in best_friends_df before filtering: {best_friends_df.columns}")

    # Check if 'freq_interaction' exists before filtering
    if 'freq_interaction' not in best_friends_df.columns:
        print("Error: 'freq_interaction' column is missing.")
        return pd.DataFrame()  # Return an empty DataFrame if column is missing

    # Filter out spurious relationships with freq_interaction < min_num
    best_friends_df = best_friends_df.loc[best_friends_df['freq_interaction'] > min_num]

    # Merge with unique IDs to recover income information
    best_friends = best_friends_df.merge(unique_ids1, on='carnet1', how='left')
    best_friends = best_friends.merge(unique_ids2.rename(columns={'carnet1': 'carnet2', 'income_1': 'income_2'}), on='carnet2', how='left')

    # Identify individuals who don't have any interactions
    non_excluded = pd.concat([best_friends['carnet1'], best_friends['carnet2']]).unique()
    lone = unique_ids.loc[~unique_ids['carnet1'].isin(non_excluded)].copy()
    lone['carnet2'] = pd.NA
    lone['freq_interaction'] = pd.NA
    lone['income_2'] = pd.NA

    # Duplicate the best_friends data for both directions of interaction
    best_friends2 = best_friends.rename(columns={'carnet1': 'carnet2', 'carnet2': 'carnet1', 'income_1': 'income_2', 'income_2': 'income_1'})
    best_friends_f = pd.concat([best_friends, best_friends2, lone])

    # Calculate income type proportions
    props = best_friends_f['income_1'].value_counts().reindex([0, 1], fill_value=0)
    N_B = props.get(1, 0)
    N_W = props.get(0, 0)
    N = N_B + N_W
    mu_B = N_B / N if N > 0 else 0
    mu_W = N_W / N if N > 0 else 0

    # Define interaction types
    def define_type(row):
        if pd.isna(row['freq_interaction']):
            return 1 if row['income_1'] == 1 else 4
        elif row['income_1'] == row['income_2']:
            return 2 if row['income_1'] == 1 else 6
        else:
            return 3 if row['income_1'] == 1 else 5

    best_friends_f['type'] = best_friends_f.apply(define_type, axis=1)

    # Calculate type shares and adjusted shares
    type_counts = best_friends_f['type'].value_counts().reindex(range(1, 7), fill_value=0)
    type_share = type_counts / len(best_friends_f)
    adjusted_share = type_share / [mu_B, mu_B, mu_B, mu_W, mu_W, mu_W]

    # Construct the final dataframe
    final = pd.DataFrame({
        'mu_B0': [adjusted_share[1]],
        'mu_BB': [adjusted_share[2]],
        'mu_BW': [adjusted_share[3]],
        'mu_W0': [adjusted_share[4]],
        'mu_WB': [adjusted_share[5]],
        'mu_WW': [adjusted_share[6]],
        'N_B': [N_B],
        'N_W': [N_W],
        'term': [term],
        'major': [major]
    })

    return final




def eq_D1L1(s_BB, s_BW, s_WB, s_WW, p_BB, p_BW, p_WB, p_WW, ratio_BW, ratio_WB):
    # First, compute the shares of isolates
    s_B0 = 1 - s_BB - s_BW
    s_W0 = 1 - s_WW - s_WB

    # A) Bounds on Equilibrium Type Shares

    # Lower bounds on cross-race friendships
    s_BW_lo = np.minimum(p_BW * (1 - p_BB), ratio_WB * p_WB * (1 - p_WW))
    s_WB_lo = np.minimum(p_WB * (1 - p_WW), ratio_BW * p_BW * (1 - p_BB))

    # Upper bounds on cross-race friendships
    s_BW_hi = np.minimum(p_BW, ratio_WB * p_WB)
    s_WB_hi = np.minimum(p_WB, ratio_BW * p_BW)

    # Lower bounds on isolates
    s_B0_lo = 1 - p_BB - np.minimum(p_BW * (1 - p_BB), ratio_WB * p_WB)
    s_W0_lo = 1 - p_WW - np.minimum(p_WB * (1 - p_WW), ratio_BW * p_BW)

    # Upper bounds on isolates
    s_B0_hi = 1 - p_BB - np.minimum(p_BW * (1 - p_BB), np.maximum(ratio_WB * p_WB * (1 - p_WW) - p_BB * p_BW, 0))
    s_W0_hi = 1 - p_WW - np.minimum(p_WB * (1 - p_WW), np.maximum(ratio_BW * p_BW * (1 - p_BB) - p_WW * p_WB, 0))

    # B) Implied Allocation Parameters
    a_BB = (s_BB - p_BB + p_BB * p_BW) / (p_BB * p_BW)
    a_WW = (s_WW - p_WW + p_WW * p_WB) / (p_WW * p_WB)

    # Compute the implied shares of cross-race friendships
    s_BW_max = p_BW - a_BB * p_BB * p_BW
    s_WB_max = p_WB - a_WW * p_WW * p_WB

    s_BW_eq = np.minimum(s_BW_max, ratio_WB * s_WB_max)
    s_WB_eq = np.minimum(s_WB_max, ratio_BW * s_BW_max)

    # C) Check Conditions are Satisfied

    # Allocation parameters are between 0 and 1
    sat_aBB = (-0.0001 <= a_BB) & (a_BB <= 1.0001)
    sat_aWW = (-0.0001 <= a_WW) & (a_WW <= 1.0001)

    # Implied bounds are satisfied
    sat_BW = (s_BW_lo - 0.0001 <= s_BW) & (s_BW <= s_BW_hi + 0.0001)
    sat_WB = (s_WB_lo - 0.0001 <= s_WB) & (s_WB <= s_WB_hi + 0.0001)

    sat_BW = (s_BW_eq - 0.0101 <= s_BW) & (s_BW <= s_BW_eq + 0.0101)
    sat_WB = (s_WB_eq - 0.0101 <= s_WB) & (s_WB <= s_WB_eq + 0.0101)

    sat_B0 = (s_B0_lo - 0.0001 <= s_B0) & (s_B0 <= s_B0_hi + 0.0001)
    sat_W0 = (s_W0_lo - 0.0001 <= s_W0) & (s_W0 <= s_W0_hi + 0.0001)

    # Shares of cross-race friendships are balanced
    sat_cross = (s_WB == ratio_BW * s_BW)

    # Result - is it equilibrium (TRUE/FALSE):
    equilibrium = sat_aBB & sat_aWW & sat_BW & sat_WB & sat_B0 & sat_W0 ##& sat_cross

    return equilibrium
