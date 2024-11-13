import os
import pandas as pd
import numpy as np
import config
import csv
from itertools import combinations

class Exposure:
    def __init__(self):
        self.folder_path = config.NETWORK_FOLDER_PATH_GEN
        self.df_exposures = pd.read_csv(config.DATAFRAME_EXPOSURES_PATH, low_memory=False,encoding='latin1',on_bad_lines='skip',sep=";",dtype={'CARNET':str})
        self.output_dir = config.EXPOSURES_PATH
        os.makedirs(self.output_dir, exist_ok=True)


    def extract_first_6_numeric(self, filename):
        """Extract the first 6-digit number from the filename as a string."""
        numeric_part = ''.join([char for char in filename if char.isdigit()])
        return numeric_part[:6] if numeric_part else None

    def extract_major(self, filename):
        """Extract major name as a string between the fourth underscore and '.csv' in filename."""
        parts = filename.split('_')
        if len(parts) > 4:
            return str(parts[4].split('.')[0])  # Return as string
        return None

    def run_construction(self):
        """Main method to process all files and construct the networks."""
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]

        for file in files:
            print(f"Processing file: {file}")
            df = pd.read_csv(file)
            term = self.extract_first_6_numeric(file)  # Extract term as string first
            major = self.extract_major(file)  # Extract major as string

            # Ensure term is treated as a numeric type if possible
            try:
                term = int(term)
            except (ValueError, TypeError):
                print(f"Error converting term to integer for file {file}")
                continue

            # Check if major is valid and non-empty
            if not major or major == "nan":
                print(f"Invalid major extracted for file {file}")
                continue

            print(f"Term: {term}, Major: {major}")
            ##WE NEED TO EXCLUDE THE TOTAL FILE ... OTHERWISE THIS BECOMES UNBEARABLE
            if major=='total':
                continue
            ##Lets subset the dataframe
            df_exp = self.df_exposures[(self.df_exposures['PERIODO']==term)]

            print("Initial len of file is ", len(df))
            df = df.dropna()
            print("After dropping NaNs, len of file is ", len(df))

            unique_ids=list(set(df['carnet1']).union(set(df['carnet2'])))
            unique_ids=[str(int(i)) for i in unique_ids]
            df_exp=df_exp[df_exp['CARNET'].isin(unique_ids)]

            # Calculate total credits per student
            credits_per_student = df_exp.groupby('CARNET')['CREDITOS'].sum().to_dict()

            # Initialize an empty DataFrame to store exposure values
            exposure_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids, dtype=float)

            # Iterate over each pair of unique IDs
            for id1, id2 in combinations(unique_ids, 2):
                # Filter classes for each student
                classes_id1 = df_exp[df_exp['CARNET'] == id1][['COD_MAT_SECC', 'CREDITOS']].dropna()
                classes_id2 = df_exp[df_exp['CARNET'] == id2][['COD_MAT_SECC', 'CREDITOS']].dropna()

                # Find common classes based on 'COD_MAT_SECC'
                common_classes = pd.merge(classes_id1, classes_id2, on='COD_MAT_SECC', suffixes=('_id1', '_id2'))

                # Calculate total shared credits for common classes
                shared_credits = common_classes['CREDITOS_id1'].sum()  # or 'CREDITOS_id2' should be the same
                min_total_credits = min(credits_per_student.get(id1, 0), credits_per_student.get(id2, 0))
                if min_total_credits > 0:
                    exposure_score = shared_credits / min_total_credits
                else:
                    exposure_score = 0.0

                # Update the matrix
                exposure_matrix.at[id1, id2] = exposure_score
                exposure_matrix.at[id2, id1] = exposure_score  # Make it symmetric

            self.save_results(exposure_matrix, term, major)

    def save_results(self, exposure_matrix, term, major):
        """Save the exposure matrix to a CSV file with a specific naming format."""
        filename = f"exposure_{term}_{major}.csv"
        file_path = os.path.join(self.output_dir, filename)

        # Save the exposure matrix to CSV
        exposure_matrix.to_csv(file_path, index=True)
        print(f"Exposure matrix saved to {file_path}")



if __name__=="__main__":
    assignments = Exposure()
    assignments.run_construction()




