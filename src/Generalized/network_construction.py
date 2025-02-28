import os
import pandas as pd
import numpy as np
from utils.generalized_utils import share_typez_gen
import config


class NetworkConstructionGeneralized:
    def __init__(self):
        self.folder_path = config.NETWORK_FOLDER_PATH_GEN
        self.folder_class_path = config.ASSIGNMENT_PATH
        self.output_dir = config.TYPE_SHARES_FOLDER_PATH_GEN
        os.makedirs(self.output_dir, exist_ok=True)

        # Set income labels and room levels
        income_labels = ['A', 'B']
        room_levels = [1, 2]

        # Generate column names for mu_ based on income and room combinations
        mu_columns = []
        for income_label1 in income_labels:
            for room1 in room_levels:
                # Column for being alone
                mu_columns.append(f"mu_{((income_label1, room1), 0)}")
                for income_label2 in income_labels:
                    for room2 in room_levels:
                        # Column for pair interactions
                        mu_columns.append(f"mu_{((income_label1, room1), (income_label2, room2))}")

        # Generate column names for N_ based on income and room combinations
        N_columns = [f"N_{income_label}_{room}" for income_label in income_labels for room in room_levels]

        # Generate column names for raw N_ values
        N_raw_columns = [f"N_{income_label}_{room}_raw" for income_label in income_labels for room in room_levels]

        # Combine all columns into initial_df
        self.initial_df = pd.DataFrame(
            columns=mu_columns + N_columns + N_raw_columns + ['total_individuals', "term", "major"])

    def extract_first_6_numeric(self, filename):
        """Extract the first 6-digit number from the filename as a string."""
        numeric_part = ''.join([char for char in filename if char.isdigit()])
        return numeric_part[:6] if numeric_part else None

    def extract_major(self, filename):
        """Extract major name as a string between the fourth underscore and '.csv' in filename."""
        parts = filename.split('_')
        if len(parts) > 4:
            return str(parts[4].split('.')[0])
        return None

    def run_construction(self):
        """Main method to process all files and construct the networks."""
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]

        for file in files:
            print(f"Processing file: {file}")
            df = pd.read_csv(file)
            term = self.extract_first_6_numeric(file)
            major = self.extract_major(file)

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
            if (term == 201620 or term == 201820) and major == 'Historia':
                print("Skipping due to errors")
                continue

            # Call the share_typez function
            aux = share_typez_gen(df, 1, major, term, self.folder_class_path)

            # Convert the returned dict to DataFrame
            aux_df = pd.DataFrame(aux, columns=self.initial_df.columns)

            # Append to the initial DataFrame
            self.initial_df = pd.concat([self.initial_df, aux_df], ignore_index=True)
            print(f"Processed: {file}")

        # Save the results to CSV
        self.save_results()

    def save_results(self):
        """Save the results to CSV and Excel."""
        observed_type_shares_path = os.path.join(self.output_dir, "Observed_type_shares_generalized.csv")
        non_zeros_path = os.path.join(self.output_dir, "Observed_type_shares_non_zeros_generalized.csv")
        excel_path = os.path.join(self.output_dir, "Observed_type_shares_non_zeros_generalized.xlsx")

        # Save the CSV files
        self.initial_df = self.initial_df.fillna(0)
        self.initial_df.to_csv(observed_type_shares_path, index=False)
        self.initial_df.to_csv(non_zeros_path, index=False)

        # Save the Excel file
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            self.initial_df.to_excel(writer, index=False)


if __name__ == "__main__":
    network_constructor = NetworkConstructionGeneralized()
    network_constructor.run_construction()