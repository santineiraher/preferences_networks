import os
import pandas as pd
import numpy as np
from utils.helper_functions_simple import share_typez  # Assuming you already have the share_typez function
import config

class NetworkConstruction:
    def \
            __init__(self):
        self.folder_path = config.NETWORK_FOLDER_PATH
        self.output_dir = config.TYPE_SHARES_FOLDER_PATH
        os.makedirs(self.output_dir, exist_ok=True)
        self.initial_df = pd.DataFrame(columns=[
            "mu_B0", "mu_BB", "mu_BW", "mu_W0", "mu_WB", "mu_WW", "N_B", "N_W", "term", "major"
        ])

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

            # Call the share_typez function
            aux = share_typez(df, 1, major, term)

            # Convert the returned dict to DataFrame without wrapping it in another list
            aux_df = pd.DataFrame(aux, columns=self.initial_df.columns)

            # Append to the initial DataFrame
            self.initial_df = pd.concat([self.initial_df, aux_df], ignore_index=True)
            print(f"Processed: {file}")

        # Save the results to CSV
        self.save_results()

    def save_results(self):
        """Save the results to CSV and Excel."""
        observed_type_shares_path = os.path.join(self.output_dir, "Observed_type_shares.csv")
        non_zeros_path = os.path.join(self.output_dir, "Observed_type_shares_non_zeros.csv")
        excel_path = os.path.join(self.output_dir, "Observed_type_shares_non_zeros.xlsx")

        # Save the CSV files
        self.initial_df=self.initial_df.fillna(0)
        self.initial_df.to_csv(observed_type_shares_path, index=False)
        self.initial_df.to_csv(non_zeros_path, index=False)

        # Save the Excel file
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            self.initial_df.to_excel(writer, index=False)

if __name__ == "__main__":
    network_constructor = NetworkConstruction()
    network_constructor.run_construction()
