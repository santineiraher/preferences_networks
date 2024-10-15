import os
import pandas as pd
import config  # Import the configuration file
from utils.helper_functions import data_construction


class DataframeConstruction:
    def __init__(self):
        # Use low_memory=False to avoid dtype warning during CSV read
        self.df_3seqs = pd.read_csv(config.DATAFRAME_CSV_PATH, low_memory=False)
        self.output_dir = config.NETWORK_FOLDER_PATH
        os.makedirs(self.output_dir, exist_ok=True)

    def run_construction(self):
        semesters = config.SEMESTERS
        filtered_program_title1 = self.get_unique_programs()

        # Process each semester
        for semester in semesters:
            # Process each unique program title
            for idx, program_title in enumerate(filtered_program_title1):
                # Get the translated name (from the third name in the translation dictionary)
                translated_name = self.get_translated_name(program_title)

                if translated_name:
                    # Construct dataframe for the specific program title and save it
                    local_df = data_construction(self.df_3seqs, semester, program_title)
                    file_name = f"{self.output_dir}/net_{semester}_{translated_name}.csv"
                    local_df.to_csv(file_name, index=False)
                    print(f"Exported: {file_name}")
                else:
                    print(f"No translation found for: {program_title}")

            # Construct a total dataframe for the semester (without filtering by program title)
            total_df = data_construction(self.df_3seqs, semester, "",total=True)
            total_file_name = f"{self.output_dir}/net_{semester}_total.csv"
            total_df.to_csv(total_file_name, index=False)
            print(f"Exported total dataframe for semester {semester}: {total_file_name}")

    def get_unique_programs(self):
        # Convert program_title1 to strings to handle mixed types (float/None)
        self.df_3seqs['program_title1'] = self.df_3seqs['program_title1'].astype(str)
        unique_program_title1 = self.df_3seqs['program_title1'].unique()

        # Filtering to exclude unwanted program titles
        exclude_terms = [
            "educ\\.", "Children", "Language and culture", "Int. accounting",
            "Directed studies", "Exchange - gral. studies",
            "Language and sociocultural studies", "General eng"
        ]

        filtered_program_title1 = [
            title for title in unique_program_title1
            if not any(term in title for term in exclude_terms)
        ]
        filtered_program_title1 = [title for title in filtered_program_title1 if pd.notna(title)]

        return filtered_program_title1

    def get_translated_name(self, program_title):
        """Retrieve the translated name from the config.py.TRANSLATIONS."""
        # Search for the translation in config.py.TRANSLATIONS
        if program_title in config.TRANSLATIONS:
            return config.TRANSLATIONS[program_title][1]  # Get the file-friendly translation (second element)
        return None


# If the script is executed directly, run the construction process
if __name__ == "__main__":
    df_constructor = DataframeConstruction()
    df_constructor.run_construction()
