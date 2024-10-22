import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import seaborn as sns

# Dictionary for translating majors
translations = {
    "Architecture": ["Arquitectura", "Arquitectura"],
    "Biology": ["Biología", "Biología"],
    "Biomedical_eng": ["Ing Biomédica", "IngBiomédica"],
    "Business": ["Administración", "Administración"],
    "Chemical_eng": ["Ing Química", "IngQuímica"],
    "Chemistry": ["Química", "Química"],
    "Civil_eng": ["Ing Civil", "IngCivil"],
    "Computer_Science": ["Ing Sistemas", "IngSistemas"],
    "Economics": ["Economía", "Economía"],
    "Electric_eng": ["Ing Eléctrica", "IngEléctrica"],
    "Electronic_eng": ["Ing Electrónica", "IngElectrónica"],
    "Environmental_eng": ["Ing Ambiental", "IngAmbiental"],
    "Geociences": ["Geociencias", "Geociencias"],
    "Government_and_Policy": ["Gobierno", "Gobierno"],
    "Industrial_eng": ["Ing Industrial", "IngIndustrial"],
    "Law": ["Derecho", "Derecho"],
    "Literature": ["Literatura", "Literatura"],
    "Mathematics": ["Matemáticas", "Matemáticas"],
    "Mechanical_eng": ["Ing Mecánica", "IngMecánica"],
    "Medicine": ["Medicina", "Medicina"],
    "Microbiology": ["Microbiología", "Microbiología"],
    "Music": ["Música", "Música"],
    "Physics": ["Física", "Física"],
    "Anthropology": ["Antropología", "Antropología"],
    "Art_History": ["Historia del Arte", "HistArte"],
    "Phylosophy": ["Filosofía", "Filosofía"],
    "Political_Science": ["Ciencia Política", "Cpol"],
    "Psychology": ["Psicología", "Psicología"],
    "Diseño": ["Diseño", "Diseño"],
    "total": ["total", "total"]
}

# Helper function to trim filenames
def trim_up_to_paramdist(x):
    return re.sub(r'.*paramdist', '', x)

# Helper function to extract term
def extract_first_6_numeric(x):
    match = re.search(r"\d{6}", x)
    return match.group(0) if match else None

# Helper function to extract major
def extract_major(string):
    # Remove the file extension first
    base_name = string.rsplit(".", 1)[0]
    # Split by underscores and extract the third part
    parts = base_name.split("_")
    return parts[2] if len(parts) > 2 else None

class PreferenceAnalysis:
    def __init__(self, folder_path=config.PARAM_DIST_FOLDER_PATH, output_dir=config.PREF_ANALYSIS_OUTPUT_DIR):
        self.folder_path = folder_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_analysis(self):
        sns.set(style="whitegrid")  # Set Seaborn style for a cleaner look
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]

        for file in files:
            params = pd.read_csv(file)
            if params.empty:
                print(f"Current database is empty for file: {file}")
                continue

            filei = trim_up_to_paramdist(file)
            print(f"Processing file: {filei}")

            term = extract_first_6_numeric(filei)
            term_i = int(term) if term else None
            major_i = extract_major(filei)

            print(f"Processing major: {major_i} in term {term_i}")

            # Add diff_W and diff_B columns
            equilibria = params.copy()
            equilibria.columns = ["pBB", "pBW", "pWB", "pWW"]
            equilibria["diff_W"] = equilibria["pWW"] - equilibria["pWB"]
            equilibria["diff_B"] = equilibria["pBB"] - equilibria["pBW"]

            # Prepare plot
            image_name = os.path.join(self.output_dir, f"Pref_params_{term_i}_{major_i}.png")
            gridpts = np.linspace(0, 1, 26)

            # Create figure with six subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            plt.subplots_adjust(hspace=0.4, wspace=0.4)

            # Scatter plot 1: pBB vs pBW
            sns.scatterplot(ax=axes[0, 0], x=equilibria["pBB"], y=equilibria["pBW"], color='black')
            axes[0, 0].set(xlim=(gridpts.min(), gridpts.max()), ylim=(gridpts.min(), gridpts.max()))
            axes[0, 0].set_title("pBB vs pBW")
            axes[0, 0].set_xlabel("f_BB")
            axes[0, 0].set_ylabel("f_BA")

            # Scatter plot 2: pWW vs pWB
            sns.scatterplot(ax=axes[0, 1], x=equilibria["pWW"], y=equilibria["pWB"], color='black')
            axes[0, 1].set(xlim=(gridpts.min(), gridpts.max()), ylim=(gridpts.min(), gridpts.max()))
            axes[0, 1].set_title("pWW vs pWB")
            axes[0, 1].set_xlabel("f_AA")
            axes[0, 1].set_ylabel("f_AB")

            # Histogram of diff_B (in terms of probability)
            sns.histplot(ax=axes[0, 2], data=equilibria, x="diff_B", bins=30, color='blue', kde=True, stat="density")
            axes[0, 2].set_title("Histogram of diff_B")
            axes[0, 2].set_xlabel("f_BB - f_BA")
            axes[0, 2].set_ylabel("Probability")

            # Scatter plot 3: pBW vs pWB
            sns.scatterplot(ax=axes[1, 0], x=equilibria["pBW"], y=equilibria["pWB"], color='black')
            axes[1, 0].set(xlim=(gridpts.min(), gridpts.max()), ylim=(gridpts.min(), gridpts.max()))
            axes[1, 0].set_title("pBW vs pWB")
            axes[1, 0].set_xlabel("f_BA")
            axes[1, 0].set_ylabel("f_AB")

            # Scatter plot 4: pBB vs pWW
            sns.scatterplot(ax=axes[1, 1], x=equilibria["pBB"], y=equilibria["pWW"], color='black')
            axes[1, 1].set(xlim=(gridpts.min(), gridpts.max()), ylim=(gridpts.min(), gridpts.max()))
            axes[1, 1].set_title("pBB vs pWW")
            axes[1, 1].set_xlabel("f_BB")
            axes[1, 1].set_ylabel("f_AA")

            # Histogram of diff_W (in terms of probability)
            sns.histplot(ax=axes[1, 2], data=equilibria, x="diff_W", bins=30, color='red', kde=True, stat="density")
            axes[1, 2].set_title("Histogram of diff_W")
            axes[1, 2].set_xlabel("f_AA - f_AB")
            axes[1, 2].set_ylabel("Probability")

            # Save the plot
            plt.savefig(image_name)
            plt.close()
            print(f"Saved plot to {image_name}")

if __name__ == "__main__":
    pref_analysis = PreferenceAnalysis()
    pref_analysis.run_analysis()
