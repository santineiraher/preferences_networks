import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    parts = string.split("_")
    return parts[2] if len(parts) > 2 else None

class PreferenceAnalysis:
    def __init__(self, folder_path="Datasets_param_dist", output_dir="Results/Parameter_sets"):
        self.folder_path = folder_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_analysis(self):
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

            carrera = translations.get(major_i, ["Unknown", "Unknown"])
            print(f"Processing major: {carrera[0]}")

            # Add diff_W and diff_B columns
            equilibria = params.copy()
            equilibria.columns = ["pBB", "pBW", "pWB", "pWW"]
            equilibria["diff_W"] = equilibria["pWW"] - equilibria["pWB"]
            equilibria["diff_B"] = equilibria["pBB"] - equilibria["pBW"]

            # Prepare plot
            image_name = os.path.join(self.output_dir, f"Pref_params_{term_i}_{carrera[1]}.png")
            gridpts = np.linspace(0, 1, 26)

            # Plotting
            plt.figure(figsize=(10, 10))
            plt.subplots_adjust(hspace=0.4, wspace=0.4)

            # Scatter plot 1
            plt.subplot(2, 3, 1)
            plt.scatter(equilibria["pBB"], equilibria["pBW"], c='blue', marker='o')
            plt.xlim([gridpts.min(), gridpts.max()])
            plt.ylim([gridpts.min(), gridpts.max()])
            plt.xlabel("f_BB")
            plt.ylabel("f_BA")
            plt.title("pBB vs pBW")

            # Scatter plot 2
            plt.subplot(2, 3, 2)
            plt.scatter(equilibria["pWW"], equilibria["pWB"], c='green', marker='o')
            plt.xlim([gridpts.min(), gridpts.max()])
            plt.ylim([gridpts.min(), gridpts.max()])
            plt.xlabel("f_AA")
            plt.ylabel("f_AB")
            plt.title("pWW vs pWB")

            # Histogram of diff_B
            plt.subplot(2, 3, 3)
            plt.hist(equilibria["diff_B"], bins=30, color='blue')
            plt.title("Histogram of diff_B")
            plt.xlabel("f_BB - f_BA")
            plt.ylabel("Count")

            # Scatter plot 3
            plt.subplot(2, 3, 4)
            plt.scatter(equilibria["pBW"], equilibria["pWB"], c='red', marker='o')
            plt.xlim([gridpts.min(), gridpts.max()])
            plt.ylim([gridpts.min(), gridpts.max()])
            plt.xlabel("f_BA")
            plt.ylabel("f_AB")
            plt.title("pBW vs pWB")

            # Scatter plot 4
            plt.subplot(2, 3, 5)
            plt.scatter(equilibria["pBB"], equilibria["pWW"], c='purple', marker='o')
            plt.xlim([gridpts.min(), gridpts.max()])
            plt.ylim([gridpts.min(), gridpts.max()])
            plt.xlabel("f_BB")
            plt.ylabel("f_AA")
            plt.title("pBB vs pWW")

            # Histogram of diff_W
            plt.subplot(2, 3, 6)
            plt.hist(equilibria["diff_W"], bins=30, color='red')
            plt.title("Histogram of diff_W")
            plt.xlabel("f_AA - f_AB")
            plt.ylabel("Count")

            # Save the plot
            plt.savefig(image_name)
            plt.close()
            print(f"Saved plot to {image_name}")

if __name__ == "__main__":
    pref_analysis = PreferenceAnalysis()
    pref_analysis.run_analysis()
