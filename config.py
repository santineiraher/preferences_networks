# Configuration file for all constants used across scripts
import numpy as np


# Paths for data inputs
## C:/Users/Santiago Neira/Dropbox/Torniquetes_TRT/Data/dyads.freqs_3segs.csv
##"C:/Users/neira santiago/Dropbox/Torniquetes_TRT/Data/dyads.freqs_3segs.csv"
DATAFRAME_CSV_PATH = "C:/Users/Santiago Neira/Dropbox/Torniquetes_TRT/Data/dyads.freqs_3segs.csv"
OBSERVED_TYPE_SHARES_FILE = "data/Datasets_type_shares/Observed_type_shares_non_zeros.csv"

# Paths for results inside the "data" folder in the project
NETWORK_FOLDER_PATH = "../../data/Datasets/Networks_semesters_majors"
TYPE_SHARES_FOLDER_PATH="../../data/Datasets/Type_shares"
PARAM_DIST_FOLDER_PATH = "../../data/Datasets/Param_dist"
COUNTERFACTUALS_FOLDER_PATH = "../../data/Datasets/Counterfactuals"
PREF_ANALYSIS_OUTPUT_DIR = "../../data/Results/Parameter_sets"
FACTUAL_ANALYSIS_DIR = "../../data/Results/Factuals"

RESULTS_DIR="../data/Results"

# Translation of major names
TRANSLATIONS = {
    "39: Literature": ["Literatura", "Literatura"],
    "1: Business": ["Administración", "Administración"],
    "8: Law": ["Derecho", "Derecho"],
    "15: Physics": ["Física", "Física"],
    "20: Computer Science": ["Ing Sistemas", "IngSistemas"],
    "28: Chemical eng.": ["Ing Química", "IngQuímica"],
    "26: Industrial eng.": ["Ing Industrial", "IngIndustrial"],
    "27: Mechanical eng.": ["Ing Mecánica", "IngMecánica"],
    "6: Political Science": ["Ciencia Política", "Cpol"],
    "4: Arts": ["Artes", "Artes"],
    "14: Philosophy": ["Filosofía", "Filosofía"],
    "24: Electronic eng.": ["Ing Electrónica", "IngElectrónica"],
    "2: Anthropology": ["Antropología", "Antropología"],
    "44: Psychology": ["Psicología", "Psicología"],
    "40: Mathematics": ["Matemáticas", "Matemáticas"],
    "43: Music": ["Música", "Música"],
    "11: Economics": ["Economía", "Economía"],
    "41: Medicine": ["Medicina", "Medicina"],
    "18: History": ["Historia", "Historia"],
    "9: Design": ["Diseño", "Diseño"],
    "3: Architecture": ["Arquitectura", "Arquitectura"],
    "45: Chemistry": ["Química", "Química"],
    "5: Biology": ["Biología", "Biología"],
    "29: Civil eng.": ["Ing Civil", "IngCivil"],
    "23: Electric eng.": ["Ing Eléctrica", "IngEléctrica"],
    "21: Environmental eng.": ["Ing Ambiental", "IngAmbiental"],
    "16: Geociences": ["Geociencias", "Geociencias"],
    "22: Biomedical eng.": ["Ing Biomédica", "IngBiomédica"],
    "19: Art History": ["Historia del Arte", "HistArte"],
    "17: Government and Policy": ["Gobierno y Política", "GobPolítica"],
    "42: Microbiology": ["Microbiología", "Microbiología"]
}


# Grid points
GRID_POINTS = list(np.linspace(0, 1, 26))

# Other constants
SEED = 123
SEMESTERS=[201610,201620,201710,201720,201810,201820]


agent_characteristics = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
