# main.py

# Import the necessary modules
from src.dataframe_construction import DataframeConstruction
from src.network_construction import NetworkConstruction
from src.parameter_distribution import ParameterDistribution
import config  # Import config file if we need to use any constants
from src.counterfactuals import Counterfactuals  # Import the new class

from src.pref_analysis import PreferenceAnalysis  # Import the new PreferenceAnalysis class


def main():
    # Step 1: Construct relevant dataframes
    print("Starting dataframe construction...")
    df_constructor = DataframeConstruction()
    df_constructor.run_construction()
    print("Dataframe construction complete.")

    # Step 2: Construct share types from network data
    print("Starting network construction for share-types...")
    network_constructor = NetworkConstruction()
    network_constructor.run_construction()
    print("Network construction and share-type generation complete.")

    # Step 3: Recovering parameter distribution
    print("Starting parameter distribution...")
    param_dist = ParameterDistribution()
    param_dist.run_distribution()
    print("Parameter distribution complete.")

    # Step 4: Counterfactuals
    print("Starting counterfactual analysis...")
    counterfactuals = Counterfactuals()
    counterfactuals.run_counterfactuals()
    print("Counterfactual analysis complete.")

    # Step 5: Preferences analysis
    print("Starting preferences analysis...")
    pref_analysis = PreferenceAnalysis()
    pref_analysis.run_analysis()
    print("Preferences analysis complete.")


if __name__ == "__main__":
    main()
