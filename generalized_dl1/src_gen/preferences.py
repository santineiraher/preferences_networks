import itertools
import config  # Importing agent_characteristics from config

class PreferenceClass:
    def __init__(self, covariates):
        self.covariates = covariates
        self.preference_classes = self._build_preference_classes()

    def _build_preference_classes(self):
        preference_classes = {}
        for x_i in self.covariates:
            # Include the empty set along with the powerset
            for S in self._powerset_with_empty(self.covariates):
                preference_classes[(x_i, tuple(S))] = self._build_preference_set(x_i, S)
        return preference_classes

    def _build_preference_set(self, x_i, S):
        # Preference set based on S and covariate x_i
        return [(x_i, 0)] + [(x_i, x_j) for x_j in S]

    def _powerset_with_empty(self, iterable):
        # Return the powerset of the covariates (including empty set)
        return itertools.chain.from_iterable(itertools.combinations(iterable, r) for r in range(0, len(iterable)+1))

    def get_all_preference_classes(self):
        return self.preference_classes

# Example usage
if __name__ == "__main__":
    preference_class_obj = PreferenceClass(config.agent_characteristics)
    preference_classes = preference_class_obj.get_all_preference_classes()

    import pandas as pd

    # Convert preference classes to a table format for exporting
    preference_classes_list = []

    for key, value in preference_classes.items():
        agent, subset = key
        preference_classes_list.append({
            'Agent': agent,
            'Subset': subset,
            'Preference Class': value
        })

    # Create a DataFrame
    df_preference_classes = pd.DataFrame(preference_classes_list)

    # Save to CSV
    output_path = '../output/preference_classes_table.csv'
    df_preference_classes.to_csv(output_path, index=False)
