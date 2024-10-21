import config

class AssignmentParameters:
    def __init__(self, valid_combinations, preference_classes):
        self.agent_characteristics = config.agent_characteristics  # Use agent characteristics from config
        self.valid_combinations = valid_combinations
        self.preference_classes = preference_classes
        self.alpha = self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initialize assignment parameters. If agents do not match the preference classes,
        the corresponding alpha value is set to 0.
        """
        alpha = {}
        for i, (class_i, type_i) in enumerate(self.valid_combinations):
            if type_i[1] != 'NA':  # Skip invalid combinations
                continue
            x = type_i[0]

            for j, (class_j, type_j) in enumerate(self.valid_combinations):
                if type_j[1] != 'NA':
                    continue
                z = type_j[0]

                # Stability condition: if agent z is not in agent x's preference class, alpha is 0
                if z not in [ac[0] for ac in self.agent_characteristics]:
                    continue

                # Check if the preference class condition is satisfied
                if ((x, z) in self.preference_classes.get(class_i - 1, []) and
                    (z, x) in self.preference_classes.get(class_j - 1, [])):
                    alpha[(class_i, class_j)] = 1  # Assign non-zero alpha if both preferences are valid
                else:
                    alpha[(class_i, class_j)] = 0  # Otherwise, alpha is 0 by stability condition
        return alpha

    def set_assignment_parameter(self, class_i, class_j, value):
        """
        Set a specific assignment parameter.
        """
        self.alpha[(class_i, class_j)] = value

    def get_assignment_parameter(self, class_i, class_j):
        """
        Get a specific assignment parameter. Returns 0 by default if not found.
        """
        return self.alpha.get((class_i, class_j), 0)

    def enforce_probability_completeness(self):
        """
        Ensure that for each agent, the sum of assignment parameters across all preference
        classes equals 1.
        """
        for class_i in range(1, len(self.valid_combinations) + 1):
            total = sum(self.alpha.get((class_i, class_j), 0) for class_j in range(1, len(self.valid_combinations) + 1))
            if total != 1:
                # Adjust the assignment parameters so that they sum to 1
                for class_j in range(1, len(self.valid_combinations) + 1):
                    if self.alpha.get((class_i, class_j), 0) > 0:
                        self.alpha[(class_i, class_j)] /= total
