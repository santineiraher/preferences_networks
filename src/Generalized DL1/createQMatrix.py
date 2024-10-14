from itertools import product
import numpy as np
import os

# Define agent characteristics
agent_characteristics = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]

# Define network types
def generate_network_types(characteristics):
    all_pairs = list(product(characteristics, repeat=2))
    special_types = [(char, 'NA') for char in characteristics]
    network_types = all_pairs + special_types
    return network_types

all_network_types = generate_network_types(agent_characteristics)

# Define the 36 preference classes
preference_classes = [
    # First 18 classes (B-type)
    {(('B', 1), 'NA')},
    {(('B', 2), 'NA')},
    {(('B', 1), 'NA'), (('B', 1), ('B', 1))},
    {(('B', 1), 'NA'), (('B', 1), ('B', 2))},
    {(('B', 2), 'NA'), (('B', 2), ('B', 1))},
    {(('B', 2), 'NA'), (('B', 2), ('B', 2))},
    {(('B', 1), 'NA'), (('B', 1), ('A', 1))},
    {(('B', 1), 'NA'), (('B', 1), ('A', 2))},
    {(('B', 2), 'NA'), (('B', 2), ('A', 1))},
    {(('B', 2), 'NA'), (('B', 2), ('A', 2))},
    {(('B', 1), 'NA'), (('B', 1), ('B', 1)), (('B', 1), ('A', 1))},
    {(('B', 2), 'NA'), (('B', 2), ('B', 1)), (('B', 2), ('A', 1))},
    {(('B', 1), 'NA'), (('B', 1), ('B', 2)), (('B', 1), ('A', 1))},
    {(('B', 2), 'NA'), (('B', 2), ('B', 2)), (('B', 2), ('A', 1))},
    {(('B', 1), 'NA'), (('B', 1), ('B', 1)), (('B', 1), ('A', 2))},
    {(('B', 2), 'NA'), (('B', 2), ('B', 1)), (('B', 2), ('A', 2))},
    {(('B', 1), 'NA'), (('B', 1), ('B', 2)), (('B', 1), ('A', 2))},
    {(('B', 2), 'NA'), (('B', 2), ('B', 2)), (('B', 2), ('A', 2))},
    
    # Next 18 classes (A-type)
    {(('A', 1), 'NA')},
    {(('A', 2), 'NA')},
    {(('A', 1), 'NA'), (('A', 1), ('B', 1))},
    {(('A', 1), 'NA'), (('A', 1), ('B', 2))},
    {(('A', 2), 'NA'), (('A', 2), ('B', 1))},
    {(('A', 2), 'NA'), (('A', 2), ('B', 2))},
    {(('A', 1), 'NA'), (('A', 1), ('A', 1))},
    {(('A', 1), 'NA'), (('A', 1), ('A', 2))},
    {(('A', 2), 'NA'), (('A', 2), ('A', 1))},
    {(('A', 2), 'NA'), (('A', 2), ('A', 2))},
    {(('A', 1), 'NA'), (('A', 1), ('B', 1)), (('A', 1), ('A', 1))},
    {(('A', 2), 'NA'), (('A', 2), ('B', 1)), (('A', 2), ('A', 1))},
    {(('A', 1), 'NA'), (('A', 1), ('B', 2)), (('A', 1), ('A', 1))},
    {(('A', 2), 'NA'), (('A', 2), ('B', 2)), (('A', 2), ('A', 1))},
    {(('A', 1), 'NA'), (('A', 1), ('B', 1)), (('A', 1), ('A', 2))},
    {(('A', 2), 'NA'), (('A', 2), ('B', 1)), (('A', 2), ('A', 2))},
    {(('A', 1), 'NA'), (('A', 1), ('B', 2)), (('A', 1), ('A', 2))},
    {(('A', 2), 'NA'), (('A', 2), ('B', 2)), (('A', 2), ('A', 2))}
]

# Function to generate valid preference class-type combinations
def generate_valid_combinations(preference_classes):
    valid_combinations = []
    for i, pref_class in enumerate(preference_classes, 1):
        for network_type in pref_class:
            valid_combinations.append((i, network_type))
    return valid_combinations

# Generate valid combinations
valid_combinations = generate_valid_combinations(preference_classes)

# Create the matrix
matrix_size = len(valid_combinations)
matrix = np.zeros((matrix_size, matrix_size), dtype=int)

# Create a mapping of combinations to matrix indices
combination_to_index = {combo: i for i, combo in enumerate(valid_combinations)}

# Function to get the matrix index for a given combination
def get_matrix_index(combination):
    return combination_to_index[combination]

# Function to format the combination for display
def format_combination(combo):
    class_num, (agent, target) = combo
    agent_str = f"{agent[0]}{agent[1]}"
    target_str = 'NA' if target == 'NA' else f"{target[0]}{target[1]}"
    return f"{class_num}-{agent_str}{target_str}"

# Function to create assignment parameters
def generate_assignment_parameters(valid_combinations, preference_classes):
    assignment_parameters = []
    for i, (class_num, network_type) in enumerate(valid_combinations):
        # For each valid combination, create an assignment parameter
        assignment_parameters.append((network_type, f"Class {class_num}"))
    return assignment_parameters

# Function to update the matrix based on the specified rules
def update_matrix(matrix, valid_combinations, preference_classes):
    agent_characteristics = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
    
    for i, (class_i, type_i) in enumerate(valid_combinations):
        # Step 1: Check if the type associated with row i is of the form (X, NA)
        if type_i[1] != 'NA':
            continue  # If not, continue to the next row
        
        x = type_i[0]  # This is the X in (X, NA)
        
        # Step 2: Loop over columns and check conditions
        for j, (class_j, type_j) in enumerate(valid_combinations):
            # Check if the type associated with column j is of the form (Z, NA)
            if type_j[1] != 'NA':
                continue
            
            z = type_j[0]  # This is the Z in (Z, NA)
            
            # Check if Z is one of the four agent characteristics
            if z not in agent_characteristics:
                continue
            
            # Check if (X,Z) belongs to the preference class of row i
            # and (Z,X) belongs to the preference class of column j
            if ((x, z) in preference_classes[class_i - 1] and 
                (z, x) in preference_classes[class_j - 1]):
                matrix[i, j] = 1

# Update the matrix
update_matrix(matrix, valid_combinations, preference_classes)

# Function to create a mapping of indices to preference classes
def create_index_mapping(valid_combinations, preference_classes):
    mapping = {}
    for i, (class_num, network_type) in enumerate(valid_combinations):
        pref_class = preference_classes[class_num - 1]
        mapping[i] = f"Class {class_num}: {pref_class}"
    return mapping


if __name__ == "__main__":
    # Print the matrix
    print("Q-Matrix:")
    print(matrix)

    # Print the mapping
    print("\nIndex Mapping:")
    index_mapping = create_index_mapping(valid_combinations, preference_classes)
    for i, pref_class in index_mapping.items():
        print(f"{i}: {pref_class}")

    # Save the matrix and mapping to a file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    matrix_file = os.path.join(output_dir, "q_matrix.npy")
    mapping_file = os.path.join(output_dir, "index_mapping.txt")

    np.save(matrix_file, matrix)
    with open(mapping_file, "w") as f:
        for i, pref_class in index_mapping.items():
            f.write(f"{i}: {pref_class}\n")

    print(f"\nMatrix saved to: {matrix_file}")
    print(f"Index mapping saved to: {mapping_file}")