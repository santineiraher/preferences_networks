def expand_utility_matrix(template_matrix, num_indices=2):
    """
    Expands a template utility matrix into a full matrix with specific indices.

    Args:
        template_matrix (dict): Template utility matrix with 'X' and 'Y' placeholders
        num_indices (int): Number of indices to expand to (default: 2)

    Returns:
        dict: Expanded utility matrix with specific indices
    """
    expanded_matrix = {}

    # Helper function to get all possible agent combinations
    def get_agent_combinations():
        types = ['A', 'B']
        indices = range(1, num_indices + 1)
        agents = [(t, i) for t in types for i in indices]
        return agents

    # Get all possible agents
    agents = get_agent_combinations()

    # Handle alone cases (interactions with 0)
    for agent in agents:
        type_letter = agent[0]
        template_key = ((type_letter, 'X'), 0)
        if template_key in template_matrix:
            expanded_matrix[(agent, 0)] = template_matrix[template_key]

    # Handle interactions between agents
    for agent1 in agents:
        for agent2 in agents:
            type1, idx1 = agent1
            type2, idx2 = agent2

            # Skip self-interactions where indices are different
            if type1 == type2 and idx1 != idx2:
                template_key = ((type1, 'X'), (type2, 'Y'))
            else:
                template_key = ((type1, 'X'), (type2, 'X'))

            # Only add if template key exists
            if template_key in template_matrix:
                expanded_matrix[(agent1, agent2)] = template_matrix[template_key]

                # If it's a different type interaction, potentially modify the value
                if type1 != type2 or (type1 == type2 and idx1 != idx2):
                    # You can add logic here to modify the value based on indices
                    # For example, random variation or systematic changes
                    pass

    return expanded_matrix


# Example usage:
template_matrix = {
    # Alone (f(x_i, 0))
    (('A', 'X'), 0): 0.1,
    (('B', 'X'), 0): 0.1,
    # Interaction with same type
    (('A', 'X'), ('A', 'X')): 0.8,
    (('B', 'X'), ('B', 'X')): 0.9,
    # Interaction with different types
    (('A', 'X'), ('A', 'Y')): 0.6,
    (('A', 'X'), ('B', 'X')): 0.4,
    (('A', 'X'), ('B', 'Y')): 0.2,
    (('B', 'X'), ('A', 'X')): 0.4,
    (('B', 'X'), ('A', 'Y')): 0.2,
    (('B', 'X'), ('B', 'Y')): 0.5,
}

# Expand the matrix
expanded = expand_utility_matrix(template_matrix)


def create_uniform_utility_array(size_dim):
    """
    Creates a 10-dimensional array where values along each dimension
    vary uniformly between 0 and 1.

    Args:
        size_dim (int): Size of each dimension in the array

    Returns:
        np.ndarray: 10-dimensional array with uniformly distributed values
    """
    # Create a 10-dimensional array
    shape = tuple([size_dim] * 10)

    # Create uniform values for each dimension
    dimensions = [np.linspace(0, 1, size_dim) for _ in range(10)]

    # Create meshgrid for all dimensions
    mesh = np.meshgrid(*dimensions, indexing='ij')

    # Initialize the final array
    utility_array = np.zeros(shape)

    # Fill the array with the average of all dimensional values
    for dim_values in mesh:
        utility_array += dim_values

    # Normalize to keep values between 0 and 1
    utility_array /= len(mesh)

    return utility_array


