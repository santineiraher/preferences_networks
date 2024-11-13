import numpy as np

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

    def get_agent_combinations():
        types = ['A', 'B']
        indices = range(1, num_indices + 1)
        return [(t, i) for t in types for i in indices]

    agents = get_agent_combinations()

    for agent in agents:
        type_letter = agent[0]
        template_key = ((type_letter, 'X'), 0)
        if template_key in template_matrix:
            expanded_matrix[(agent, 0)] = template_matrix[template_key]

    for agent1 in agents:
        for agent2 in agents:
            type1, idx1 = agent1
            type2, idx2 = agent2

            if type1 == type2 and idx1 != idx2:
                template_key = ((type1, 'X'), (type2, 'Y'))
            else:
                template_key = ((type1, 'X'), (type2, 'X'))

            if template_key in template_matrix:
                expanded_matrix[(agent1, agent2)] = template_matrix[template_key]

    return expanded_matrix


def create_uniform_utility_array(size_dim):
    """
    Creates a 10-dimensional array with uniformly distributed values.

    Args:
        size_dim (int): Size of each dimension in the array

    Returns:
        np.ndarray: 10-dimensional array with uniformly distributed values
    """
    shape = (size_dim,) * 10
    dimensions = [np.linspace(0, 1, size_dim) for _ in range(10)]
    mesh = np.meshgrid(*dimensions, indexing='ij')
    utility_array = np.zeros(shape)
    for dim_values in mesh:
        utility_array += dim_values
    utility_array /= len(mesh)
    return utility_array


def calculate_array_memory(size_dim):
    """
    Calculates memory size of a 10-dimensional array with float64 values.

    Args:
        size_dim (int): Size of each dimension in the array

    Returns:
        tuple: (theoretical_size_bytes, actual_size_bytes, human_readable_size)
    """
    bytes_per_element = 8
    num_elements = size_dim ** 10
    theoretical_size = num_elements * bytes_per_element
    array = np.zeros((size_dim,) * 10, dtype=np.float64)
    actual_size = array.nbytes

    def bytes_to_human_readable(bytes_size):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.2f} PB"

    human_readable = bytes_to_human_readable(actual_size)
    return theoretical_size, actual_size, human_readable


def main():
    template_matrix = {
        (('A', 'X'), 0): 0.1,
        (('B', 'X'), 0): 0.1,
        (('A', 'X'), ('A', 'X')): 0.8,
        (('B', 'X'), ('B', 'X')): 0.9,
        (('A', 'X'), ('A', 'Y')): 0.6,
        (('A', 'X'), ('B', 'X')): 0.4,
        (('A', 'X'), ('B', 'Y')): 0.2,
        (('B', 'X'), ('A', 'X')): 0.4,
        (('B', 'X'), ('A', 'Y')): 0.2,
        (('B', 'X'), ('B', 'Y')): 0.5,
    }

    expanded_matrix = expand_utility_matrix(template_matrix)
    print("Expanded Utility Matrix:")
    for key, value in expanded_matrix.items():
        print(f"{key}: {value}")

    size_dim = 8
    result_array = create_uniform_utility_array(size_dim)
    print("\nUniform Utility Array:")
    print(f"Array shape: {result_array.shape}")
    print(f"Minimum value: {result_array.min()}")
    print(f"Maximum value: {result_array.max()}")
    print(f"Mean value: {result_array.mean()}")

    print("\nMemory Requirements for Different Array Sizes:")
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"{'Size':>4} | {'Elements':>15} | {'Memory':>15}")
    print("-" * 50)
    for size in sizes:
        theoretical, actual, human = calculate_array_memory(size)
        elements = size ** 10
        print(f"{size:4d} | {elements:15,d} | {human:>15}")
    return expanded_matrix


if __name__ == "__main__":
    result=main()
    print(result)
