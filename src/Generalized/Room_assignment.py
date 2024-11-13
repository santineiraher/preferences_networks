import pandas as pd
import numpy as np
import config
import os

class Assignment:
    def __init__(self, num_blocks=2):
        self.folder_path = config.EXPOSURES_PATH
        self.output_dir = config.ASSIGNMENT_PATH
        self.num_blocks = num_blocks
        os.makedirs(self.output_dir, exist_ok=True)
        self.summary_file = os.path.join(self.output_dir, "likelihood_summary.csv")

    def extract_first_6_numeric(self, filename):
        """Extract the first 6-digit term from the filename."""
        base_filename = os.path.basename(filename)
        parts = base_filename.split('_')
        return parts[1] if len(parts) > 1 and parts[1].isdigit() else None

    def extract_major(self, filename):
        """Extract the major name from the filename."""
        base_filename = os.path.basename(filename)
        parts = base_filename.split('_')
        return parts[2].split('.')[0] if len(parts) > 2 else None

    def fit_weighted_sbm(self, exposure_matrix):
        """
        Fit a weighted stochastic block model to the exposure matrix.

        Parameters:
        exposure_matrix : pd.DataFrame
            The weighted adjacency matrix where weights are between 0 and 1.

        Returns:
        dict containing:
            - 'block_assignments': array of block assignments for each identifier.
            - 'block_matrix': matrix of edge probabilities between blocks.
            - 'likelihood': final log-likelihood value.
        """
        n = len(exposure_matrix)
        best_likelihood = float('-inf')
        best_result = None

        for _ in range(5):  # Using n_init=10 as in the original optimization
            # Random initialization of block assignments
            block_assignments = np.random.randint(0, self.num_blocks, size=n)

            for iter in range(100):  # max_iter=100 as in the original optimization
                old_assignments = block_assignments.copy()

                # Step 1: Update block matrix
                block_matrix = np.zeros((self.num_blocks, self.num_blocks))
                counts = np.zeros((self.num_blocks, self.num_blocks))

                for i in range(n):
                    for j in range(i + 1, n):
                        bi, bj = block_assignments[i], block_assignments[j]
                        block_matrix[bi, bj] += exposure_matrix.iloc[i, j]
                        block_matrix[bj, bi] += exposure_matrix.iloc[i, j]
                        counts[bi, bj] += 1
                        counts[bj, bi] += 1

                # Avoid division by zero
                counts = np.maximum(counts, 1)
                block_matrix /= counts

                # Step 2: Update node assignments
                for node in range(n):
                    likelihoods = np.zeros(self.num_blocks)
                    for proposed_block in range(self.num_blocks):
                        block_assignments[node] = proposed_block
                        likelihoods[proposed_block] = self.calculate_likelihood(
                            exposure_matrix, block_assignments, block_matrix)

                    block_assignments[node] = np.argmax(likelihoods)

                # Check convergence
                if np.array_equal(old_assignments, block_assignments):
                    break

            # Calculate final likelihood
            final_likelihood = self.calculate_likelihood(
                exposure_matrix, block_assignments, block_matrix)

            if final_likelihood > best_likelihood:
                best_likelihood = final_likelihood
                best_result = {
                    'block_assignments': block_assignments.copy(),
                    'block_matrix': block_matrix.copy(),
                    'likelihood': final_likelihood
                }

        return best_result
    def calculate_likelihood(self, exposure_matrix, block_assignments, block_matrix):
        """
        Calculate the log-likelihood of the current model.
        """
        n = len(exposure_matrix)
        likelihood = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = block_assignments[i], block_assignments[j]
                p = block_matrix[bi, bj]
                w = exposure_matrix.iloc[i, j]

                # Log likelihood for weighted edges
                if w > 0:
                    likelihood += w * np.log(max(p, 1e-10))
                if w < 1:
                    likelihood += (1 - w) * np.log(max(1 - p, 1e-10))

        return likelihood

    def save_results(self, block_assignments, term, major):
        filename = f"room_assignments_{term}_{major}.csv"
        file_path = os.path.join(self.output_dir, filename)
        assignment_df = pd.DataFrame({
            'Identifier': block_assignments.index,
            'Room': block_assignments
        })
        assignment_df.to_csv(file_path, index=False)
        print(f"Block assignments saved to {file_path}")

    def save_summary(self, term, major, likelihood):
        summary_df = pd.DataFrame({
            'Term': [term],
            'Major': [major],
            'Likelihood': [likelihood]
        })
        if not os.path.exists(self.summary_file):
            summary_df.to_csv(self.summary_file, index=False)
        else:
            summary_df.to_csv(self.summary_file, mode='a', header=False, index=False)

    def run_assignment(self):
        files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        for file in files:
            print(f"Processing file: {file}")
            df = pd.read_csv(file, index_col=0)  # Load the exposure matrix with identifiers as index
            term = self.extract_first_6_numeric(file)
            major = self.extract_major(file)
            try:
                term = int(term)
            except (ValueError, TypeError):
                print(f"Error converting term to integer for file {file}")
                continue
            if not major or major == "nan":
                print(f"Invalid major extracted for file {file}")
                continue
            print(f"Term: {term}, Major: {major}")
            result = self.fit_weighted_sbm(df)
            if result:
                block_assignments = pd.Series(result['block_assignments'], index=df.index)
                self.save_results(block_assignments, term, major)
                self.save_summary(term, major, result['likelihood'])

# Example usage
if __name__ == "__main__":
    assignment = Assignment(num_blocks=2)
    assignment.run_assignment()



