import numpy as np
class MatrixComparison:
# L1 norm of the difference between two matrices
    def matrix_difference_norm(matrix1, matrix2):
        """
        Compute the L1 norm of the difference between two matrices.
        
        Args:
            matrix1 (np.ndarray): The first matrix.
            matrix2 (np.ndarray): The second matrix.
        
        Returns:
            float: The L1 norm of the difference between the two matrices.
        """
        return np.linalg.norm(matrix1 - matrix2, ord=1)

    # Frobenuis norm of the difference between two matrices
    def matrix_difference_frobenius(matrix1, matrix2):
        """
        Compute the Frobenius norm of the difference between two matrices.
        
        Args:
            matrix1 (np.ndarray): The first matrix.
            matrix2 (np.ndarray): The second matrix.
        
        Returns:
            float: The Frobenius norm of the difference between the two matrices.
        """
        return np.linalg.norm(matrix1 - matrix2, ord='fro')

    # Nuclear norm of the difference between two matrices
    def matrix_difference_nuclear(matrix1, matrix2):
        """
        Compute the nuclear norm of the difference between two matrices.
        
        Args:
            matrix1 (np.ndarray): The first matrix.
            matrix2 (np.ndarray): The second matrix.
        
        Returns:
            float: The nuclear norm of the difference between the two matrices.
        """
        return np.linalg.norm(matrix1 - matrix2, ord='nuc')

    # Cosine similarity between two matrices
    def matrix_cosine_similarity(matrix1, matrix2):
        """
        Compute the cosine similarity between two matrices.
        
        Args:
            matrix1 (np.ndarray): The first matrix.
            matrix2 (np.ndarray): The second matrix.
        
        Returns:
            float: The cosine similarity between the two matrices.
        """
        return np.dot(matrix1.flatten(), matrix2.flatten()) / (np.linalg.norm(matrix1) * np.linalg.norm(matrix2))

    # KL divergence between two matrices
    def matrix_kl_divergence(matrix1, matrix2):
        """
        Compute the Kullback-Leibler divergence between two matrices.
        
        Args:
            matrix1 (np.ndarray): The first matrix.
            matrix2 (np.ndarray): The second matrix.
        
        Returns:
            float: The Kullback-Leibler divergence between the two matrices.
        """
        return np.sum(matrix1 * np.log(matrix1 / matrix2) - matrix1 + matrix2)

    # Wasserstein distance between two matrices
    def matrix_wasserstein_distance(matrix1, matrix2):
        """
        Compute the Wasserstein distance between two matrices.
        
        Args:
            matrix1 (np.ndarray): The first matrix.
            matrix2 (np.ndarray): The second matrix.
        
        Returns:
            float: The Wasserstein distance between the two matrices.
        """
        return np.linalg.norm(matrix1 - matrix2, ord=1)