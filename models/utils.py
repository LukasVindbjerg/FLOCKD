# Model utilities.
# E.g. load weights in particular format
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn


# Get the weights from the model as a list of numpy arrays
def get_weight_matrices(model):
    """
    Extracts weight matrices from a PyTorch model as a list of numpy arrays.
    
    Args:
        model (nn.Module): The PyTorch model to extract weights from.
    
    Returns:
        List[np.ndarray]: A list containing the weight matrices as numpy arrays.
    """
    weight_matrices = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weight_matrices.append(layer.weight.data.cpu().numpy())
    return weight_matrices


import matplotlib.pyplot as plt

def compute_similarity_matrix(modelsTrainedOnX, modelsTrainedOnY, layer, operation):
    """
    Computes a similarity matrix between two lists of models based on the L1 norm of their weight matrices.
    
    Args:
        modelsTrainedOnX (List[nn.Module]): List of models trained on datasplit X.
        modelsTrainedOnY (List[nn.Module]): List of models trained on datasplit Y.
    
    Returns:
        np.ndarray: A 2D array representing the similarity matrix.
    """
    N = len(modelsTrainedOnX)
    similarity_matrix = np.zeros((N, N))

    for i in range(N):
        weights_X = get_weight_matrices(modelsTrainedOnX[i])
        weights_X_layer = weights_X[layer]

        for j in range(N):
            weights_Y = get_weight_matrices(modelsTrainedOnY[j])
            weights_Y_layer = weights_Y[layer]

            similarity = operation(weights_X_layer, weights_Y_layer)
            similarity_matrix[i, j] = similarity

    return similarity_matrix

def visualize_similarity_matrix(similarity_matrix):
    """
    Visualizes the similarity matrix using a heatmap.
    
    Args:
        similarity_matrix (np.ndarray): The similarity matrix to visualize.
    """
    labels_x = [f"Model X{i}" for i in range(similarity_matrix.shape[0])]
    labels_y = [f"Model Y{i}" for i in range(similarity_matrix.shape[1])]

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", xticklabels=labels_x, yticklabels=labels_y, cmap="viridis")
    plt.title("Model Similarity Heatmap")
    plt.xlabel("Models Trained on Y")
    plt.ylabel("Models Trained on X")
    plt.tight_layout()
    plt.show()


# Example usage:
# modelsTrainedOnX = [...]  # List of models trained on datasplit X
# modelsTrainedOnY = [...]  # List of models trained on datasplit Y
# similarity_matrix = compute_similarity_matrix(modelsTrainedOnX, modelsTrainedOnY)
# visualize_similarity_matrix(similarity_matrix)
