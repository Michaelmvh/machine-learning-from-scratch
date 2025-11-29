import numpy as np

def CrossEntropyLoss(predictions, targets)-> float:
    """
    Compute the Cross Entropy Loss between predictions and targets.

    Parameters:
    predictions (np.ndarray): Predicted probabilities (shape: [batch_size, num_classes]).
    targets (np.ndarray): One-hot encoded true labels (shape: [batch_size, num_classes]).
    
    Returns:
    float: The computed Cross Entropy Loss.
    """
    return -np.sum(targets * np.log(predictions + 1e-15)) / predictions.shape[0]


"""
Open questions:
- Why do we add a small constant (1e-15) to predictions in the log function?
- Understand the shape requirements for predictions and targets.
"""