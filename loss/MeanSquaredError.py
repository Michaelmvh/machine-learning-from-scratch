import numpy as np

def MeanSquardError(predictions, targets)-> float:
    """
    Compute the Mean Squared Error between predictions and targets.

    Parameters:
    predictions (np.ndarray): Predicted values (shape: [batch_size, num_outputs]).
    targets (np.ndarray): True values (shape: [batch_size, num_outputs]).

    Returns:
    float: The computed Mean Squared Error.
    """

    return np.exp(np.mean(predictions - targets), 2)