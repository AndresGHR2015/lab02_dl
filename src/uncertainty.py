import torch
import torch.nn as nn

class MonteCarloDropoutEstimator:
    """
    Estimates predictive uncertainty using the Monte Carlo Dropout technique.
    Forces the network to retain stochasticity during inference by keeping Dropout active,
    allowing multiple forward passes to sample from an approximate posterior distribution.
    """
    
    def __init__(self, model: nn.Module, num_iterations: int = 50):
        """
        Initializes the estimator with a specific model and iteration count.
        
        Args:
            model: The neural network model to evaluate. Must contain Dropout layers.
            num_iterations: The number of stochastic forward passes to perform per sample.
            
        Raises:
            ValueError: If the number of iterations is less than 1.
        """
        if num_iterations < 1:
            raise ValueError("The number of iterations must be at least 1.")
            
        self.model = model
        self.num_iterations = num_iterations
        
    def estimate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates predictions and quantifies uncertainty for a batch of inputs.
        Applies a sigmoid function to convert raw logits into probability distributions
        before calculating the statistical moments.
        
        Args:
            x: A batch of input feature tensors.
            
        Returns:
            A tuple containing:
                - The mean probability tensor across all Monte Carlo iterations.
                - The standard deviation tensor representing the predictive uncertainty.
        """
        self.model.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_iterations):
                logits = self.model(x)
                probabilities = torch.sigmoid(logits)
                predictions.append(probabilities)
                
        stacked_predictions = torch.stack(predictions)
        
        mean_predictions = torch.mean(stacked_predictions, dim=0)
        std_predictions = torch.std(stacked_predictions, dim=0)
        
        return mean_predictions, std_predictions
