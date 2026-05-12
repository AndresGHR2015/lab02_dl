import torch
import torch.nn as nn

class ShallowNeuralNetwork(nn.Module):
    """
    A shallow neural network architecture comprising a single hidden layer, ReLU activation, 
    and Dropout, mapping input tabular features to raw output logits for multilabel classification.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        """
        Initializes the neural network layers.
        
        Args:
            input_dim: The number of input features.
            hidden_dim: The number of neurons in the single hidden layer.
            output_dim: The number of output classes (which map to binary labels in a multilabel context).
            dropout_rate: The probability of zeroing a neuron's output during training to prevent overfitting.
            
        Raises:
            ValueError: If any dimensional parameter is less than or equal to zero.
        """
        super().__init__()
        
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("Dimensions must be strictly positive integers.")
            
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the network.
        
        Args:
            x: A batch of input feature tensors.
            
        Returns:
            A tensor containing the raw unnormalized logits for each output class.
        """
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
