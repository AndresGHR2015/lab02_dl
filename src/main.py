# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor
from src.data_loader import CognitiveDataset
from src.models import ShallowNeuralNetwork
from src.evaluation import NestedCrossValidator
from src.uncertainty import MonteCarloDropoutEstimator



def save_model(model: nn.Module, filepath: str) -> None:
    """
    Saves the trained PyTorch model weights to disk.
    
    Args:
        model: The trained neural network model.
        filepath: The destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)

def main() -> None:
    """
    Orchestrates the entire machine learning pipeline: data generation, preprocessing, 
    nested cross-validation evaluation, final training, and predictive uncertainty estimation
    across all target variables.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    feature_columns = [
        "Día", "Mes", "Año", "Estación", "País", "Ciudad", 
        "CalleLugar", "NumeroPiso", "Miguel2", "González2", 
        "Avenida2", "Imperial2", "A682", "Caldera2", "Copiapo2"
    ]
    
    raw_data = pd.read_spss("data/raw/15 atributos R0-R5.sav")
    
    targets = ["GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5"]
    all_metrics = {}
    
    for target_column in targets:
        print(f"\n{'='*50}\nProcessing target: {target_column}\n{'='*50}")
        
        preprocessor = DataPreprocessor(feature_columns=feature_columns)
        features_df, target_df = preprocessor.fit_transform(raw_data, target_column)
        x_numpy = features_df.values
        y_numpy = target_df.values
        
        print(f"Data preprocessed for {target_column}. Features shape: {x_numpy.shape}, Target shape: {y_numpy.shape}")
        
        hyperparameter_grid = [
            {"hidden_dim": 32, "dropout_rate": 0.1, "lr": 0.01, "batch_size": 32, "epochs": 150},
            {"hidden_dim": 64, "dropout_rate": 0.2, "lr": 0.005, "batch_size": 64, "epochs": 150},
            {"hidden_dim": 128, "dropout_rate": 0.2, "lr": 0.001, "batch_size": 128, "epochs": 150}
        ]
        
        validator = NestedCrossValidator(device=device, outer_folds=5, inner_folds=3)
        print("Starting Nested Cross-Validation...")
        cv_metrics = validator.execute(x_numpy, y_numpy, hyperparameter_grid)
        
        print(f"\n--- Nested Cross-Validation Results ({target_column}) ---")
        print(json.dumps(cv_metrics, indent=4))
        all_metrics[target_column] = cv_metrics
        
        best_params = hyperparameter_grid[1] 
        
        final_model = ShallowNeuralNetwork(
            input_dim=x_numpy.shape[1],
            hidden_dim=best_params["hidden_dim"],
            output_dim=y_numpy.shape[1],
            dropout_rate=best_params["dropout_rate"]
        ).to(device)
        
        num_positives = torch.tensor((y_numpy == 1).sum(axis=0), dtype=torch.float32) + 1e-5
        num_negatives = torch.tensor((y_numpy == 0).sum(axis=0), dtype=torch.float32)
        pos_weight = (num_negatives / num_positives).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
        
        full_dataset = CognitiveDataset(x_numpy, y_numpy)
        full_loader = DataLoader(full_dataset, batch_size=best_params["batch_size"], shuffle=True)
        
        print(f"\nTraining final model on complete dataset for {target_column}...")
        final_model.train()
        for epoch in range(best_params["epochs"]):
            epoch_loss = 0.0
            for batch_x, batch_y in full_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = final_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        model_path = os.path.join("models", f"best_model_{target_column}.pth")
        save_model(final_model, model_path)
        print(f"Final model for {target_column} saved securely at: {model_path}")
        
        print("\nEvaluating Predictive Uncertainty via Monte Carlo Dropout...")
        estimator = MonteCarloDropoutEstimator(model=final_model, num_iterations=50)
        
        sample_x = torch.tensor(x_numpy[:1], dtype=torch.float32).to(device)
        mean_probs, std_probs = estimator.estimate(sample_x)
        
        print("Sample Output (Mean Probabilities):", mean_probs.cpu().numpy())
        print("Sample Output (Predictive Uncertainty/Std Dev):", std_probs.cpu().numpy())
        
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "metrics_summary.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"\n{'='*50}")
    print("ALL TARGETS PROCESSED.")
    print(f"Consolidated metrics saved to: {results_path}")
    print("Consolidated Metrics Summary:")
    print(json.dumps(all_metrics, indent=4))

if __name__ == "__main__":
    main()
