import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    hamming_loss, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import copy

from src.models import ShallowNeuralNetwork
from src.data_loader import CognitiveDataset

class MetricsCalculator:
    """
    Computes a comprehensive suite of evaluation metrics tailored for multilabel classification tasks.
    """
    
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates Hamming Loss, Precision (Micro), Recall (Micro), F1 (Micro), F1 (Macro), 
        and Exact Match (Subset Accuracy) between ground truth labels and binary predictions.
        
        Args:
            y_true: Ground truth multilabel indicators.
            y_pred: Predicted binary multilabel indicators.
            
        Returns:
            A dictionary containing the computed metrics.
        """
        return {
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "precision_micro": float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
            "recall_micro": float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            "exact_match": float(accuracy_score(y_true, y_pred))
        }

class NestedCrossValidator:
    """
    Orchestrates the Nested Cross-Validation strategy to rigorously evaluate hyperparameters
    and generalization performance using Iterative Multilabel Stratification.
    """
    
    def __init__(self, device: torch.device, outer_folds: int = 5, inner_folds: int = 3):
        """
        Initializes the cross-validator.
        
        Args:
            device: The computation device (CPU or CUDA).
            outer_folds: The number of folds for performance evaluation.
            inner_folds: The number of folds for hyperparameter tuning.
        """
        self.device = device
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        
    def execute(self, x: np.ndarray, y: np.ndarray, param_grid: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Executes the full nested cross-validation loop.
        
        Args:
            x: Input feature array.
            y: Multilabel target array.
            param_grid: A list of dictionaries, where each dictionary represents a hyperparameter configuration.
            
        Returns:
            A dictionary containing the aggregated mean metrics across all outer folds.
        """
        outer_cv = MultilabelStratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=42)
        outer_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y)):
            x_train_outer, y_train_outer = x[train_idx], y[train_idx]
            x_test_outer, y_test_outer = x[test_idx], y[test_idx]
            
            best_params = self._tune_hyperparameters(x_train_outer, y_train_outer, param_grid)
            
            metrics = self._evaluate_model(x_train_outer, y_train_outer, x_test_outer, y_test_outer, best_params)
            outer_metrics.append(metrics)
            
        return self._aggregate_metrics(outer_metrics)
        
    def _tune_hyperparameters(self, x: np.ndarray, y: np.ndarray, param_grid: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs inner cross-validation to select the optimal hyperparameters based on F1 Micro.
        
        Args:
            x: Training feature array for the current outer fold.
            y: Training target array for the current outer fold.
            param_grid: The grid of hyperparameters to evaluate.
            
        Returns:
            The best hyperparameter configuration dictionary.
        """
        inner_cv = MultilabelStratifiedKFold(n_splits=self.inner_folds, shuffle=True, random_state=42)
        best_score = -1.0
        best_params = None
        
        for params in param_grid:
            fold_scores = []
            
            for train_idx, val_idx in inner_cv.split(x, y):
                x_train_inner, y_train_inner = x[train_idx], y[train_idx]
                x_val_inner, y_val_inner = x[val_idx], y[val_idx]
                
                metrics = self._evaluate_model(
                    x_train_inner, y_train_inner, 
                    x_val_inner, y_val_inner, 
                    params,
                    epochs=params.get('epochs', 20)
                )
                fold_scores.append(metrics['f1_micro'])
                
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                
        if best_params is None:
            best_params = param_grid[0]
            
        return best_params
        
    def _evaluate_model(
        self, 
        x_train: np.ndarray, y_train: np.ndarray, 
        x_val: np.ndarray, y_val: np.ndarray, 
        params: Dict[str, Any],
        epochs: int = 30
    ) -> Dict[str, float]:
        """
        Trains a model configuration and evaluates it on a validation set.
        
        Args:
            x_train: Training features.
            y_train: Training targets.
            x_val: Validation features.
            y_val: Validation targets.
            params: Dictionary containing architecture and training parameters.
            epochs: Number of training epochs.
            
        Returns:
            A dictionary of calculated metrics on the validation set.
        """
        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]
        
        model = ShallowNeuralNetwork(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            output_dim=output_dim,
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        num_positives = torch.tensor((y_train == 1).sum(axis=0), dtype=torch.float32) + 1e-5
        num_negatives = torch.tensor((y_train == 0).sum(axis=0), dtype=torch.float32)
        pos_weight = (num_negatives / num_positives).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        
        train_dataset = CognitiveDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        val_dataset = CognitiveDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                probabilities = torch.sigmoid(logits)
                binary_preds = (probabilities > 0.5).float()
                all_preds.append(binary_preds.cpu().numpy())
                
        y_pred = np.vstack(all_preds)
        return MetricsCalculator.calculate(y_val, y_pred)
        
    def _aggregate_metrics(self, outer_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Computes the mean of metrics gathered across outer folds.
        
        Args:
            outer_metrics: A list of metric dictionaries.
            
        Returns:
            A dictionary containing the mean values for each metric.
        """
        aggregated = {}
        if not outer_metrics:
            return aggregated
            
        keys = outer_metrics[0].keys()
        for key in keys:
            aggregated[key] = float(np.mean([m[key] for m in outer_metrics]))
            
        return aggregated
