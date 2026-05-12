import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List

class DataPreprocessor:
    """
    Handles the transformation of tabular data into a format suitable for neural network training.
    Specifically isolates features from target variables and applies One-Hot encoding to convert
    multiclass target variables into multilabel representations.
    """
    
    def __init__(self, feature_columns: List[str]):
        """
        Initializes the DataPreprocessor.
        
        Args:
            feature_columns: A list of column names representing the input features.
        """
        self.feature_columns = feature_columns
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.imputer_numeric = SimpleImputer(strategy='median')
        self.imputer_binary = SimpleImputer(strategy='most_frequent')
        self.numeric_columns = []
        self.binary_columns = []
        
    def fit_transform(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits the One-Hot encoder on the target column and transforms it, while extracting features.
        
        Args:
            data: The complete DataFrame containing both features and the target.
            target_column: The specific name of the target variable to be processed.
            
        Returns:
            A tuple containing:
                - A DataFrame of the extracted features.
                - A DataFrame of the One-Hot encoded (multilabel) target variable.
                
        Raises:
            ValueError: If the required feature columns or target column are missing from the data.
        """
        self._validate_columns(data, target_column)
        
        features = data[self.feature_columns].copy()
        
        self.binary_columns = [col for col in self.feature_columns if features[col].dropna().nunique() <= 2]
        self.numeric_columns = [col for col in self.feature_columns if col not in self.binary_columns]
        
        if self.numeric_columns:
            features[self.numeric_columns] = self.imputer_numeric.fit_transform(features[self.numeric_columns])
            
        if self.binary_columns:
            features[self.binary_columns] = self.imputer_binary.fit_transform(features[self.binary_columns])
            
        scaled_features = self.scaler.fit_transform(features)
        features_df = pd.DataFrame(scaled_features, columns=self.feature_columns, index=data.index)
        
        target_data = data[[target_column]]
        encoded_target = self.encoder.fit_transform(target_data)
        
        encoded_target_df = pd.DataFrame(
            encoded_target, 
            columns=[f"{target_column}_{cat}" for cat in self.encoder.categories_[0]],
            index=data.index
        )
        
        return features_df, encoded_target_df
        
    def transform(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms features and the target column using the previously fitted One-Hot encoder.
        
        Args:
            data: The complete DataFrame containing both features and the target.
            target_column: The specific name of the target variable to be processed.
            
        Returns:
            A tuple containing:
                - A DataFrame of the extracted features.
                - A DataFrame of the One-Hot encoded (multilabel) target variable.
                
        Raises:
            ValueError: If the required feature columns or target column are missing from the data.
            RuntimeError: If the encoder has not been fitted yet.
        """
        self._validate_columns(data, target_column)
        
        if not hasattr(self.encoder, 'categories_'):
            raise RuntimeError("The encoder has not been fitted yet. Call fit_transform first.")
            
        features = data[self.feature_columns].copy()
        
        if self.numeric_columns:
            features[self.numeric_columns] = self.imputer_numeric.transform(features[self.numeric_columns])
            
        if self.binary_columns:
            features[self.binary_columns] = self.imputer_binary.transform(features[self.binary_columns])
            
        scaled_features = self.scaler.transform(features)
        features_df = pd.DataFrame(scaled_features, columns=self.feature_columns, index=data.index)
        
        target_data = data[[target_column]]
        encoded_target = self.encoder.transform(target_data)
        
        encoded_target_df = pd.DataFrame(
            encoded_target, 
            columns=[f"{target_column}_{cat}" for cat in self.encoder.categories_[0]],
            index=data.index
        )
        
        return features_df, encoded_target_df
        
    def _validate_columns(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Validates the presence of expected columns in the provided DataFrame.
        
        Args:
            data: The DataFrame to validate.
            target_column: The target column that must be present.
            
        Raises:
            ValueError: If columns are missing.
        """
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
            
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' is missing from the data.")
