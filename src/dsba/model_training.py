"""
This module is just a convenience to train a simple classifier.
Its presence is a bit artificial for the exercice and not required to develop an MLOps platform.
The MLOps course is not about model training.
"""

from dataclasses import dataclass
import logging
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.base import ClassifierMixin, RegressorMixin

from dsba.model_registry import ClassifierMetadata
from .preprocessing import split_features_and_target, preprocess_dataframe


def train_simple_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a simple classifier")
    df = preprocess_dataframe(df)
    X, y = split_features_and_target(df, target_column)
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)

    logging.info("Done training a simple classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="xgboost",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="",
        performance_metrics={},
    )
    return model, metadata

def train_random_forest_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info(f"Start training a Random Forest classifier with id: {model_id}")
    df_processed = preprocess_dataframe(df.copy())
    X, y = split_features_and_target(df_processed, target_column)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    logging.info(f"Done training Random Forest classifier: {model_id}")
    
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="random_forest",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="A Random Forest classifier",
        performance_metrics={}, 
    )
    return model, metadata


import random
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np # Make sure numpy is imported if not already

# --- New Model Classes ---

class RandomClassifier(BaseEstimator, ClassifierMixin):
    """A classifier that predicts randomly 0 or 1."""
    def fit(self, X, y=None):
        # No actual fitting needed for this simple model
        self.classes_ = [0, 1] # Assuming binary classification 0 or 1
        return self

    def predict(self, X):
        # Predict randomly for each sample in X
        return np.random.randint(0, 2, size=len(X))

class NaivePositiveClassifier(BaseEstimator, ClassifierMixin):
    """A classifier that always predicts 1 (positive)."""
    def fit(self, X, y=None):
        # No actual fitting needed
        self.classes_ = [0, 1] # Assuming binary classification 0 or 1
        return self

    def predict(self, X):
        # Always predict 1
        return np.ones(len(X), dtype=int)

class NaiveNegativeClassifier(BaseEstimator, ClassifierMixin):
    """A classifier that always predicts 0 (negative)."""
    def fit(self, X, y=None):
        self.classes_ = [0, 1]
        return self

    def predict(self, X):
        # Always predict 0
        return np.zeros(len(X), dtype=int)

# --- New Training Functions (inspired by train_simple_classifier) ---

def train_random_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    """Trains' the random classifier."""
    logging.info("Start 'training' a random classifier")
    # Note: Preprocessing might not be strictly necessary for these naive models,
    # but we keep it for consistency with the existing workflow.
    # If the target column exists, it needs to be removed before 'fit' even if fit does nothing.
    if target_column in df.columns:
         df_processed = df.drop(columns=[target_column])
    else:
         df_processed = df.copy() # Work on a copy
    
    # df_processed = preprocess_dataframe(df_processed) # Optional: depends if preprocessing affects predict input shape
    
    # The naive models don't really use the data for fitting
    model = RandomClassifier()
    # model.fit(df_processed, None) # Fit doesn't use data here, can be skipped if fit is empty

    logging.info("Done 'training' a random classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="random_classifier",
        target_column=target_column,
        hyperparameters={}, # No hyperparameters
        description="Predicts 0 or 1 randomly.",
        performance_metrics={}, # Needs evaluation step to fill this
    )
    return model, metadata

def train_naive_positive_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    """Trains' the naive positive classifier."""
    logging.info("Start 'training' a naive positive classifier")
    if target_column in df.columns:
         df_processed = df.drop(columns=[target_column])
    else:
         df_processed = df.copy() 
         
    df_processed = preprocess_dataframe(df_processed)

    model = NaivePositiveClassifier()

    logging.info("Done 'training' a naive positive classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="naive_positive",
        target_column=target_column,
        hyperparameters={},
        description="Always predicts 1.",
        performance_metrics={},
    )
    return model, metadata

def train_naive_negative_classifier(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    """Trains' the naive negative classifier."""
    logging.info("Start 'training' a naive negative classifier")
    if target_column in df.columns:
         df_processed = df.drop(columns=[target_column])
    else:
         df_processed = df.copy()

    df_processed = preprocess_dataframe(df_processed) 

    model = NaiveNegativeClassifier()

    logging.info("Done 'training' a naive negative classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="naive_negative",
        target_column=target_column,
        hyperparameters={},
        description="Always predicts 0.",
        performance_metrics={},
    )
    return model, metadata


# --- Add this function to dsba/model_training.py ---

# Make sure this import is at the top of the file with other sklearn imports
from sklearn.ensemble import RandomForestClassifier
# Ensure pandas, logging, datetime, ClassifierMixin, ClassifierMetadata,
# preprocess_dataframe, split_features_and_target are also imported at the top

