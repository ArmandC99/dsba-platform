"""
This module is just a convenience to train a simple classifier.
Its presence is a bit artificial for the exercice and not required to develop an MLOps platform.
The MLOps course is not about model training.
"""

from dataclasses import dataclass
import logging
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression

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

def logistic_regression(
    df: pd.DataFrame, target_column: str, model_id: str
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a logistic classifier")
    df = preprocess_dataframe(df)
    X, y = split_features_and_target(df, target_column) 
    model = LogisticRegression(max_iter=1000)
    model.fit(X,y)
    
    logging.info("Done training a logistic classifier")
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="logistic",
        target_column=target_column,
        hyperparameters={"random_state": 42},
        description="",
        performance_metrics={},
    )
    return model, metadata 
