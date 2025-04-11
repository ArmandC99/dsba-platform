from pandas import DataFrame, Series
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# from sklearn.model_selection import train_test_split


def split_features_and_target(
    df: DataFrame, target_column: str
) -> tuple[DataFrame, Series]:
    """
    Splits a DataFrame into features and target, which is a common format used by machine learning libraries such as scikit-learn.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_dataframe(
    df: DataFrame, test_size: float = 0.2
) -> tuple[DataFrame, DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=42)


def preprocess_dataframe(df):
    """
    Prétraiter le DataFrame en encodant les colonnes catégoriques.
    
    Les algorithmes de machine learning traitent généralement des nombres.
    Ainsi, ce prétraitement va :
        - Pour un modèle LogisticRegression uniquement :
            * Gérer les valeurs manquantes dans "Embarked" et "Age"
            * Transformer la colonne "Embarked" en variables indicatrices (dummies)
            * Supprimer les colonnes "Name", "Ticket" et "Cabin"
        - Pour toutes les colonnes de type 'object', appliquer un LabelEncoder pour les convertir en nombres.
    
    Parameters:
        df (pandas.DataFrame): Le DataFrame à prétraiter.
        model: L'instance du modèle qui sera utilisé en aval.
        
    Returns:
        pandas.DataFrame: Le DataFrame prétraité.
    """

    # Encodage de toutes les colonnes de type 'object' avec LabelEncoder
    for column in df.select_dtypes(include=["object"]):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
    
    return df