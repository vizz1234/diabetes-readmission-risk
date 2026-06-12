import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Add current directory to path to allow import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from icd9_mapping import add_icd9_features

@dataclass
class PreprocessingArtifact:
    numeric_impute_values: dict
    categorical_encoders: dict
    scaler: StandardScaler
    feature_columns: list
    version: str

# List of all 23 medications
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]

NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "total_visits",
    "medication_change_count"
]

CATEGORICAL_COLS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "change",
    "diabetesMed"
] + MEDICATIONS

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to fill '?' with NaN, apply ICD-9 mapping, and compute derived features."""
    df = df.replace("?", np.nan)
    
    # 1. ICD-9 categories mapping
    df = add_icd9_features(df)
    
    # 2. Derived features
    # total_visits = number_outpatient + number_emergency + number_inpatient
    df["total_visits"] = (
        df["number_outpatient"].fillna(0) +
        df["number_emergency"].fillna(0) +
        df["number_inpatient"].fillna(0)
    )
    
    # medication_change_count from the *change columns (dosage Up/Down in the 23 medication columns)
    df["medication_change_count"] = df[MEDICATIONS].isin(["Up", "Down"]).sum(axis=1)
    
    # Force ID columns to string to prevent numeric scaling and allow proper categorical encoding
    for id_col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)
            
    return df

def fit_preprocessing(train_df: pd.DataFrame, version: str = "v1") -> PreprocessingArtifact:
    """
    Fits preprocessing steps on the training set.
    """
    train_df = train_df.copy()
    
    # Prepare dataframe features and mapping
    prepared_df = _prepare_df(train_df)
    
    # Calculate numeric imputes (medians)
    numeric_impute_values = {}
    for col in NUMERIC_COLS:
        numeric_impute_values[col] = prepared_df[col].median()
        
    # Fit Categorical Encoders
    categorical_encoders = {}
    for col in CATEGORICAL_COLS:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        # Fill NaN with 'missing' or 'Unknown' before fitting
        col_data = prepared_df[col].astype(str).fillna("missing").values.reshape(-1, 1)
        encoder.fit(col_data)
        categorical_encoders[col] = encoder
        
    # Impute numeric columns for scaling
    imputed_numeric = prepared_df[NUMERIC_COLS].copy()
    for col in NUMERIC_COLS:
        imputed_numeric[col] = imputed_numeric[col].fillna(numeric_impute_values[col])
        
    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(imputed_numeric)
    
    # Feature columns in final output order
    feature_columns = NUMERIC_COLS + CATEGORICAL_COLS + ["has_diabetes_diag"]
    
    return PreprocessingArtifact(
        numeric_impute_values=numeric_impute_values,
        categorical_encoders=categorical_encoders,
        scaler=scaler,
        feature_columns=feature_columns,
        version=version
    )

def apply_preprocessing(df: pd.DataFrame, artifact: PreprocessingArtifact):
    """
    Transforms df using artifact's fitted values. Never refits.
    Returns X, y (y is None if target column absent).
    """
    df = df.copy()
    prepared_df = _prepare_df(df)
    
    # Extract target if exists
    y = None
    if "readmitted" in prepared_df.columns:
        y = (prepared_df["readmitted"] == "<30").astype(int).values
        
    # Impute numeric columns
    imputed_numeric = prepared_df[NUMERIC_COLS].copy()
    for col in NUMERIC_COLS:
        imputed_numeric[col] = imputed_numeric[col].fillna(artifact.numeric_impute_values.get(col, 0.0))
        
    # Scale numeric columns
    scaled_numeric = artifact.scaler.transform(imputed_numeric)
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=NUMERIC_COLS, index=df.index)
    
    # Encode categorical columns
    encoded_cat_df = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_COLS:
        col_data = prepared_df[col].astype(str).fillna("missing").values.reshape(-1, 1)
        encoder = artifact.categorical_encoders[col]
        encoded_cat_df[col] = encoder.transform(col_data).flatten()
        
    # Build final DataFrame
    X = pd.concat([scaled_numeric_df, encoded_cat_df], axis=1)
    
    # Add has_diabetes_diag flag as-is
    X["has_diabetes_diag"] = prepared_df["has_diabetes_diag"].values
    
    # Reorder columns to match feature_columns order
    X = X[artifact.feature_columns]
    
    return X, y

def save_artifact(artifact: PreprocessingArtifact, version: str, path: str = "artifacts"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(artifact, os.path.join(path, f"preprocessing_{version}.joblib"))

def load_artifact(version: str, path: str = "artifacts") -> PreprocessingArtifact:
    return joblib.load(os.path.join(path, f"preprocessing_{version}.joblib"))
