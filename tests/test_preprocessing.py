import pytest
import numpy as np
import pandas as pd
from src.pipeline.preprocessing import fit_preprocessing, apply_preprocessing, PreprocessingArtifact, MEDICATIONS

def test_preprocessing_fitting_and_imputation():
    # Construct a small dummy training set
    train_data = {
        "encounter_id": [1, 2, 3],
        "patient_nbr": [100, 200, 300],
        "race": ["Caucasian", "AfricanAmerican", "?"],
        "gender": ["Female", "Male", "Female"],
        "age": ["[50-60)", "[60-70)", "[70-80)"],
        "weight": ["?", "?", "?"],
        "admission_type_id": [1, 2, 1],
        "discharge_disposition_id": [1, 1, 3],
        "admission_source_id": [7, 7, 1],
        "time_in_hospital": [2.0, 4.0, np.nan],  # Median of 2 and 4 should be 3
        "payer_code": ["MC", "?", "MD"],
        "medical_specialty": ["FamilyPractice", "InternalMedicine", "?"],
        "num_lab_procedures": [40, 50, 60],
        "num_procedures": [1, 2, 0],
        "num_medications": [10, 15, 20],
        "number_outpatient": [0, 1, 2],
        "number_emergency": [0, 0, 1],
        "number_inpatient": [1, 0, 0],
        "diag_1": ["250.01", "428", "599"],  # 250.01 is diabetes
        "diag_2": ["428", "250.02", "other"], # 250.02 is diabetes
        "diag_3": ["?", "428", "250.03"],     # 250.03 is diabetes
        "number_diagnoses": [5, 6, 7],
        "max_glu_serum": ["None", "None", "None"],
        "A1Cresult": ["None", "None", "None"],
        "change": ["No", "Ch", "No"],
        "diabetesMed": ["Yes", "Yes", "Yes"],
        "readmitted": ["<30", ">30", "NO"]
    }
    # Add medications
    for med in MEDICATIONS:
        if med == "insulin":
            train_data[med] = ["Steady", "Up", "No"]
        else:
            train_data[med] = ["No", "No", "No"]
            
    train_df = pd.DataFrame(train_data)
    
    # Fit preprocessing
    artifact = fit_preprocessing(train_df, version="test_v1")
    
    # Verify train imputation value for time_in_hospital is median (3.0)
    assert artifact.numeric_impute_values["time_in_hospital"] == 3.0
    
    # Run transform on training
    X_train, y_train = apply_preprocessing(train_df, artifact)
    
    # Verify shapes and types
    assert len(X_train) == 3
    assert list(y_train) == [1, 0, 0]
    
    # Verify that 'has_diabetes_diag' is created and correct
    # Row 0: diag_1="250.01" (diabetes) -> 1
    # Row 1: diag_2="250.02" (diabetes) -> 1
    # Row 2: diag_3="250.03" (diabetes) -> 1
    assert list(X_train["has_diabetes_diag"]) == [1, 1, 1]
    
    # Let's create a test set with missing time_in_hospital to check if it gets imputed using train median (3.0)
    test_data = train_data.copy()
    test_data["time_in_hospital"] = [np.nan, np.nan, np.nan]
    test_df = pd.DataFrame(test_data)
    
    X_test, _ = apply_preprocessing(test_df, artifact)
    
    # Reconstruct scaled time_in_hospital for 3.0
    # Mean of [2, 4, 3] is 3.0, std is sqrt((1+1+0)/3) = 0.816497
    # So 3.0 scaled should be 0.0
    assert np.allclose(X_test["time_in_hospital"], 0.0, atol=1e-5)


def test_preprocessing_unknown_categories():
    train_data = {
        "encounter_id": [1, 2],
        "patient_nbr": [100, 200],
        "race": ["Caucasian", "AfricanAmerican"],
        "gender": ["Female", "Male"],
        "age": ["[50-60)", "[60-70)"],
        "weight": ["?", "?"],
        "admission_type_id": [1, 2],
        "discharge_disposition_id": [1, 1],
        "admission_source_id": [7, 7],
        "time_in_hospital": [2.0, 4.0],
        "payer_code": ["MC", "?"],
        "medical_specialty": ["FamilyPractice", "InternalMedicine"],
        "num_lab_procedures": [40, 50],
        "num_procedures": [1, 2],
        "num_medications": [10, 15],
        "number_outpatient": [0, 1],
        "number_emergency": [0, 0],
        "number_inpatient": [1, 0],
        "diag_1": ["250.01", "428"],
        "diag_2": ["428", "250.02"],
        "diag_3": ["?", "428"],
        "number_diagnoses": [5, 6],
        "max_glu_serum": ["None", "None"],
        "A1Cresult": ["None", "None"],
        "change": ["No", "Ch"],
        "diabetesMed": ["Yes", "Yes"],
        "readmitted": ["<30", ">30"]
    }
    for med in MEDICATIONS:
        train_data[med] = ["No", "No"]
        
    train_df = pd.DataFrame(train_data)
    artifact = fit_preprocessing(train_df, version="test_v2")
    
    # Create test set with an UNKNOWN race and UNKNOWN medical_specialty
    test_data = train_data.copy()
    test_data["race"] = ["Asian", "Hispanic"]
    test_data["medical_specialty"] = ["Cardiology", "Pediatrics"]
    test_df = pd.DataFrame(test_data)
    
    # Verify transforming doesn't raise error and maps unknown values to -1
    X_test, _ = apply_preprocessing(test_df, artifact)
    
    assert X_test["race"].iloc[0] == -1
    assert X_test["race"].iloc[1] == -1
    assert X_test["medical_specialty"].iloc[0] == -1
    assert X_test["medical_specialty"].iloc[1] == -1
