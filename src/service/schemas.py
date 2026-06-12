from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PatientFeatures(BaseModel):
    # Patient demographic details
    race: Optional[str] = Field(default="Unknown", description="Race of the patient")
    gender: Optional[str] = Field(default="Unknown", description="Gender of the patient")
    age: Optional[str] = Field(default="Unknown", description="Age category, e.g., '[50-60)'")
    
    # Encounter and admission details
    admission_type_id: Optional[int] = Field(default=1, description="Admission type ID")
    discharge_disposition_id: Optional[int] = Field(default=1, description="Discharge disposition ID")
    admission_source_id: Optional[int] = Field(default=7, description="Admission source ID")
    time_in_hospital: Optional[int] = Field(default=1, description="Time in hospital in days")
    medical_specialty: Optional[str] = Field(default="Unknown", description="Medical specialty of admitting physician")
    
    # Clinical counts
    num_lab_procedures: Optional[int] = Field(default=0, description="Number of lab procedures performed")
    num_procedures: Optional[int] = Field(default=0, description="Number of non-lab procedures performed")
    num_medications: Optional[int] = Field(default=0, description="Number of distinct medications prescribed")
    number_outpatient: Optional[int] = Field(default=0, description="Number of outpatient visits in the preceding year")
    number_emergency: Optional[int] = Field(default=0, description="Number of emergency visits in the preceding year")
    number_inpatient: Optional[int] = Field(default=0, description="Number of inpatient visits in the preceding year")
    
    # Diagnosis codes (ICD-9)
    diag_1: Optional[str] = Field(default="?", description="Primary diagnosis code")
    diag_2: Optional[str] = Field(default="?", description="Secondary diagnosis code")
    diag_3: Optional[str] = Field(default="?", description="Additional diagnosis code")
    number_diagnoses: Optional[int] = Field(default=0, description="Number of diagnoses entered into the system")
    
    # Testing results
    max_glu_serum: Optional[str] = Field(default="None", description="Glucose serum test result")
    A1Cresult: Optional[str] = Field(default="None", description="A1c test result")
    
    # Medication changes
    change: Optional[str] = Field(default="No", description="Indicates if there was a change in diabetic medications")
    diabetesMed: Optional[str] = Field(default="No", description="Indicates if any diabetic medication was prescribed")
    
    # 23 specific medications
    metformin: Optional[str] = "No"
    repaglinide: Optional[str] = "No"
    nateglinide: Optional[str] = "No"
    chlorpropamide: Optional[str] = "No"
    glimepiride: Optional[str] = "No"
    acetohexamide: Optional[str] = "No"
    glipizide: Optional[str] = "No"
    glyburide: Optional[str] = "No"
    tolbutamide: Optional[str] = "No"
    pioglitazone: Optional[str] = "No"
    rosiglitazone: Optional[str] = "No"
    acarbose: Optional[str] = "No"
    miglitol: Optional[str] = "No"
    troglitazone: Optional[str] = "No"
    tolazamide: Optional[str] = "No"
    examide: Optional[str] = "No"
    citoglipton: Optional[str] = "No"
    insulin: Optional[str] = "No"
    glyburide_metformin: Optional[str] = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: Optional[str] = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Optional[str] = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Optional[str] = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: Optional[str] = Field(default="No", alias="metformin-pioglitazone")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[50-60)",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "time_in_hospital": 3,
                "medical_specialty": "FamilyPractice",
                "num_lab_procedures": 40,
                "num_procedures": 1,
                "num_medications": 10,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 1,
                "diag_1": "250.01",
                "diag_2": "428",
                "diag_3": "?",
                "number_diagnoses": 5,
                "max_glu_serum": "None",
                "A1Cresult": "None",
                "change": "No",
                "diabetesMed": "Yes",
                "insulin": "Steady"
            }
        }

class PredictResponse(BaseModel):
    risk_score: float = Field(..., description="Predicted readmission probability")
    risk_band: str = Field(..., description="Risk tier: 'low', 'medium', or 'high'")
    top_factors: List[Dict[str, Any]] = Field(..., description="Top local SHAP feature explanations")
    model_version: str = Field(..., description="MLflow run/model version identifier")
    preprocessing_version: str = Field(..., description="Preprocessing version identifier")

class BatchPredictRequest(BaseModel):
    patients: List[PatientFeatures]

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    batch_id: str
