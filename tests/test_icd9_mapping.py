import pytest
import pandas as pd
from src.pipeline.icd9_mapping import icd9_to_category, add_icd9_features

def test_icd9_to_category_mapping():
    # Circulatory: 390-459 or 785
    assert icd9_to_category("390") == "circulatory"
    assert icd9_to_category("428") == "circulatory"
    assert icd9_to_category("459") == "circulatory"
    assert icd9_to_category("785") == "circulatory"
    
    # Respiratory: 460-519 or 786
    assert icd9_to_category("460") == "respiratory"
    assert icd9_to_category("486") == "respiratory"
    assert icd9_to_category("519") == "respiratory"
    assert icd9_to_category("786") == "respiratory"
    
    # Digestive: 520-579 or 787
    assert icd9_to_category("520") == "digestive"
    assert icd9_to_category("577") == "digestive"
    assert icd9_to_category("787") == "digestive"
    
    # Diabetes: 250.xx
    assert icd9_to_category("250") == "diabetes"
    assert icd9_to_category("250.01") == "diabetes"
    assert icd9_to_category("250.9") == "diabetes"
    
    # Injury: 800-999
    assert icd9_to_category("800") == "injury"
    assert icd9_to_category("950") == "injury"
    assert icd9_to_category("999") == "injury"
    
    # Musculoskeletal: 710-739
    assert icd9_to_category("710") == "musculoskeletal"
    assert icd9_to_category("735") == "musculoskeletal"
    
    # Genitourinary: 580-629 or 788
    assert icd9_to_category("580") == "genitourinary"
    assert icd9_to_category("788") == "genitourinary"
    
    # Neoplasms: 140-239
    assert icd9_to_category("140") == "neoplasms"
    assert icd9_to_category("200") == "neoplasms"
    
    # Other / Missing cases
    assert icd9_to_category("V58") == "other"
    assert icd9_to_category("E849") == "other"
    assert icd9_to_category("?") == "missing"
    assert icd9_to_category(None) == "missing"

def test_add_icd9_features():
    df = pd.DataFrame({
        "diag_1": ["250", "428", "999"],
        "diag_2": ["428", "?", "250.01"],
        "diag_3": [None, "580", "other_invalid"]
    })
    
    out_df = add_icd9_features(df)
    
    # Check category columns
    assert out_df["diag_1_category"].iloc[0] == "diabetes"
    assert out_df["diag_2_category"].iloc[1] == "missing"
    assert out_df["diag_3_category"].iloc[0] == "missing"
    
    # Check has_diabetes_diag flag
    # Row 0: diag_1 is 250 -> 1
    # Row 1: diag_2 is ?, diag_1 is 428, diag_3 is 580 -> 0
    # Row 2: diag_2 is 250.01 -> 1
    assert list(out_df["has_diabetes_diag"]) == [1, 0, 1]
