import pandas as pd

def icd9_to_category(code) -> str:
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "missing"
    code = str(code).strip()
    if not code or code == "?":
        return "missing"
    if code.startswith(("V", "E")):
        return "other"
    try:
        n = float(code)
    except ValueError:
        return "other"
    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 250 <= n < 251:
        return "diabetes"
    if 800 <= n <= 999:
        return "injury"
    if 710 <= n <= 739:
        return "musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "genitourinary"
    if 140 <= n <= 239:
        return "neoplasms"
    return "other"

def add_icd9_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies icd9_to_category to diag_1, diag_2, and diag_3 columns.
    Creates a has_diabetes_diag indicator column indicating if any of diag_1, diag_2, diag_3 map to 'diabetes'.
    """
    df = df.copy()
    
    # Ensure missing diag columns are filled
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_category"] = df[col].apply(icd9_to_category)
        else:
            df[f"{col}_category"] = "missing"
            
    # Calculate has_diabetes_diag flag
    df["has_diabetes_diag"] = (
        (df["diag_1_category"] == "diabetes") |
        (df["diag_2_category"] == "diabetes") |
        (df["diag_3_category"] == "diabetes")
    ).astype(int)
    
    return df
