import os
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ShapExplainer:
    def __init__(self, model, preprocessing_artifact, X_train_sample=None):
        """
        model: Trained classifier model.
        preprocessing_artifact: PreprocessingArtifact used to process the inputs.
        X_train_sample: Optional sample dataframe or numpy array of preprocessed training data.
        """
        self.model = model
        self.prep = preprocessing_artifact
        self.feature_names = preprocessing_artifact.feature_columns
        
        # Determine appropriate explainer type
        # For tree classifiers, we prefer TreeExplainer (supports RF, LightGBM)
        # For Logistic Regression, we can use LinearExplainer
        if isinstance(model, LogisticRegression):
            if X_train_sample is not None:
                self.explainer = shap.LinearExplainer(model, X_train_sample)
            else:
                # Use a dummy background dataset
                dummy_background = np.zeros((1, len(self.feature_names)))
                self.explainer = shap.LinearExplainer(model, dummy_background)
        elif isinstance(model, RandomForestClassifier) or type(model).__name__ == "LGBMClassifier":
            self.explainer = shap.TreeExplainer(model)
        else:
            # Fallback to general Explainer
            if X_train_sample is not None:
                self.explainer = shap.Explainer(model, X_train_sample)
            else:
                self.explainer = shap.Explainer(model)

    def _extract_shap_values(self, explanation):
        """
        Extracts raw SHAP values for the positive class (class 1) from the explanation object/array.
        """
        # SHAP explanation objects or arrays can be shaped differently based on explainer types.
        if hasattr(explanation, "values"):
            vals = explanation.values
        else:
            vals = explanation
            
        # If vals is a list of arrays (e.g. TreeExplainer on RandomForestClassifier), pick positive class
        if isinstance(vals, list):
            return vals[1]
            
        # If vals is 3D (n_samples, n_features, 2), pick class 1
        if len(vals.shape) == 3 and vals.shape[2] == 2:
            return vals[:, :, 1]
            
        return vals

    def get_global_importance(self, X: pd.DataFrame) -> dict:
        """
        Returns a dictionary of feature names mapped to their mean absolute SHAP value.
        """
        # Ensure correct features order
        X_eval = X[self.feature_names]
        
        explanation = self.explainer(X_eval)
        vals = self._extract_shap_values(explanation)
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(vals).mean(axis=0)
        
        importance = {}
        for feat, val in zip(self.feature_names, mean_abs_shap):
            importance[feat] = float(val)
            
        # Sort features by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
        return sorted_importance

    def get_top_factors(self, single_row: pd.DataFrame, n: int = 3) -> list[dict]:
        """
        Returns the top n features with their actual values and SHAP values for a single prediction.
        single_row: pd.DataFrame with 1 row of preprocessed features.
        """
        # Ensure correct features order and shape
        X_eval = single_row[self.feature_names]
        
        explanation = self.explainer(X_eval)
        vals = self._extract_shap_values(explanation)
        
        # Flatten SHAP values for single sample
        if len(vals.shape) > 1:
            vals = vals[0]
            
        raw_vals = X_eval.iloc[0].values
        
        factors = []
        for feat, val, shap_val in zip(self.feature_names, raw_vals, vals):
            factors.append({
                "feature": feat,
                "value": float(val),
                "shap_value": float(shap_val)
            })
            
        # Sort by absolute SHAP value
        sorted_factors = sorted(factors, key=lambda x: abs(x["shap_value"]), reverse=True)
        return sorted_factors[:n]
