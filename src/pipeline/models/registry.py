MODEL_GRID = {
    "logreg": {
        "estimator": "LogisticRegression",
        "fixed_params": {"class_weight": "balanced", "max_iter": 2000, "solver": "liblinear"},
        "grid": {"C": [1.0], "penalty": ["l2"]},
    },
    "random_forest": {
        "estimator": "RandomForestClassifier",
        "fixed_params": {"class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        "grid": {"n_estimators": [100], "max_depth": [8], "min_samples_leaf": [10]},
    },
    "lightgbm": {
        "estimator": "LGBMClassifier",
        "fixed_params": {"is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1},
        "grid": {"n_estimators": [100], "learning_rate": [0.1], "num_leaves": [15]},
    },
}
