"""SHAP helpers shared by training (global plots) and the app (per-customer reasons)."""
import numpy as np
import shap

TREE_MODELS = ("XGBClassifier", "RandomForestClassifier")


def build_explainer(clf, background):
    """Pick the right SHAP explainer for the fitted classifier."""
    if type(clf).__name__ in TREE_MODELS:
        return shap.TreeExplainer(clf)
    return shap.LinearExplainer(clf, background)


def positive_class_values(explainer, X) -> np.ndarray:
    """Return an (n_rows, n_features) array of SHAP values for the churn class."""
    values = explainer.shap_values(X)
    if isinstance(values, list):          # older SHAP: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # newer SHAP: (rows, features, classes)
        values = values[:, :, 1]
    return values


def readable(feature_name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    name = feature_name.split("__", 1)[-1]
    for col in (
        "InternetService", "Contract", "PaymentMethod", "TenureGroup", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling", "SeniorCitizen",
        "PhoneService", "Dependents", "Partner", "gender",
    ):
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name
