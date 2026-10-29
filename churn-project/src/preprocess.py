"""Data loading, cleaning and feature engineering for the Telco churn project."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "telco_raw.csv"

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgChargePerMonth",
]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "TenureGroup",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Churn"


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and fix the known data-quality problems."""
    df = pd.read_csv(path)

    # TotalCharges has 11 blank strings (brand-new customers with tenure 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features used by the models."""
    df = df.copy()
    df["NumServices"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["AvgChargePerMonth"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, 1), df["MonthlyCharges"]
    )
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)
    return df


def load_model_frame() -> pd.DataFrame:
    return add_features(load_clean())
