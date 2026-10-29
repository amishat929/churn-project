"""Train, compare and save the churn models.

Run:  python train.py
Outputs go to models/ (model + metrics) and reports/ (PNG charts).
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import business, explain, theme
from src.preprocess import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, load_model_frame

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def make_pipeline(clf) -> ImbPipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return ImbPipeline([
        ("prep", prep),
        ("smote", SMOTE(random_state=RANDOM_STATE)),   # only applied while fitting
        ("clf", clf),
    ])


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------- plots
def plot_roc(curves: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=theme.SURFACE)
    theme.style_axes(ax)
    ax.grid(axis="x", color=theme.GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=theme.MUTED, label="Random guess")
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, color=theme.MODEL_COLORS[name], label=f"{name} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate", color=theme.INK_2)
    ax.set_ylabel("True positive rate", color=theme.INK_2)
    ax.set_title("ROC curves on the held-out test set", loc="left", color=theme.INK, fontsize=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, threshold: float, path: Path):
    cmap = LinearSegmentedColormap.from_list("seq", [theme.SEQ_BLUE[0], theme.SEQ_BLUE[-1]])
    fig, ax = plt.subplots(figsize=(5, 4.4), facecolor=theme.SURFACE)
    ax.imshow(cm, cmap=cmap)
    labels = ["Stayed", "Churned"]
    ax.set_xticks([0, 1], labels=labels, color=theme.INK_2)
    ax.set_yticks([0, 1], labels=labels, color=theme.INK_2)
    ax.set_xlabel("Predicted", color=theme.INK_2)
    ax.set_ylabel("Actual", color=theme.INK_2)
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.5
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="#ffffff" if dark else theme.INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})", loc="left", color=theme.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_benefit(thresholds, oof_curve, test_curve, best_t, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=theme.SURFACE)
    theme.style_axes(ax)
    ax.plot(thresholds, oof_curve, lw=2, color=theme.STAY, label="Training data (cross-validated)")
    ax.plot(thresholds, test_curve, lw=2, color=theme.CHURN, label="Test data")
    ax.axvline(best_t, color=theme.MUTED, lw=1, ls="--")
    ax.text(best_t + 0.01, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
            f"chosen threshold {best_t:.2f}", color=theme.INK_2, fontsize=9)
    ax.set_xlabel("Probability threshold for sending a retention offer", color=theme.INK_2)
    ax.set_ylabel("Net benefit per 1,000 customers", color=theme.INK_2)
    ax.set_title("Where to draw the line: profit vs. threshold", loc="left", color=theme.INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, path: Path, top=12):
    top_df = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=theme.SURFACE)
    theme.style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme.GRID, linewidth=0.8)
    ax.barh(top_df["feature"], top_df["mean_abs_shap"], color=theme.STAY, height=0.6)
    ax.set_xlabel("Average impact on churn prediction (mean |SHAP|)", color=theme.INK_2)
    ax.set_title("What drives churn the most", loc="left", color=theme.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_model_frame()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Customers: {len(df):,} | churn rate {y.mean():.1%} | train {len(X_train):,} / test {len(X_test):,}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, curves, fitted = [], {}, {}

    for name, clf in candidate_models().items():
        pipe = make_pipeline(clf)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        curves[name] = (fpr, tpr, auc)
        fitted[name] = pipe
        rows.append({
            "Model": name,
            "CV ROC-AUC": round(cv_auc.mean(), 4),
            "CV std": round(cv_auc.std(), 4),
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred), 4),
            "Recall": round(recall_score(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4),
            "Test ROC-AUC": round(auc, 4),
        })
        print(f"  {name:<20} CV AUC {cv_auc.mean():.4f} | test AUC {auc:.4f} | "
              f"recall {rows[-1]['Recall']:.3f} | F1 {rows[-1]['F1']:.3f}")

    comparison = pd.DataFrame(rows)
    best_name = comparison.sort_values("CV ROC-AUC", ascending=False).iloc[0]["Model"]
    best = fitted[best_name]
    print(f"\nBest model by cross-validated ROC-AUC: {best_name}")

    # ---- choose the business threshold on out-of-fold predictions (no test leakage)
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    oof = cross_val_predict(make_pipeline(clone(candidate_models()[best_name])),
                            X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    proba_test = best.predict_proba(X_test)[:, 1]
    oof_curve = business.benefit_curve(y_train, oof, thresholds)
    test_curve = business.benefit_curve(y_test, proba_test, thresholds)
    best_t = float(thresholds[int(np.argmax(oof_curve))])

    pred_t = (proba_test >= best_t).astype(int)
    cm = confusion_matrix(y_test, pred_t)
    contacted = int(pred_t.sum())
    net_model = business.net_benefit(y_test, proba_test, best_t)
    net_all = business.contact_everyone(y_test)
    print(f"Chosen threshold {best_t:.2f}: contacts {contacted} of {len(y_test)} customers, "
          f"net benefit {net_model:,.0f} vs {net_all:,.0f} for 'offer to everyone'")

    # ---- SHAP (global importance)
    prep, clf = best.named_steps["prep"], best.named_steps["clf"]
    feat_names = [explain.readable(n) for n in prep.get_feature_names_out()]
    Xt_test = pd.DataFrame(prep.transform(X_test), columns=feat_names)
    Xt_bg = prep.transform(X_train.sample(200, random_state=RANDOM_STATE))
    explainer = explain.build_explainer(clf, Xt_bg)
    sv = explain.positive_class_values(explainer, Xt_test.values)
    importance = (pd.DataFrame({"feature": feat_names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    try:
        plt.figure()
        shap.summary_plot(sv, Xt_test, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:  # the beeswarm is a nice-to-have
        print("  (skipped SHAP beeswarm:", exc, ")")

    # ---- save charts + artifacts
    plot_roc(curves, REPORTS_DIR / "roc_curves.png")
    plot_confusion(cm, best_t, REPORTS_DIR / "confusion_matrix.png")
    plot_benefit(thresholds, oof_curve / len(y_train) * 1000, test_curve / len(y_test) * 1000,
                 best_t, REPORTS_DIR / "net_benefit.png")
    plot_importance(importance, REPORTS_DIR / "feature_importance.png")

    joblib.dump(best, MODELS_DIR / "churn_model.joblib")
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    pd.DataFrame({"churned": y_test.values, "probability": proba_test}).to_csv(
        MODELS_DIR / "test_predictions.csv", index=False)

    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "best_model": best_name,
        "threshold": best_t,
        "n_customers": int(len(df)),
        "churn_rate": float(y.mean()),
        "test_size": int(len(y_test)),
        "at_threshold": {
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "f1": float(f1_score(y_test, pred_t)),
            "contacted": contacted,
            "true_positives": int(tp), "false_positives": int(fp),
            "false_negatives": int(fn), "true_negatives": int(tn),
        },
        "business": {
            "offer_cost": business.OFFER_COST,
            "customer_value": business.CUSTOMER_VALUE,
            "success_rate": business.SUCCESS_RATE,
            "net_benefit_model_test": net_model,
            "net_benefit_offer_everyone_test": net_all,
        },
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved model, metrics and charts.")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
