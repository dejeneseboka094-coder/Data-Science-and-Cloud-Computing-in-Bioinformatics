#!/usr/bin/env python3

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE, "data", "processed")
RESULTS_DIR = os.path.join(BASE, "results", "ml")
MODEL_DIR = os.path.join(BASE, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

EXPR_FILE = os.path.join(DATA_DIR, "paired_log_cpm.tsv")
META_FILE = os.path.join(DATA_DIR, "ml_metadata.tsv")

RANDOM_STATE = 42
N_SPLITS = 5

K_VALUES = [100, 250, 500, 1000]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, y_prob
        )
    }


def make_rf():
    """Create Random Forest model."""

    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def make_xgb():
    """Create XGBoost model."""

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RNA-seq MACHINE LEARNING CLASSIFICATION")
print("=" * 70)

print("\nLoading expression matrix...")

expr = pd.read_csv(
    EXPR_FILE,
    sep="\t",
    index_col=0
)

expr.index = expr.index.astype(str)

print(f"Expression matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")

print("\nLoading metadata...")

meta = pd.read_csv(
    META_FILE,
    sep="\t"
)

print(f"Metadata rows: {len(meta)}")

# ------------------------------------------------------------
# Ensure sample order matches
# ------------------------------------------------------------

if not all(expr.columns == meta["sample_id"]):
    raise ValueError(
        "Expression matrix and metadata sample order do not match."
    )

print("Sample order verified: PASS")


# ============================================================
# DEFINE TARGET
# ============================================================

# Tumor = 1
# Normal = 0

y = (
    meta["class"]
    .map({"Normal": 0, "Tumor": 1})
    .astype(int)
    .values
)

groups = meta["patient_id"].values

X = expr.T

print("\nClassification target:")
print("Normal =", np.sum(y == 0))
print("Tumor  =", np.sum(y == 1))

print(f"Patients: {len(np.unique(groups))}")


# ============================================================
# PATIENT-LEVEL TRAIN/TEST SPLIT
# ============================================================

# These were generated previously using random_state=42.
# Reconstruct the same split from patient IDs.

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=8,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y[train_idx]
y_test = y[test_idx]

groups_train = groups[train_idx]
groups_test = groups[test_idx]

print("\nPatient-level split:")
print(f"Training samples: {len(train_idx)}")
print(f"Testing samples : {len(test_idx)}")
print(f"Training patients: {len(np.unique(groups_train))}")
print(f"Testing patients : {len(np.unique(groups_test))}")

overlap = set(groups_train) & set(groups_test)

if overlap:
    raise ValueError(
        f"Patient leakage detected: {overlap}"
    )

print("Patient leakage check: PASS")


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

all_cv_results = []

print("\n" + "=" * 70)
print("NESTED FEATURE SELECTION + CROSS-VALIDATION")
print("=" * 70)

for k in K_VALUES:

    print(f"\n{'-' * 70}")
    print(f"FEATURE SET: TOP {k} GENES")
    print(f"{'-' * 70}")

    for model_name in ["RandomForest", "XGBoost"]:

        fold_results = []

        print(f"\nModel: {model_name}")

        for fold, (cv_train_idx, cv_val_idx) in enumerate(
            cv.split(X_train, y_train, groups_train),
            start=1
        ):

            X_cv_train = X_train.iloc[cv_train_idx]
            X_cv_val = X_train.iloc[cv_val_idx]

            y_cv_train = y_train[cv_train_idx]
            y_cv_val = y_train[cv_val_idx]

            # ------------------------------------------------
            # IMPORTANT:
            # Feature selection is fitted ONLY on CV training
            # data. This prevents feature-selection leakage.
            # ------------------------------------------------

            selector = SelectKBest(
                score_func=f_classif,
                k=k
            )

            X_cv_train_selected = selector.fit_transform(
                X_cv_train,
                y_cv_train
            )

            X_cv_val_selected = selector.transform(
                X_cv_val
            )

            # ------------------------------------------------
            # Create model
            # ------------------------------------------------

            if model_name == "RandomForest":
                model = make_rf()
            else:
                model = make_xgb()

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model.fit(
                X_cv_train_selected,
                y_cv_train
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            y_pred = model.predict(
                X_cv_val_selected
            )

            y_prob = model.predict_proba(
                X_cv_val_selected
            )[:, 1]

            metrics = calculate_metrics(
                y_cv_val,
                y_pred,
                y_prob
            )

            metrics.update({
                "model": model_name,
                "k_genes": k,
                "fold": fold
            })

            fold_results.append(metrics)
            all_cv_results.append(metrics)

            print(
                f"  Fold {fold}: "
                f"Accuracy={metrics['accuracy']:.3f}, "
                f"F1={metrics['f1']:.3f}, "
                f"ROC-AUC={metrics['roc_auc']:.3f}"
            )

        fold_df = pd.DataFrame(fold_results)

        print(
            f"  Mean: "
            f"Accuracy={fold_df['accuracy'].mean():.3f}, "
            f"F1={fold_df['f1'].mean():.3f}, "
            f"ROC-AUC={fold_df['roc_auc'].mean():.3f}"
        )


# ============================================================
# SAVE CV RESULTS
# ============================================================

cv_df = pd.DataFrame(all_cv_results)

cv_file = os.path.join(
    RESULTS_DIR,
    "cv_fold_results.tsv"
)

cv_df.to_csv(
    cv_file,
    sep="\t",
    index=False
)

print(f"\nSaved fold-level CV results:")
print(cv_file)


# ============================================================
# SUMMARIZE CV RESULTS
# ============================================================

summary = (
    cv_df
    .groupby(["model", "k_genes"])
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        f1_sd=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_sd=("roc_auc", "std")
    )
    .reset_index()
)

summary = summary.sort_values(
    ["roc_auc_mean", "f1_mean"],
    ascending=False
)

summary_file = os.path.join(
    RESULTS_DIR,
    "cv_summary.tsv"
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print(f"\nSaved:")
print(summary_file)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best = summary.iloc[0]

best_model_name = best["model"]
best_k = int(best["k_genes"])

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model: {best_model_name}")
print(f"Number of genes: {best_k}")
print(f"Mean CV ROC-AUC: {best['roc_auc_mean']:.4f}")
print(f"Mean CV F1: {best['f1_mean']:.4f}")


# ============================================================
# FINAL FEATURE SELECTION
# ============================================================

print("\nFitting final feature selector on all training samples...")

final_selector = SelectKBest(
    score_func=f_classif,
    k=best_k
)

X_train_final = final_selector.fit_transform(
    X_train,
    y_train
)

X_test_final = final_selector.transform(
    X_test
)

selected_mask = final_selector.get_support()

selected_genes = X_train.columns[selected_mask]

feature_scores = final_selector.scores_[selected_mask]
feature_pvalues = final_selector.pvalues_[selected_mask]

final_features = pd.DataFrame({
    "GeneID": selected_genes.astype(str),
    "F_score": feature_scores,
    "p_value": feature_pvalues
})

final_features = final_features.sort_values(
    "F_score",
    ascending=False
)

feature_file = os.path.join(
    RESULTS_DIR,
    "final_selected_genes.tsv"
)

final_features.to_csv(
    feature_file,
    sep="\t",
    index=False
)

print(
    f"Final selected genes: {len(selected_genes)}"
)

print(f"Saved: {feature_file}")


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining final model...")

if best_model_name == "RandomForest":
    final_model = make_rf()
else:
    final_model = make_xgb()

final_model.fit(
    X_train_final,
    y_train
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("INDEPENDENT TEST SET EVALUATION")
print("=" * 70)

test_pred = final_model.predict(
    X_test_final
)

test_prob = final_model.predict_proba(
    X_test_final
)[:, 1]

test_metrics = calculate_metrics(
    y_test,
    test_pred,
    test_prob
)

for metric, value in test_metrics.items():
    print(f"{metric:10s}: {value:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Normal", "Tumor"],
        zero_division=0
    )
)


# ============================================================
# SAVE TEST METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "model": best_model_name,
        "k_genes": best_k,
        **test_metrics
    }
])

metrics_file = os.path.join(
    RESULTS_DIR,
    "test_metrics.tsv"
)

metrics_df.to_csv(
    metrics_file,
    sep="\t",
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

pred_df = meta.iloc[test_idx].copy()

pred_df["true_class"] = np.where(
    y_test == 1,
    "Tumor",
    "Normal"
)

pred_df["predicted_class"] = np.where(
    test_pred == 1,
    "Tumor",
    "Normal"
)

pred_df["tumor_probability"] = test_prob

pred_file = os.path.join(
    RESULTS_DIR,
    "test_predictions.tsv"
)

pred_df.to_csv(
    pred_file,
    sep="\t",
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred
)

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Tumor"])
ax.set_yticklabels(["Normal", "Tumor"])

ax.set_xlabel("Predicted")
ax.set_ylabel("True")

ax.set_title(
    f"{best_model_name} - Test Confusion Matrix"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_prob
)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {test_metrics['roc_auc']:.3f}"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_title(
    f"{best_model_name} - Test ROC Curve"
)

ax.legend()

plt.tight_layout()

roc_file = os.path.join(
    RESULTS_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nExtracting model feature importance...")

if hasattr(final_model, "feature_importances_"):

    importance = final_model.feature_importances_

    importance_df = pd.DataFrame({
        "GeneID": selected_genes.astype(str),
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_file = os.path.join(
        RESULTS_DIR,
        "feature_importance.tsv"
    )

    importance_df.to_csv(
        importance_file,
        sep="\t",
        index=False
    )

    print(
        "\nTop 20 important genes:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

else:

    importance_df = pd.DataFrame()

    print(
        "Model does not expose feature_importances_."
    )


# ============================================================
# SAVE MODELS
# ============================================================

selector_file = os.path.join(
    MODEL_DIR,
    "final_feature_selector.joblib"
)

model_file = os.path.join(
    MODEL_DIR,
    "final_model.joblib"
)

joblib.dump(
    final_selector,
    selector_file
)

joblib.dump(
    final_model,
    model_file
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "model": best_model_name,
    "number_of_genes": best_k,
    "random_state": RANDOM_STATE,
    "cv_folds": N_SPLITS,
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "training_patients": int(len(np.unique(groups_train))),
    "testing_patients": int(len(np.unique(groups_test))),
    "cv_mean_roc_auc": float(best["roc_auc_mean"]),
    "cv_mean_f1": float(best["f1_mean"]),
    "test_roc_auc": float(test_metrics["roc_auc"]),
    "test_f1": float(test_metrics["f1"])
}

info_file = os.path.join(
    RESULTS_DIR,
    "model_info.json"
)

with open(info_file, "w") as f:
    json.dump(
        model_info,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ML PIPELINE COMPLETED")
print("=" * 70)

print("\nBest model:")
print(f"  {best_model_name}")

print("\nSelected genes:")
print(f"  {best_k}")

print("\nCross-validation ROC-AUC:")
print(f"  {best['roc_auc_mean']:.4f}")

print("\nIndependent test ROC-AUC:")
print(f"  {test_metrics['roc_auc']:.4f}")

print("\nIndependent test F1:")
print(f"  {test_metrics['f1']:.4f}")

print("\nOutput files:")
print(f"  {cv_file}")
print(f"  {summary_file}")
print(f"  {feature_file}")
print(f"  {metrics_file}")
print(f"  {pred_file}")
print(f"  {cm_file}")
print(f"  {roc_file}")

if not importance_df.empty:
    print(f"  {importance_file}")

print(f"  {selector_file}")
print(f"  {model_file}")
print(f"  {info_file}")

print("\nDone.")
