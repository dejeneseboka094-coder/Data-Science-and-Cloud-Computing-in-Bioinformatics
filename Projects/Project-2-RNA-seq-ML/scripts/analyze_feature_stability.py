#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# CONFIGURATION
# ============================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE, "data", "processed")
RESULTS_DIR = os.path.join(BASE, "results", "ml")

os.makedirs(RESULTS_DIR, exist_ok=True)

EXPR_FILE = os.path.join(
    DATA_DIR,
    "paired_log_cpm.tsv"
)

META_FILE = os.path.join(
    DATA_DIR,
    "ml_metadata.tsv"
)

RANDOM_STATE = 42
N_SPLITS = 5
K_GENES = 500


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE-SELECTION STABILITY ANALYSIS")
print("=" * 70)

X = pd.read_csv(
    EXPR_FILE,
    sep="\t",
    index_col=0
)

X.index = X.index.astype(str)

X = X.T

meta = pd.read_csv(
    META_FILE,
    sep="\t"
)

if not all(X.index == meta["sample_id"]):
    raise ValueError(
        "Expression matrix and metadata do not match."
    )

y = (
    meta["class"]
    .map({"Normal": 0, "Tumor": 1})
    .astype(int)
    .values
)

groups = meta["patient_id"].values


# ============================================================
# RECREATE SAME PATIENT-LEVEL TRAINING SET
# ============================================================

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=8,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
y_train = y[train_idx]
groups_train = groups[train_idx]

print("\nTraining samples:", len(X_train))
print("Training patients:", len(np.unique(groups_train)))
print("Genes available:", X_train.shape[1])


# ============================================================
# STRATIFIED GROUP CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Store selected genes from every fold
selection_records = []

# Store rank information
rank_records = []


# ============================================================
# FEATURE SELECTION WITHIN EACH CV FOLD
# ============================================================

for fold, (cv_train_idx, cv_val_idx) in enumerate(
    cv.split(
        X_train,
        y_train,
        groups_train
    ),
    start=1
):

    print(f"\nProcessing CV fold {fold}...")

    X_cv_train = X_train.iloc[cv_train_idx]
    y_cv_train = y_train[cv_train_idx]

    selector = SelectKBest(
        score_func=f_classif,
        k=K_GENES
    )

    selector.fit(
        X_cv_train,
        y_cv_train
    )

    selected_mask = selector.get_support()

    selected_genes = X_train.columns[selected_mask]

    scores = selector.scores_[selected_mask]

    # Sort by F-score
    order = np.argsort(scores)[::-1]

    selected_genes = selected_genes[order]
    scores = scores[order]

    print(
        f"  Selected {len(selected_genes)} genes"
    )

    for rank, (gene, score) in enumerate(
        zip(selected_genes, scores),
        start=1
    ):

        selection_records.append({
            "fold": fold,
            "GeneID": str(gene),
            "F_score": float(score),
            "rank": rank
        })

        rank_records.append({
            "fold": fold,
            "GeneID": str(gene),
            "rank": rank
        })


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

selection_df = pd.DataFrame(
    selection_records
)

rank_df = pd.DataFrame(
    rank_records
)


# ============================================================
# SAVE RAW FOLD SELECTIONS
# ============================================================

fold_file = os.path.join(
    RESULTS_DIR,
    "cv_feature_selection.tsv"
)

selection_df.to_csv(
    fold_file,
    sep="\t",
    index=False
)

print(
    f"\nSaved fold-level feature selection:"
)
print(fold_file)


# ============================================================
# CALCULATE STABILITY
# ============================================================

stability = (
    selection_df
    .groupby("GeneID")
    .agg(
        folds_selected=("fold", "nunique"),
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        mean_F_score=("F_score", "mean"),
        max_F_score=("F_score", "max")
    )
    .reset_index()
)

stability["selection_frequency"] = (
    stability["folds_selected"] / N_SPLITS
)

# Rank primarily by number of folds selected,
# then by mean rank.
stability = stability.sort_values(
    [
        "folds_selected",
        "mean_rank",
        "mean_F_score"
    ],
    ascending=[
        False,
        True,
        False
    ]
)


# ============================================================
# SAVE STABILITY RESULTS
# ============================================================

stability_file = os.path.join(
    RESULTS_DIR,
    "feature_selection_stability.tsv"
)

stability.to_csv(
    stability_file,
    sep="\t",
    index=False
)

print(
    f"\nSaved stability results:"
)
print(stability_file)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION STABILITY")
print("=" * 70)

print(
    "\nGenes selected in all 5 CV folds:"
)

stable_5 = stability[
    stability["folds_selected"] == 5
]

print(
    "Number:",
    len(stable_5)
)

print(
    stable_5.head(30).to_string(
        index=False
    )
)


print(
    "\nGenes selected in at least 4/5 folds:"
)

stable_4 = stability[
    stability["folds_selected"] >= 4
]

print(
    "Number:",
    len(stable_4)
)

print(
    stable_4.head(30).to_string(
        index=False
    )
)


# ============================================================
# SAVE TOP STABLE GENES
# ============================================================

stable_file = os.path.join(
    RESULTS_DIR,
    "stable_genes.tsv"
)

stable_4.to_csv(
    stable_file,
    sep="\t",
    index=False
)

print(
    f"\nSaved:"
)
print(stable_file)


print("\nDone.")
