#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# --------------------------------------------------
# Files
# --------------------------------------------------

EXPRESSION = Path("data/processed/paired_log_cpm.tsv")
METADATA = Path("data/processed/paired_metadata.tsv")
ANNOTATION = Path("results/ml/consensus_gene_annotation.tsv")

OUTPUT = Path("results/ml/consensus_gene_expression_validation.tsv")

# --------------------------------------------------
# Load data
# --------------------------------------------------

expr = pd.read_csv(EXPRESSION, sep="\t", index_col=0)
expr.index = expr.index.astype(str)

metadata = pd.read_csv(METADATA, sep="\t")
annotation = pd.read_csv(ANNOTATION, sep="\t")

print("=" * 70)
print("CONSENSUS GENE EXPRESSION VALIDATION")
print("=" * 70)

print(f"Expression matrix: {expr.shape}")
print(f"Metadata: {metadata.shape}")
print(f"Consensus genes: {len(annotation)}")
print()

# --------------------------------------------------
# Verify sample structure
# --------------------------------------------------

metadata = metadata.set_index("sample_id")

common_samples = [
    s for s in expr.columns
    if s in metadata.index
]

expr = expr[common_samples]
metadata = metadata.loc[common_samples]

print(f"Matched samples: {len(common_samples)}")

# --------------------------------------------------
# Validate each consensus gene
# --------------------------------------------------

results = []

for _, gene in annotation.iterrows():

    gene_id = str(gene["GeneID"])
    symbol = str(gene["Symbol"])

    # Gene IDs may be numeric in expression matrix
    matching_rows = [
        idx for idx in expr.index.astype(str)
        if idx == gene_id
    ]

    if not matching_rows:
        print(f"WARNING: Gene {gene_id} not found")
        continue

    row = matching_rows[0]

    values = expr.loc[row]

    normal = values[
        metadata["class"] == "Normal"
    ]

    tumor = values[
        metadata["class"] == "Tumor"
    ]

    # --------------------------------------------------
    # Match patient pairs
    # --------------------------------------------------

    normal_patients = metadata.loc[
        metadata["class"] == "Normal",
        "patient_id"
    ]

    tumor_patients = metadata.loc[
        metadata["class"] == "Tumor",
        "patient_id"
    ]

    normal_by_patient = pd.Series(
        normal.values,
        index=normal_patients.values
    )

    tumor_by_patient = pd.Series(
        tumor.values,
        index=tumor_patients.values
    )

    paired_patients = sorted(
        set(normal_by_patient.index)
        &
        set(tumor_by_patient.index)
    )

    normal_paired = normal_by_patient.loc[
        paired_patients
    ]

    tumor_paired = tumor_by_patient.loc[
        paired_patients
    ]

    differences = tumor_paired - normal_paired

    # --------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------

    try:
        statistic, pvalue = wilcoxon(
            tumor_paired,
            normal_paired,
            alternative="two-sided"
        )
    except ValueError:
        statistic = np.nan
        pvalue = 1.0

    # --------------------------------------------------
    # Expression direction
    # --------------------------------------------------

    mean_normal = normal_paired.mean()
    mean_tumor = tumor_paired.mean()

    median_normal = normal_paired.median()
    median_tumor = tumor_paired.median()

    mean_difference = differences.mean()
    median_difference = differences.median()

    if median_difference > 0:
        direction = "Up in Tumor"
    elif median_difference < 0:
        direction = "Down in Tumor"
    else:
        direction = "No difference"

    results.append({
        "GeneID": gene_id,
        "Symbol": symbol,
        "Description": gene["Description"],
        "Consensus_Rank": gene["consensus_rank"],
        "RF_Importance": gene["importance"],
        "N_Pairs": len(paired_patients),
        "Mean_Normal": mean_normal,
        "Mean_Tumor": mean_tumor,
        "Median_Normal": median_normal,
        "Median_Tumor": median_tumor,
        "Mean_Paired_Difference": mean_difference,
        "Median_Paired_Difference": median_difference,
        "Wilcoxon_Statistic": statistic,
        "P_Value": pvalue,
        "Direction": direction
    })

# --------------------------------------------------
# Multiple testing correction
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df["FDR"] = multipletests(
    results_df["P_Value"],
    method="fdr_bh"
)[1]

results_df["Significant_FDR_0.05"] = (
    results_df["FDR"] < 0.05
)

# Sort by FDR
results_df = results_df.sort_values(
    ["FDR", "Consensus_Rank"]
)

# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

# --------------------------------------------------
# Report
# --------------------------------------------------

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "GeneID",
            "Symbol",
            "Median_Paired_Difference",
            "P_Value",
            "FDR",
            "Direction"
        ]
    ].to_string(index=False)
)

print()
print(
    f"Genes tested: {len(results_df)}"
)

print(
    f"FDR < 0.05: "
    f"{results_df['Significant_FDR_0.05'].sum()}"
)

print()
print(f"Saved: {OUTPUT}")
