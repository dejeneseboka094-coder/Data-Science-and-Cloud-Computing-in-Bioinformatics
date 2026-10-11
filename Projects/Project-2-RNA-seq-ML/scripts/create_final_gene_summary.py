#!/usr/bin/env python

import pandas as pd
from pathlib import Path

INPUT = Path(
    "results/ml/consensus_gene_expression_validation.tsv"
)

OUTPUT = Path(
    "results/ml/final_consensus_gene_summary.tsv"
)

df = pd.read_csv(INPUT, sep="\t")

# Clean missing symbols
df["Symbol"] = df["Symbol"].fillna("")

df.loc[
    df["Symbol"].str.strip().isin(["", "nan"]),
    "Symbol"
] = "GeneID_" + df.loc[
    df["Symbol"].str.strip().isin(["", "nan"]),
    "GeneID"
].astype(str)

# Approximate fold-change representation
# based on median paired log2 difference
df["Approx_Paired_Fold_Change"] = (
    2 ** df["Median_Paired_Difference"]
)

# Select useful columns
summary = df[
    [
        "Consensus_Rank",
        "GeneID",
        "Symbol",
        "Description",
        "RF_Importance",
        "N_Pairs",
        "Median_Normal",
        "Median_Tumor",
        "Median_Paired_Difference",
        "Approx_Paired_Fold_Change",
        "Direction",
        "P_Value",
        "FDR",
        "Significant_FDR_0.05"
    ]
].copy()

# Sort by direction then statistical significance
summary = summary.sort_values(
    ["Direction", "FDR"]
)

summary.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print("=" * 70)
print("FINAL CONSENSUS GENE SUMMARY")
print("=" * 70)

print(
    summary[
        [
            "Symbol",
            "Direction",
            "Median_Paired_Difference",
            "Approx_Paired_Fold_Change",
            "FDR"
        ]
    ].to_string(index=False)
)

print()
print(f"Output: {OUTPUT}")
print(f"Genes: {len(summary)}")
