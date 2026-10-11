#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Files
# --------------------------------------------------

EXPRESSION = Path("data/processed/paired_log_cpm.tsv")
METADATA = Path("data/processed/paired_metadata.tsv")
ANNOTATION = Path("results/ml/consensus_gene_expression_validation.tsv")

OUTPUT = Path("results/ml/consensus_gene_heatmap.png")

# --------------------------------------------------
# Load data
# --------------------------------------------------

expr = pd.read_csv(
    EXPRESSION,
    sep="\t",
    index_col=0
)

expr.index = expr.index.astype(str)

metadata = pd.read_csv(METADATA, sep="\t")
annotation = pd.read_csv(ANNOTATION, sep="\t")

# --------------------------------------------------
# Identify consensus genes
# --------------------------------------------------

gene_ids = annotation["GeneID"].astype(str).tolist()

symbols = {}

for _, row in annotation.iterrows():
    gene_id = str(row["GeneID"])
    symbol = str(row["Symbol"])

    if symbol == "nan" or symbol.strip() == "":
        symbol = f"GeneID_{gene_id}"

    symbols[gene_id] = symbol

# --------------------------------------------------
# Select genes
# --------------------------------------------------

available_genes = [
    gene for gene in gene_ids
    if gene in expr.index
]

gene_expr = expr.loc[available_genes].copy()

# --------------------------------------------------
# Order samples by patient and class
# --------------------------------------------------

metadata = metadata.set_index("sample_id")

metadata = metadata.loc[
    [s for s in expr.columns if s in metadata.index]
]

metadata["class_order"] = metadata["class"].map({
    "Normal": 0,
    "Tumor": 1
})

metadata = metadata.sort_values(
    ["patient_id", "class_order"]
)

ordered_samples = metadata.index.tolist()

gene_expr = gene_expr[ordered_samples]

# --------------------------------------------------
# Z-score each gene
# --------------------------------------------------

gene_z = gene_expr.copy()

for gene in gene_z.index:
    values = gene_z.loc[gene]

    gene_z.loc[gene] = (
        values - values.mean()
    ) / values.std(ddof=0)

# --------------------------------------------------
# Gene labels
# --------------------------------------------------

gene_labels = [
    symbols[gene]
    for gene in gene_z.index
]

gene_z.index = gene_labels

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig_width = max(14, len(ordered_samples) * 0.22)

fig, ax = plt.subplots(
    figsize=(fig_width, 8)
)

im = ax.imshow(
    gene_z.values,
    aspect="auto",
    interpolation="nearest"
)

# --------------------------------------------------
# X-axis labels
# --------------------------------------------------

ax.set_xticks(
    np.arange(len(ordered_samples))
)

ax.set_xticklabels(
    [
        metadata.loc[s, "class"]
        for s in ordered_samples
    ],
    rotation=90,
    fontsize=5
)

ax.set_yticks(
    np.arange(len(gene_z.index))
)

ax.set_yticklabels(
    gene_z.index,
    fontsize=9
)

ax.set_xlabel(
    "Patient-paired samples ordered Normal → Tumor"
)

ax.set_ylabel(
    "Consensus genes"
)

ax.set_title(
    "Consensus ML Gene Expression Across Paired Breast Tissue Samples",
    fontsize=13,
    pad=15
)

# --------------------------------------------------
# Add patient boundaries
# --------------------------------------------------

patient_values = metadata["patient_id"].tolist()

for i in range(1, len(patient_values)):

    if patient_values[i] != patient_values[i - 1]:

        ax.axvline(
            i - 0.5,
            linewidth=0.4
        )

# --------------------------------------------------
# Color bar
# --------------------------------------------------

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    "Gene-wise Z-score"
)

# --------------------------------------------------
# Layout
# --------------------------------------------------

plt.tight_layout()

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("CONSENSUS GENE HEATMAP")
print("=" * 70)

print(f"Genes plotted: {len(gene_z)}")
print(f"Samples plotted: {len(ordered_samples)}")
print(f"Output: {OUTPUT}")
print("Saved successfully.")
