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
VALIDATION = Path(
    "results/ml/consensus_gene_expression_validation.tsv"
)

OUTPUT = Path(
    "results/ml/paired_top_consensus_genes.png"
)

# --------------------------------------------------
# Load data
# --------------------------------------------------

expr = pd.read_csv(
    EXPRESSION,
    sep="\t",
    index_col=0
)

expr.index = expr.index.astype(str)

metadata = pd.read_csv(
    METADATA,
    sep="\t"
)

validation = pd.read_csv(
    VALIDATION,
    sep="\t"
)

metadata = metadata.set_index("sample_id")

# --------------------------------------------------
# Select six strongest genes
# --------------------------------------------------

top_genes = [
    "55248",       # PACC1
    "126374",      # WTIP
    "762",         # CA4
    "51176",       # LEF1
    "94274",       # PPP1R14A
    "55083"        # KIF26B
]

# Get symbols
gene_info = validation.set_index(
    validation["GeneID"].astype(str)
)

# --------------------------------------------------
# Create figure
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10)
)

axes = axes.flatten()

# --------------------------------------------------
# Plot each gene
# --------------------------------------------------

for ax, gene_id in zip(axes, top_genes):

    gene_row = expr.loc[gene_id]

    gene_metadata = metadata.loc[
        gene_row.index
    ].copy()

    gene_metadata["expression"] = gene_row.values

    # Patient IDs
    patients = sorted(
        gene_metadata["patient_id"].unique()
    )

    normal_values = []
    tumor_values = []

    for patient in patients:

        patient_data = gene_metadata[
            gene_metadata["patient_id"] == patient
        ]

        normal = patient_data.loc[
            patient_data["class"] == "Normal",
            "expression"
        ]

        tumor = patient_data.loc[
            patient_data["class"] == "Tumor",
            "expression"
        ]

        if len(normal) == 1 and len(tumor) == 1:
            normal_values.append(normal.iloc[0])
            tumor_values.append(tumor.iloc[0])

    # --------------------------------------------------
    # Paired lines
    # --------------------------------------------------

    x_normal = np.zeros(len(normal_values))
    x_tumor = np.ones(len(tumor_values))

    for n, t in zip(
        normal_values,
        tumor_values
    ):
        ax.plot(
            [0, 1],
            [n, t],
            linewidth=0.8,
            alpha=0.45
        )

    # --------------------------------------------------
    # Points
    # --------------------------------------------------

    ax.scatter(
        x_normal,
        normal_values,
        s=25,
        zorder=3,
        label="Normal"
    )

    ax.scatter(
        x_tumor,
        tumor_values,
        s=25,
        zorder=3,
        label="Tumor"
    )

    # --------------------------------------------------
    # Gene information
    # --------------------------------------------------

    row = gene_info.loc[gene_id]

    symbol = str(row["Symbol"])

    if symbol == "nan":
        symbol = f"GeneID {gene_id}"

    direction = str(
        row["Direction"]
    )

    fdr = float(
        row["FDR"]
    )

    ax.set_title(
        f"{symbol}\n{direction}, FDR={fdr:.2e}"
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels(
        ["Normal", "Tumor"]
    )

    ax.set_ylabel(
        "log2(CPM + 1)"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

# --------------------------------------------------
# Legend
# --------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False
)

fig.suptitle(
    "Paired Expression of Selected Consensus ML Genes",
    fontsize=15,
    y=0.98
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

# --------------------------------------------------
# Save
# --------------------------------------------------

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
print("PAIRED CONSENSUS GENE EXPRESSION")
print("=" * 70)

print("Genes plotted:")

for gene_id in top_genes:
    row = gene_info.loc[gene_id]

    symbol = str(row["Symbol"])

    if symbol == "nan":
        symbol = f"GeneID {gene_id}"

    print(
        f"  {symbol} ({gene_id})"
    )

print()
print(f"Output: {OUTPUT}")
print("Saved successfully.")
