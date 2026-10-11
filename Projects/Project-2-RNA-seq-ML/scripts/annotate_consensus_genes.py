#!/usr/bin/env python

from Bio import Entrez
import pandas as pd
import time
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

Entrez.email = "dejeneseboka094@gmail.com"

INPUT = Path("results/ml/consensus_genes.tsv")
OUTPUT = Path("results/ml/consensus_gene_annotation.tsv")

# --------------------------------------------------
# Read consensus genes
# --------------------------------------------------

df = pd.read_csv(INPUT, sep="\t")

gene_ids = (
    df["GeneID"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .tolist()
)

print("=" * 70)
print("NCBI GENE ANNOTATION")
print("=" * 70)
print(f"Genes to annotate: {len(gene_ids)}")
print()

# --------------------------------------------------
# Query NCBI Gene
# --------------------------------------------------

print("Querying NCBI Gene...")

handle = Entrez.esummary(
    db="gene",
    id=",".join(gene_ids),
    retmode="xml"
)

result = Entrez.read(handle)
handle.close()

documents = result["DocumentSummarySet"]["DocumentSummary"]

print(f"Records returned: {len(documents)}")
print()

# --------------------------------------------------
# Parse annotation
# --------------------------------------------------

annotations = {}

for record in documents:

    uid = str(record.attributes.get("uid", ""))

    annotations[uid] = {
        "GeneID": uid,
        "Symbol": str(record.get("NomenclatureSymbol", "")),
        "Description": str(record.get("Description", "")),
        "Chromosome": str(record.get("Chromosome", "")),
        "MapLocation": str(record.get("MapLocation", "")),
        "Organism": str(
            record.get("Organism", {}).get("ScientificName", "")
        )
    }

# --------------------------------------------------
# Merge annotation with ML results
# --------------------------------------------------

annotation_df = pd.DataFrame(
    list(annotations.values())
)

annotation_df["GeneID"] = annotation_df["GeneID"].astype(str)

df["GeneID"] = df["GeneID"].astype(str)

merged = df.merge(
    annotation_df,
    on="GeneID",
    how="left",
    suffixes=("", "_NCBI")
)

# --------------------------------------------------
# Reorder columns
# --------------------------------------------------

preferred_columns = [
    "Rank",
    "GeneID",
    "Symbol",
    "Description",
    "Chromosome",
    "MapLocation",
    "Organism",
    "Importance"
]

available_columns = [
    col for col in preferred_columns
    if col in merged.columns
]

remaining_columns = [
    col for col in merged.columns
    if col not in available_columns
]

merged = merged[
    available_columns + remaining_columns
]

# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

merged.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

print("=" * 70)
print("ANNOTATION COMPLETE")
print("=" * 70)

print(f"Output: {OUTPUT}")
print(f"Genes in output: {len(merged)}")
print(
    f"Successfully annotated: "
    f"{merged['Symbol'].notna().sum()}"
)

print()
print("Annotated genes:")
print(
    merged[
        ["GeneID", "Symbol", "Description"]
    ].to_string(index=False)
)

print()
print("Saved successfully.")
