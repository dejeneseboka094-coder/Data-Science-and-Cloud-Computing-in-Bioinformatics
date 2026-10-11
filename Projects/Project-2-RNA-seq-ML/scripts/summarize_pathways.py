#!/usr/bin/env python

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("results/pathways")
OUTPUT_DIR = Path("results/pathways")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "Tumor-up": INPUT_DIR / "tumor_up_enrichment.tsv",
    "Tumor-down": INPUT_DIR / "tumor_down_enrichment.tsv"
}

all_summaries = []

for direction, file in files.items():

    df = pd.read_csv(file, sep="\t")

    # Keep significant terms only
    if "significant" in df.columns:
        df = df[df["significant"] == True].copy()

    # Sort by p-value
    df = df.sort_values("p_value")

    # Extract useful columns
    summary = df[
        [
            "source",
            "native",
            "name",
            "p_value",
            "intersection_size",
            "query_size",
            "term_size",
            "precision",
            "recall"
        ]
    ].copy()

    summary.insert(0, "Direction", direction)

    all_summaries.append(summary)

    # Save top 20 for each direction
    top20 = summary.head(20)

    top_file = OUTPUT_DIR / (
        "tumor_up_top20.tsv"
        if direction == "Tumor-up"
        else "tumor_down_top20.tsv"
    )

    top20.to_csv(top_file, sep="\t", index=False)

# Combine both directions
combined = pd.concat(all_summaries, ignore_index=True)

combined_file = OUTPUT_DIR / "pathway_enrichment_summary.tsv"
combined.to_csv(combined_file, sep="\t", index=False)

# Create a report-friendly table
report_columns = [
    "Direction",
    "source",
    "native",
    "name",
    "p_value",
    "intersection_size"
]

report_table = combined[report_columns].copy()

report_table.to_csv(
    OUTPUT_DIR / "pathway_report_table.tsv",
    sep="\t",
    index=False
)

print("=" * 70)
print("PATHWAY ENRICHMENT SUMMARY")
print("=" * 70)

for direction in ["Tumor-up", "Tumor-down"]:

    subset = combined[combined["Direction"] == direction]

    print(f"\n{direction}")
    print("-" * 70)
    print(f"Significant terms: {len(subset)}")

    print("\nTop 10 terms:")
    print(
        subset[
            ["source", "native", "name",
             "p_value", "intersection_size"]
        ].head(10).to_string(index=False)
    )

print("\n" + "=" * 70)
print("Files created:")
print("=" * 70)

print("results/pathways/tumor_up_top20.tsv")
print("results/pathways/tumor_down_top20.tsv")
print("results/pathways/pathway_enrichment_summary.tsv")
print("results/pathways/pathway_report_table.tsv")
