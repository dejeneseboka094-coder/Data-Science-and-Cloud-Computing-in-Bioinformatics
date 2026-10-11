#!/usr/bin/env python

import pandas as pd
import requests
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VALIDATION = Path(
    "results/ml/consensus_gene_expression_validation.tsv"
)

EXPRESSION = Path(
    "data/processed/filtered_counts.tsv"
)

OUTDIR = Path("results/pathways")
OUTDIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

SOURCES = [
    "GO:BP",
    "GO:MF",
    "GO:CC",
    "REAC",
    "KEGG"
]

# --------------------------------------------------
# Load gene lists and background
# --------------------------------------------------

validation = pd.read_csv(VALIDATION, sep="\t")
validation["GeneID"] = validation["GeneID"].astype(str)

up_genes = validation.loc[
    validation["Direction"] == "Up in Tumor",
    "GeneID"
].tolist()

down_genes = validation.loc[
    validation["Direction"] == "Down in Tumor",
    "GeneID"
].tolist()

expression = pd.read_csv(
    EXPRESSION,
    sep="\t",
    index_col=0
)

background = expression.index.astype(str).tolist()

print("=" * 70)
print("PATHWAY ENRICHMENT ANALYSIS")
print("=" * 70)
print(f"Tumor-up genes: {len(up_genes)}")
print(f"Tumor-down genes: {len(down_genes)}")
print(f"Background genes: {len(background)}")
print()

# --------------------------------------------------
# Query g:Profiler REST API
# --------------------------------------------------

gene_sets = {
    "tumor_up": up_genes,
    "tumor_down": down_genes
}

for set_name, genes in gene_sets.items():

    print("-" * 70)
    print(f"Analyzing: {set_name}")
    print("-" * 70)

    payload = {
        "organism": "hsapiens",
        "query": genes,
        "sources": SOURCES,
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "domain_scope": "custom",
        "background": background,
        "numeric_ns": "ENTREZGENE",
        "no_evidences": True
    }

    response = requests.post(
        API_URL,
        json=payload,
        headers={"User-Agent": "Project2-RNAseq-ML"},
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "result" not in data:
        raise RuntimeError(
            f"Unexpected g:Profiler response: {data}"
        )

    results = pd.DataFrame(data["result"])

    output_file = OUTDIR / f"{set_name}_enrichment.tsv"

    if results.empty:

        results.to_csv(
            output_file,
            sep="\t",
            index=False
        )

        print("No enrichment terms returned.")

    else:

        results.to_csv(
            output_file,
            sep="\t",
            index=False
        )

        print(f"Results saved: {output_file}")
        print(f"Terms returned: {len(results)}")

        if "significant" in results.columns:

            significant = results[
                results["significant"] == True
            ]

        else:

            significant = pd.DataFrame()

        print(
            "Terms significant after FDR correction:",
            len(significant)
        )

        if not significant.empty:

            columns = [
                c for c in [
                    "source",
                    "native",
                    "name",
                    "p_value",
                    "intersection_size"
                ]
                if c in significant.columns
            ]

            print(
                significant[columns]
                .head(15)
                .to_string(index=False)
            )

        else:

            print(
                "No terms passed the FDR significance threshold."
            )

    print()

print("=" * 70)
print("PATHWAY ANALYSIS FINISHED")
print(f"Output directory: {OUTDIR}")
print("=" * 70)
