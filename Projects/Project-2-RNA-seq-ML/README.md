# Machine Learning-Based Classification of Breast Tumor and Adjacent Non-Tumor Tissue Using RNA-seq Data

## Project overview

This project evaluates machine-learning classification of breast tumor and adjacent non-tumor tissue using RNA-seq data from GEO accession GSE233242.

The workflow includes expression-data processing, exploratory data analysis, patient-level model evaluation, feature-selection stability analysis, consensus-gene validation, and pathway enrichment.

## Dataset and sample accounting

- GEO accession: GSE233242
- Upstream filtered expression matrix: 18,863 genes across 84 samples
- Samples represented in the machine-learning metadata: 82
- Complete tumor–normal patient pairs: 41
- Samples used for machine learning: 41 tumor and 41 normal
- Training partition: 66 samples from 33 patients
- Held-out test partition: 16 samples from 8 patients

Two samples present in the upstream source metadata were not included in the analysis metadata: GSM7416151 (Tumor 19) and GSM7416157 (Normal 22). The analysis uses the 82 samples represented in the machine-learning and paired metadata tables.

## Main software environment

- Python 3.10.22
- Snakemake 7.32.4
- pandas 2.3.3
- scikit-learn 1.7.2
- XGBoost 3.2.0
- joblib 1.6.0

Additional packages used by the workflow should be recorded in the dependency file.

## Project structure

- `data/raw/`: source data and metadata files
- `data/processed/`: processed expression matrices and sample metadata
- `scripts/`: Python analysis scripts
- `results/eda/`: exploratory data analysis outputs
- `results/ml/`: model evaluation, feature selection, and consensus-gene outputs
- `results/pathways/`: pathway enrichment outputs
- `models/`: saved model and feature-selector objects
- `hpc/`: HPC batch script
- `Snakefile`: Snakemake workflow
- `config.yaml`: workflow configuration
- `logs/`: HPC execution logs

## Running the workflow

Activate an environment containing the required dependencies, then run:

```bash
snakemake --cores 4
```

To inspect the planned execution without running jobs:

```bash
snakemake -n
```

The workflow uses the project paths configured in `config.yaml`. The training script also uses the project directory under the user's home directory; update that path if running elsewhere.

## Key results

- Dataset: GEO accession GSE233242.
- Analysis set: 41 complete tumor–normal patient pairs.
- Training: 33 patients (66 samples).
- Held-out testing: 8 patients (16 samples).
- Selected model: Random Forest with 500 selected features.
- Mean cross-validation ROC-AUC: approximately 0.967.
- Held-out test ROC-AUC: 1.000 on the small test set.

The perfect held-out result is preliminary and should be interpreted
cautiously because the test set contains only eight patients.
Independent external validation is needed to assess generalizability.

The 19 consensus genes and pathway enrichment results are exploratory
findings, not clinically validated biomarkers or definitive evidence
of altered biological pathways.

## Evaluation and limitations

Model evaluation uses patient-level grouping to avoid placing samples from the same patient in both training and test partitions. Feature selection is performed within cross-validation training folds.

The held-out test set contains only 16 samples from 8 patients. Any perfect test-set metrics must therefore be interpreted cautiously and do not establish clinical readiness or external generalizability.

Consensus-gene expression validation and pathway enrichment are exploratory analyses. They are not independent external validation of the predictive model or definitive evidence of clinically useful biomarkers.

## Reproducibility

The project includes the Snakemake workflow, analysis scripts, configuration, processed data, result tables, figures, and saved model artifacts. Record the exact dependency versions and HPC execution details before final submission.

## Citation

Dataset: GEO accession GSE233242. Cite the original dataset publication and GEO record in the final report.

## Author and course

Prepared for the Data Science and Cloud Computing in Bioinformatics course.
