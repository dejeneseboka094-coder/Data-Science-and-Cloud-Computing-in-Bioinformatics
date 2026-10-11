#!/bin/bash
#SBATCH --job-name=project2_ml
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/project2_%j.out
#SBATCH --error=logs/project2_%j.err

set -euo pipefail

echo "=========================================="
echo "Project 2: RNA-seq Machine Learning"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

cd "$HOME/Project-2-RNA-seq-ML"

echo "Activating project2_ml environment..."

export PATH="$HOME/.conda/envs/project2_ml/bin:$PATH"

echo "Python:"
which python
python --version

echo "Environment:"
echo "$CONDA_PREFIX"

echo "Snakemake:"
which snakemake
snakemake --version

echo "=========================================="
echo "Running Snakemake"
echo "=========================================="

snakemake \
    --cores 2 \
    --printshellcmds \
    --rerun-incomplete

echo "=========================================="
echo "Workflow completed successfully."
echo "End time: $(date)"
echo "=========================================="
