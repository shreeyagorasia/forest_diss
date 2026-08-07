#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/data_processing/export_model_tables.sh
#
# Purpose:
#   Regenerates the derived parquet/CSV model tables from
#   data/processed/master/clean_master_*.parquet.
#
# Logs:
#   stdout -> logs/data_processing/export_tables_<jobid>.out
#   stderr -> logs/data_processing/export_tables_<jobid>.err
#
# Notes:
#   This is a CPU data-processing job. It does not request a GPU.

#SBATCH -p Teaching
#SBATCH --job-name=export_tables
#SBATCH --output=logs/data_processing/export_tables_%j.out
#SBATCH --error=logs/data_processing/export_tables_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G

cd ~/forest_diss

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

python -m data_processing.export_model_tables
