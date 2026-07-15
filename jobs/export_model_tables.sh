#!/bin/bash
#SBATCH -p Teaching
#SBATCH --job-name=export_tables
#SBATCH --output=logs/export_tables_%j.out
#SBATCH --error=logs/export_tables_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G

cd ~/forest_diss

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

python -m data_processing.export_model_tables

