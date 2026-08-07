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

# Toolchain lives in a shared TA home directory under a dated folder name that gets replaced
# periodically -- hardcoding one date breaks silently (a confusing torch/CUDA import error deep
# inside the Python job, not an obvious "toolchain missing" message) the next time it rotates.
# Finds whatever toolchain-* currently exists instead, picks the most recently modified one, and
# fails loudly with a clear message immediately if none exist at all.
echo "Node: $(hostname)"  # printed BEFORE the toolchain check, on purpose -- if
# /home/htang2 isn'"'"'t mounted on this specific node, everything below dies immediately, and
# without this line the log would never say which node was the problem.
TOOLCHAIN_RC=$(ls -1t /home/htang2/toolchain-*/toolchain.rc 2>/dev/null | head -1)
if [ -z "${TOOLCHAIN_RC}" ]; then
  echo "ERROR: no toolchain.rc found under /home/htang2/toolchain-*/ on node $(hostname) -- /home/htang2 may not be mounted here. Ask a TA if this recurs on the same node." >&2
  exit 1
fi
. "${TOOLCHAIN_RC}"
. .venv/bin/activate

python -m data_processing.export_model_tables
