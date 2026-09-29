#!/bin/bash
# worker.sbatch - one resident worker on one node (spec section 2 "node-resident worker").
# Submitted by submit.sh, which injects the config values via --export; do not edit the
# SBATCH lines by hand - change config.yaml and resubmit.
#
# Expected exported variables: RUN_ROOT, RUNNER_SETS, VASP_MODULE, CONDA_ENV, PSP_DIR
#SBATCH --job-name=heo_dft
#SBATCH --nodes=1
#SBATCH --output=logs/worker_%j.log

set -euo pipefail
cd "${RUN_ROOT}"
mkdir -p logs

module load "${VASP_MODULE}"
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"
export PMG_VASP_PSP_DIR="${PSP_DIR}"

echo "[sbatch] job ${SLURM_JOB_ID} on $(hostname), sets=${RUNNER_SETS}"
python runner.py --root "${RUN_ROOT}" --config config.yaml --sets "${RUNNER_SETS}"
