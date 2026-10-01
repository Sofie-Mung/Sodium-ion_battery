#!/bin/bash
# submit.sh - submit 1..5 resident workers (spec section 6: qlaunch rapidfire -m N analog).
#
#   ./submit.sh --pilot 1     pilot workers (only the 8 pilot rows are claimable)
#   ./submit.sh 5             main run - REFUSED until all pilot rows are done (spec section 4)
#   ./submit.sh               main run with n_workers from config.yaml
set -euo pipefail
cd "$(dirname "$0")"
RUN_ROOT="$(pwd)"

SETS="top30,anchor,E_Na"
if [[ "${1:-}" == "--pilot" ]]; then
    SETS="pilot"; shift
fi

cfg() { python -c "import yaml,sys; print(yaml.safe_load(open('config.yaml'))['$1'])"; }
N="${1:-$(cfg n_workers)}"
PARTITION="$(cfg partition)"
NTASKS="$(cfg ntasks)"
WALLTIME="$(cfg walltime)"
VASP_MODULE="$(cfg vasp_module)"
CONDA_ENV="$(cfg conda_env)"
PSP_DIR="$(cfg pmg_vasp_psp_dir)"

if [[ "$SETS" != "pilot" ]]; then
    # pilot gate (also enforced inside runner.py; this fails fast at submit time)
    python - <<'EOF'
import pandas as pd, sys
df = pd.read_csv("manifest.csv", dtype=str).fillna("")
p = df[df["set"] == "pilot"]
bad = p[p["status"] != "done"]
if len(p) == 0 or len(bad):
    sys.exit(f"pilot gate: {len(bad)}/{len(p)} pilot rows not done - run './submit.sh --pilot 1' "
             "first, finish the section-4 checklist, then set kmode/t_est_hours in config.yaml")
EOF
fi

mkdir -p logs
for i in $(seq 1 "$N"); do
    sbatch --partition="$PARTITION" --ntasks-per-node="$NTASKS" --time="$WALLTIME" \
        --export=ALL,RUN_ROOT="$RUN_ROOT",RUNNER_SETS="$SETS",VASP_MODULE="$VASP_MODULE",CONDA_ENV="$CONDA_ENV",PSP_DIR="$PSP_DIR" \
        worker.sbatch
done
echo "submitted $N worker(s), sets=$SETS. Monitor: squeue -u \$USER ; python sync_manifest.py --root $RUN_ROOT"
