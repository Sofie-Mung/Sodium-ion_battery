# RUNPOD_RUNBOOK.md — how to run `sse_screening.ipynb` on a RunPod pod

## 0. What to upload (into one directory on the Network Volume, e.g. `/workspace/sse_screening`)

```
sse_screening.ipynb   sse_worker.py   PILOT_CRITERIA.md   inputs/Li6PS5Cl.cif   inputs/exp_reference.csv
```
(`tests/` is optional; `SSE_PIPELINE_SPEC.md` is documentation.)

## 1. Pod

* Template: PyTorch (CUDA 12.x), 4× RTX PRO 4500 or A100, Network Volume mounted at `/workspace`.
* Terminal, before starting Jupyter (the notebook reads these from the environment):

```bash
export MP_API_KEY=...            # Materials Project (hull references). Never write it into the notebook.
export RUNPOD_API_KEY=...        # optional: REST fallback for the auto-stop when runpodctl is missing
export SSE_WORKDIR=/workspace/sse_screening
cd $SSE_WORKDIR
python -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > /tmp/constraints.txt   # keep the pod's torch
pip install -q -c /tmp/constraints.txt torch-sim-atomistic orb-models sevenn kinisi python-pptx mp-api pymatgen ase tables
```

Cell 2 installs anything still missing and records the versions in `outputs/run_manifest.json`.

## 2. Run order and where it stops for you

| Cells | What happens | Stops if |
|---|---|---|
| 1–3 | CONFIG, watchdog, packages, both models loaded, one 416-atom energy/force each → `run_manifest.json` | a model does not load (SevenNet falls back to ASE automatically and records it) |
| 4–6 | charge-ledger test, 72 pilot structures + invariants, kinisi synthetic check | any assert |
| 7 | smoke test: A, both models, NPT 2 ps + NVT 5 ps | trajectory not unwrapped / MD error |
| 8 | throughput: A at 600 K, 20 ps, batch of 4 → `logs/throughput.txt` | **estimated pilot wall time > 72 h** → `RuntimeError`; decide `md_nvt_ps_pilot` (or set `confirm_over_budget=True`) and re-run from cell 1 (everything done so far is reused) |
| 9 | Stage I-M: A, F × both models × 500/600/700 K → `results_model_selection.csv` | **no model gives σ_F/σ_A CI-lower-bound > 1** |
| 10 | Stage I: 9 compositions × 3 configs × 3 T × 2 models (162 MD), MP hull references relaxed per model, gates, `figures.pptx` slides 1–4 | **G-a fails**; otherwise prints the G-stab proposal and ends |
| 11 | Stage II — runs only with `RUN_STAGE_II=True` **and** `CONFIG["stab_threshold_eV"]` set | filter passes nothing |
| 12–13 | DFT hand-off export, manifest, `outputs/DONE`, `runpodctl stop pod` | — |

Recommended: run from a terminal so output survives a browser disconnect (all prints also go to `logs/notebook_stdout.log`):

```bash
nohup jupyter nbconvert --to notebook --execute --inplace sse_screening.ipynb --ExecutePreprocessor.timeout=-1 > logs/nbconvert.log 2>&1 &
tail -f logs/notebook_stdout.log
```

Worker logs: `logs/<stage>_w<k>.log` (one per GPU). Progress of a long stage: count `outputs/trajectories/<model>/*/result.json`.

## 3. After Stage I (the user decision point)

1. Read `outputs/results_single_summary.csv`, `outputs/gates.json`, `outputs/figures.pptx` (slides 1–4), `logs/throughput.txt`.
2. Decide the G-stab threshold (proposal printed at the end of cell 10; default 0.03 eV/atom), write it into `PILOT_CRITERIA.md` (G-stab row only) and into cell 1: `CONFIG["stab_threshold_eV"] = ...`.
3. Set `RUN_STAGE_II = True` in cell 1 and Run All again. Cells 3–10 are cached (no recomputation).

## 4. Resume / restart rules

* Every result lives on disk keyed by (model, composition, config, T); re-running any cell skips finished work.
* A killed MD job is recomputed from scratch (the trajectory of a partial job is not reused).
* Never edit `PILOT_CRITERIA.md` except the G-stab row before Stage II; the notebook checks its md5 and aborts otherwise.
* `CONFIG["auto_terminate"]=False` keeps the pod up after the last cell; the watchdog then only logs.

## 5. Known deviations to confirm

* `inputs/Li6PS5Cl.cif` (as provided) is the ordered MP-style cell with Li on **24g** (doublet midpoints) and a = 10.28 Å (PBE). Spec 5.2 (48h-pair sampling) cannot be applied to it; `allow_24g_host=True` in cell 1 accepts this. Supplying an experimental CIF with Li on 48h (occupancy 0.5) restores the spec behaviour without code changes.
* NPT uses an isotropic barostat (torch-sim `npt_nose_hoover_isotropic`); the NVT cell is the mean cell of the second half of NPT.
* SevenNet checkpoint: `7net-omni` with `modal="mpa"` (falls back to `7net-mf-ompa`, then to the ASE calculator); the one actually used is in `run_manifest.json`.
