#!/usr/bin/env python
"""make_dft_inputs.py -- VASP input decks for the Tier-1 DFT check of the top-N screening hits.

Reads   <WORKDIR>/results/top30_cif_index.json   (written by notebook cell 12)
        <WORKDIR>/results/top30_cifs/*.cif        (MLIP-relaxed O3/P3 x pristine/desodiated cells)
        <WORKDIR>/results/results_v3.csv          (optional: MLIP cell energies for the manifest)
Writes  <out>/<comp_id>__<tag>/{INCAR,KPOINTS,POSCAR,POTCAR,run.sh}   -> 4 dirs per composition
        <out>/manifest.csv                                            one row per directory
        <out>/submit_all.sh                                           sbatch loop

Currency (CLAUDE.md section 8): Materials-Project-compatible PBE(+U) through pymatgen's
MPStaticSet / MPRelaxSet.  ORB-v3 'mpa' was trained on exactly this setting (MPtrj), so the DFT
energies can be compared with the MLIP energies without any correction.  What pymatgen fills in:
  * LDAU on Co/Cr/Fe/Mn/Mo/Ni/V/W in oxides (U = 3.32/3.7/5.3/3.9/4.38/6.2/3.25/6.2 eV)
  * MAGMOM: ferromagnetic high-spin start for every TM (MP convention) -- we do NOT feed the
    predicted n_Ni3/n_Mn3 in; the converged moments are what audits that prediction ("read, don't input")
  * ENCUT 520, PBE PAW set, MP k-point density

Default = single point on the MLIP geometry (Tier-1: sign agreement + softening slope).
--relax switches to ISIF=3 relaxation (MPRelaxSet) for the definitive comparison later.

Usage
  python make_dft_inputs.py --workdir /workspace/heo_v2 --out dft_tier1
  python make_dft_inputs.py --workdir ... --out dft_tier1 --no-potcar        # no PP library on this box
  python make_dft_inputs.py --workdir ... --out dft_relax --relax
  python make_dft_inputs.py ... --slurm my_run.sh                            # your cluster's job template
"""
import argparse, csv, json, os, shutil

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.environ.get("HEO_WORKDIR", "/workspace/heo_v2"))
ap.add_argument("--out", default="dft_tier1")
ap.add_argument("--relax", action="store_true", help="MPRelaxSet with ISIF=3 instead of a single point")
ap.add_argument("--no-potcar", action="store_true", help="write POTCAR.spec instead of POTCAR")
ap.add_argument("--kdens", type=int, default=0, help="reciprocal k-point density; 0 = the set's MP default")
ap.add_argument("--slurm", default="", help="job-script template copied into every directory as run.sh")
ap.add_argument("--only", default="", help="comma-separated comp_ids to restrict to (smoke test)")
args = ap.parse_args()

index = json.load(open(f"{args.workdir}/results/top30_cif_index.json"))
cif_dir = f"{args.workdir}/results/top30_cifs"
os.makedirs(args.out, exist_ok=True)
if args.only:
    keep = set(args.only.split(","))
    index = {k: v for k, v in index.items() if k in keep}

# MLIP cell energies (eV/cell) for the manifest, if the screening table is available
mlip = None
rv = f"{args.workdir}/results/results_v3.csv"
if os.path.exists(rv):
    mlip = pd.read_csv(rv).set_index("comp_id")

# Shared INCAR overrides: identical numerics for every cell so that paired differences
# (P3 - O3, same composition, same vacancy pattern) cancel the settings out.
user_incar = {
    "ISPIN": 2,
    "ISMEAR": 0, "SIGMA": 0.05,   # insulating oxide with a coarse k-mesh: Gaussian, not tetrahedron
    "LREAL": "Auto",              # 95-108 atoms
    "NELM": 200, "EDIFF": 1e-5,
    "LWAVE": False, "LCHARG": False,
    "LORBIT": 11,                 # site-projected moments -> n_Ni3 / n_Mn3 / n_Co2 audit
    "NCORE": 4,
}
if args.relax:
    user_incar.update({"ISIF": 3, "IBRION": 2, "NSW": 200, "EDIFFG": -0.02, "ENCUT": 520})
kpts = {"reciprocal_density": args.kdens} if args.kdens else None
SetCls = MPRelaxSet if args.relax else MPStaticSet

DEFAULT_RUN = """#!/bin/bash
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH -o vasp.%j.out
module load vasp            # <- adapt to your cluster
mpirun -np $SLURM_NTASKS vasp_std
"""

rows = []
for cid, meta in index.items():
    for role, fn in meta["files"].items():            # O3_pris, P3_pris, O3_desod, P3_desod
        tag = fn.split("__", 1)[1][:-4]               # e.g. O3_desod_3
        d = os.path.join(args.out, f"{cid}__{tag}")
        st = Structure.from_file(os.path.join(cif_dir, fn))
        vis = SetCls(st, user_incar_settings=user_incar, user_kpoints_settings=kpts)
        vis.write_input(d, potcar_spec=args.no_potcar)
        if args.slurm:
            shutil.copy(args.slurm, os.path.join(d, "run.sh"))
        else:
            open(os.path.join(d, "run.sh"), "w").write(DEFAULT_RUN.format(name=f"{cid}_{tag}"[:60]))
        row = dict(dir=d, comp_id=cid, role=role, tag=tag, natoms=len(st),
                   formula=st.composition.reduced_formula, k_best=meta["k_best"],
                   mlip_dE_pris=None, mlip_dE_desod=None, mlip_E_cell=None)
        if mlip is not None and cid in mlip.index:
            r = mlip.loc[cid]
            col = {"O3_pris": "E_pris_O3", "P3_pris": "E_pris_P3",
                   "O3_desod": "E_desod_O3", "P3_desod": "E_desod_P3"}[role]
            row.update(mlip_dE_pris=float(r.get("dE_pris", float("nan"))),
                       mlip_dE_desod=float(r.get("dE_desod", float("nan"))),
                       mlip_E_cell=float(r.get(col, float("nan"))))
        rows.append(row)
        print(f"{d}: {len(st)} atoms, {st.composition.reduced_formula}")

with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(os.path.join(args.out, "submit_all.sh"), "w") as f:
    f.write("#!/bin/bash\n# submit every directory that has no finished run yet\n")
    f.write(f'for d in {args.out}/*/; do\n  [ -f "$d/vasprun.xml" ] && continue\n'
            f'  (cd "$d" && sbatch run.sh)\ndone\n')
print(f"\n{len(rows)} directories -> {args.out}/   manifest.csv, submit_all.sh written")
if args.no_potcar:
    print("POTCAR.spec written instead of POTCAR: build POTCARs on the cluster (pmg config -p <pp_dir> PMG_VASP_PSP_DIR)")
