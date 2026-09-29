#!/usr/bin/env python
"""collect_dft.py -- gather finished VASP runs and compare with the MLIP (Tier-1 check).

Reads   <out>/manifest.csv and every <out>/<dir>/vasprun.xml (+ OUTCAR for site moments)
Writes  <out>/dft_energies.csv      one row per cell: E_DFT (eV/cell), converged, per-element mean |m|
        <out>/dft_vs_mlip.csv       one row per composition: dE_pris / dE_desod for DFT and MLIP (meV/f.u.)
Prints  sign agreement, softening slope (MLIP = slope * DFT), and the moment histogram per element.

Conventions match CLAUDE.md section 3:  dE = (E_P3 - E_O3) * 1000 / 27   [meV/f.u.],
positive = O3 favoured.  The desodiated pair shares vacancy pattern k_best, exactly as in the MLIP.

Usage:  python collect_dft.py --out dft_tier1
"""
import argparse, os
import numpy as np, pandas as pd
from pymatgen.io.vasp.outputs import Vasprun, Outcar

N_FU = 27
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="dft_tier1")
args = ap.parse_args()

man = pd.read_csv(os.path.join(args.out, "manifest.csv"))
recs = []
for r in man.itertuples():
    vr = os.path.join(r.dir, "vasprun.xml")
    if not os.path.exists(vr):
        recs.append(dict(dir=r.dir, status="missing")); continue
    try:
        v = Vasprun(vr, parse_potcar_file=False, parse_dos=False, parse_eigen=False)
    except Exception as e:
        recs.append(dict(dir=r.dir, status=f"unreadable: {type(e).__name__}")); continue
    rec = dict(dir=r.dir, status="ok" if v.converged_electronic else "NOT CONVERGED",
               E_DFT=float(v.final_energy), natoms=len(v.final_structure))
    oc = os.path.join(r.dir, "OUTCAR")
    if os.path.exists(oc):
        try:
            mags = Outcar(oc).magnetization
            st = v.final_structure
            by_el = {}
            for site, m in zip(st, mags):
                by_el.setdefault(site.specie.symbol, []).append(abs(m["tot"]))
            for el, ms in by_el.items():
                if el in ("Na", "O"): continue
                rec[f"m_{el}"] = ";".join(f"{x:.2f}" for x in ms)   # every TM site, for the audit
        except Exception:
            pass
    recs.append(rec)
E = man.merge(pd.DataFrame(recs), on="dir")
E.to_csv(os.path.join(args.out, "dft_energies.csv"), index=False)
print(E.status.value_counts().to_dict())

# ---- paired differences per composition -------------------------------------------------
rows = []
for cid, g in E[E.status == "ok"].groupby("comp_id"):
    e = dict(zip(g.role, g.E_DFT))
    row = dict(comp_id=cid,
               mlip_dE_pris=float(g.mlip_dE_pris.iloc[0]), mlip_dE_desod=float(g.mlip_dE_desod.iloc[0]))
    if {"O3_pris", "P3_pris"} <= e.keys():
        row["dft_dE_pris"] = (e["P3_pris"] - e["O3_pris"]) * 1000 / N_FU
    if {"O3_desod", "P3_desod"} <= e.keys():
        row["dft_dE_desod"] = (e["P3_desod"] - e["O3_desod"]) * 1000 / N_FU
    rows.append(row)
C = pd.DataFrame(rows)
C.to_csv(os.path.join(args.out, "dft_vs_mlip.csv"), index=False)

for which in ("pris", "desod"):
    d = C.dropna(subset=[f"dft_dE_{which}", f"mlip_dE_{which}"])
    if len(d) < 3: print(f"{which}: only {len(d)} pairs finished"); continue
    x, y = d[f"dft_dE_{which}"].values, d[f"mlip_dE_{which}"].values
    slope = float(np.polyfit(x, y, 1)[0])                       # < 1 means the MLIP is softened
    sign = float(np.mean(np.sign(x) == np.sign(y)))
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    print(f"{which:5s}: n={len(d)} | sign agreement {sign:.2f} | softening slope {slope:.2f} "
          f"| Spearman {rho:.2f} | MAE {np.mean(np.abs(x - y)):.1f} meV/f.u.")

# ---- magnetic-moment audit (what the DFT says the oxidation states are) --------------------
# Rough |m| bands for octahedral oxides (audit aid, not a rule):
#   Ni2+ ~1.6-1.8  Ni3+(LS) ~0.8-1.1  Ni4+ ~0    Mn4+ ~3.0  Mn3+ ~3.7-3.9    Co3+(LS) ~0  Co2+(HS) ~2.6-2.8
for el in ("Ni", "Mn", "Co", "Fe", "V", "Cu"):
    col = f"m_{el}"
    if col not in E.columns: continue
    allm = [float(x) for s in E[col].dropna() for x in str(s).split(";")]
    if not allm: continue
    h, edges = np.histogram(allm, bins=np.arange(0, 5.01, 0.5))
    print(f"|m| {el:2s}: " + " ".join(f"[{a:.1f}-{b:.1f}]{n}" for a, b, n in zip(edges[:-1], edges[1:], h)))
