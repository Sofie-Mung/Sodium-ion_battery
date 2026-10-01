"""collect.py - STEP 4: harvest raw/*.json TaskDocs into dft_raw.csv + frames.extxyz.

All TaskDoc field access goes through the PATHS table below, because the exact emmet
TaskDoc schema is pilot-verified (spec 10-#3): if a path moved in the installed emmet
version, fix it in ONE place here.  Missing fields are recorded per row, never fatal.

frames.extxyz carries every ionic step of every relax (F0 softening + F1 labels, spec
sections 0/7).  Keys: info dft_energy [eV] (e_fr_energy), dft_e0 [eV], dft_stress_kbar
(9 comps), row_id/comp_id/phase/x_tag/step/config_type; arrays dft_forces [eV/A].
orb_static.py consumes exactly these keys.

Usage:  python collect.py --root $RUN_ROOT [--sets top30,anchor,E_Na] [--no-frames]
"""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_common as C

# ---- TaskDoc accessor table (single place to fix if the emmet schema differs; spec 10-#3)
PATHS = {
    "energy":      lambda d: float(d["output"]["energy"]),
    "structure":   lambda d: d["output"]["structure"],
    "forces":      lambda d: d["output"]["forces"],
    "stress":      lambda d: d["output"]["stress"],
    "state":       lambda d: str(d.get("state", "")),
    "mag":         lambda d: d["calcs_reversed"][0]["output"]["outcar"]["magnetization"],
    "ionic_steps": lambda d: d["calcs_reversed"][0]["output"]["ionic_steps"],
    "incar":       lambda d: d["calcs_reversed"][0]["input"]["incar"],
    "kpoints":     lambda d: d["calcs_reversed"][0]["input"]["kpoints"],
}


def get(doc, key, missing):
    try:
        return PATHS[key](doc)
    except Exception:
        missing.add(key)
        return None


def incar_summary(incar):
    """Currency audit trail (spec 8-2/8-3): the values that must be MP defaults or our overrides."""
    if not isinstance(incar, dict):
        return ""
    keys = ["ISPIN", "LORBIT", "ISIF", "NSW", "IBRION", "EDIFF", "EDIFFG",
            "ENCUT", "LDAU", "LDAUU", "MAGMOM", "ALGO", "ISMEAR", "SIGMA"]
    return ";".join(f"{k}={incar[k]}" for k in keys if k in incar)


def to_atoms(struct_dict, energy, e0, forces, stress, info):
    from ase import Atoms
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    at = AseAtomsAdaptor.get_atoms(Structure.from_dict(struct_dict))
    if forces is not None:
        at.new_array("dft_forces", np.asarray(forces, dtype=float))
    at.info.update(info)
    if energy is not None: at.info["dft_energy"] = float(energy)
    if e0 is not None:     at.info["dft_e0"] = float(e0)
    if stress is not None: at.info["dft_stress_kbar"] = np.asarray(stress, dtype=float).ravel()
    return at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sets", default="top30,anchor,E_Na,pilot")
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    sets = [s.strip() for s in args.sets.split(",")]

    man = pd.read_csv(os.path.join(root, "manifest.csv"), dtype=str).fillna("")
    man = man[man["set"].isin(sets)]
    done = man[man["status"] == "done"]
    print(f"collect: {len(done)}/{len(man)} rows done in sets {sets}")

    rows, frames, missing_all = [], [], {}
    for _, m in done.iterrows():
        rid = m["row_id"]
        fp = os.path.join(root, "raw", f"{rid}.json")
        if not os.path.exists(fp):
            print(f"[warn] {rid}: done in manifest but {fp} missing"); continue
        d = json.load(open(fp))
        missing = set()
        relax, static = d.get("relax", {}), d.get("static", {})

        st_dict = get(static, "structure", missing)
        lattice = (st_dict or {}).get("lattice", {}).get("matrix")
        forces = get(static, "forces", missing)
        mag = get(static, "mag", missing)
        steps = get(relax, "ionic_steps", missing) or []
        incar_st = get(static, "incar", missing)
        nsw = (get(relax, "incar", missing) or {}).get("NSW", C.INCAR_OVERRIDES_RELAX["NSW"])

        p_order, phase_dft, n_na = float("nan"), "", ""
        if st_dict:
            from pymatgen.core import Structure
            st = Structure.from_dict(st_dict)
            n_na = sum(1 for s in st if s.specie.symbol == "Na")
            if m["set"] != "E_Na":
                p_order = C.prismatic_order(st)[0]
                phase_dft = C.phase_of_p_order(p_order)

        rows.append({
            "row_id": rid, "set": m["set"], "comp_id": m["comp_id"], "phase": m["phase"],
            "x_tag": m["x_tag"], "na_count": m["na_count"], "k_best": m["k_best"],
            "vasp_mode": d.get("vasp_mode", m["vasp_mode"]),
            "E_dft": get(static, "energy", missing),
            "E_relax": get(relax, "energy", missing),
            "natoms": len(st_dict["sites"]) if st_dict else "",
            "n_na_dft": n_na,
            "lattice": json.dumps(lattice) if lattice else "",
            "p_order_dft": p_order, "phase_dft": phase_dft,
            "max_force": float(np.abs(np.asarray(forces)).max()) if forces else "",
            "converged_static": get(static, "state", missing) == "successful",
            "converged_relax": (get(relax, "state", missing) == "successful"
                                and len(steps) < int(nsw)),
            "n_ionic_steps": len(steps),
            "mag_sites": ";".join(f"{s['species'][0]['element'] if st_dict else '?'}:"
                                  f"{v.get('tot', float('nan')):.3f}"
                                  for s, v in zip((st_dict or {}).get("sites", []), mag or [])
                                  ) if mag else "",
            "incar_summary": incar_summary(incar_st),
            "mlip_E": m["mlip_E"], "mlip_dE": m["mlip_dE"],
            "missing_fields": ",".join(sorted(missing)),
        })
        missing_all[rid] = missing

        if not args.no_frames and steps and m["set"] != "pilot":
            for k, s in enumerate(steps):
                try:
                    frames.append(to_atoms(
                        s["structure"], s.get("e_fr_energy"), s.get("e_0_energy"),
                        s.get("forces"), s.get("stress"),
                        {"row_id": rid, "comp_id": m["comp_id"], "phase": m["phase"],
                         "x_tag": m["x_tag"], "step": k, "config_type": "relax_traj"}))
                except Exception as e:
                    print(f"[warn] {rid} step {k}: frame export failed ({e})")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, "dft_raw.csv"), index=False)
    print(f"dft_raw.csv: {len(out)} rows")

    bad = {r: m for r, m in missing_all.items() if m}
    if bad:
        print(f"[warn] missing TaskDoc fields on {len(bad)} rows - fix PATHS if systematic "
              f"(spec 10-#3): e.g. {list(bad.items())[:3]}")

    if not args.no_frames:
        from ase.io import write
        fp = os.path.join(root, "frames.extxyz")
        write(fp, frames)
        comps = {a.info["comp_id"] for a in frames}
        print(f"frames.extxyz: {len(frames)} frames, {len(comps)} compositions "
              f"(acceptance section 9)")


if __name__ == "__main__":
    main()
