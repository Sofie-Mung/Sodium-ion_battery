"""build_manifest.py - STEP 1: extract cached MLIP structures into a portable DFT bundle.

Runs on RunPod, where the MLIP cache lives (HEO_WORKDIR=/workspace/heo_v4, read-only base
HEO_BASE_WORKDIR=/workspace/heo_v2 already exported by the notebook environment).

Principle (spec section 3): NOTHING is generated - every structure is pulled from the cache
(heo_worker.relaxed_cif_path), verified, and written out as a POSCAR.  The only in-code
structure in the whole pipeline is bcc Na (built later by runner.py, spec section 5).

Usage (RunPod):
    python build_manifest.py --out dft_bundle
Usage (local logic test, no cache):
    python build_manifest.py --top30 ~/Desktop/heo_sodiun_v4/top30_v3.csv --out /tmp/bundle --dry-run

Outputs under --out:
    dft_inputs/{comp_id}/{phase}/{x_tag}/POSCAR
    manifest.csv            single source of run state (spec section 8-6)
    dft_bundle.tgz          (unless --no-tar) ready to scp to the lab server
"""
import argparse, json, os, shutil, subprocess, sys, tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_common as C

PILOT_XTAGS = ["x100", "x052"]         # spec section 4: top-1, O3/P3 x x100/x052, std+gam


def _crosscheck_lifted_functions(W):
    """Abort if the lifted copies in dft_common ever diverge from heo_worker (anti-drift guard)."""
    st = W.O3_TEMPLATE.copy()
    a = C.prismatic_order(st)[0]
    b = W.prismatic_order(st)[0]
    assert abs(a - b) < 1e-9, f"dft_common.prismatic_order diverged from heo_worker: {a} vs {b}"
    ga, la = C.stacking_report(st)
    gb, lb = W.stacking_report(st)
    assert ga == gb and la == lb, "dft_common.stacking_report diverged from heo_worker"
    ca = C.dV_report(st.lattice.matrix, st.lattice.matrix)
    assert abs(ca["dV_pct"]) < 1e-9 and abs(ca["vm_strain"]) < 1e-9
    print("[ok] lifted functions match heo_worker")


def _k_best_from_energies(recs, phase_pair_stem):
    """Paired argmin over vacancy samples, the same definition as heo_worker.metrics_from:
    k_best = argmin_k (E_P3,k - E_O3,k).  recs: {tag: {energy: ...}}."""
    ks = sorted({p[2] for t in recs if (p := C.parse_tag(t)) and p[1] == phase_pair_stem and p[2] is not None})
    ks = [k for k in ks if f"O3_{phase_pair_stem}_{k}" in recs and f"P3_{phase_pair_stem}_{k}" in recs]
    if not ks: return None, None
    d = [recs[f"P3_{phase_pair_stem}_{k}"]["energy"] - recs[f"O3_{phase_pair_stem}_{k}"]["energy"] for k in ks]
    i = int(np.argmin(d))
    return ks[i], C.de_mev_fu(recs[f"O3_{phase_pair_stem}_{ks[i]}"]["energy"],
                              recs[f"P3_{phase_pair_stem}_{ks[i]}"]["energy"])


def enumerate_rows(top30, anchors_acct, recs_of, n_top):
    """All manifest rows except E_Na and pilot.  top30: DataFrame (file order = ranking order).
    anchors_acct: {comp_id: (counts, acct)}.  recs_of: {comp_id: {tag: rec}} from checkpoints
    (None in --dry-run).  Returns (rows, errors)."""
    rows, errors = [], []
    csv_de_col = {"x100": "dE_pris", "x081": "dE_x081", "x067": "dE_x067", "x052": "dE_desod"}
    csv_kb_col = {"x081": "k_best_x081", "x067": "k_best_x067", "x052": "k_best"}

    def add(comp_id, set_name, x_tag, phase, k_best, mlip_dE, counts, acct):
        stem = C.STEM_OF_XTAG[x_tag]
        tag = C.tag_for(phase, x_tag, k_best if stem != "pris" else 0)
        rec = (recs_of or {}).get(comp_id, {}).get(tag)
        if recs_of is not None and rec is None:
            errors.append(f"{comp_id}/{tag}: no checkpoint row"); return
        rows.append({
            "row_id": f"{comp_id}__{phase}_{x_tag}",
            "set": set_name, "comp_id": comp_id, "phase": phase, "x_tag": x_tag,
            "na_count": C.NA_OF_XTAG[x_tag],
            "k_best": "" if stem == "pris" else int(k_best),
            "struct_hash": "", "mlip_E": "" if rec is None else rec["energy"],
            "mlip_dE": "" if mlip_dE is None else mlip_dE,
            "priority": C.PRIORITY["anchor"] if set_name == "anchor" else C.PRIORITY[x_tag],
            "status": "pending", "fw_id": "", "task_id": "", "note": "",
            "vasp_mode": "default", "attempts": 0,
            "magmom_json": json.dumps(C.magmom_for(counts, acct)),
            "acct_json": json.dumps(acct),
            "mlip_cell": (rec or {}).get("cell") or "",
            "poscar": f"dft_inputs/{comp_id}/{phase}/{x_tag}/POSCAR",
            "tag": tag,                       # dropped before writing; used for CIF lookup
        })

    # ---- top-30 (240 rows)
    for _, r in top30.head(n_top).iterrows():
        comp_id = r["comp_id"]
        counts = json.loads(r["site_counts"])
        acct = {k: int(r[k]) for k in ("n_Ni3", "n_Mn3", "n_Co2", "n_V3", "n_V5")}
        for x_tag in ["x100", "x081", "x067", "x052"]:
            kb = None if x_tag == "x100" else int(float(r[csv_kb_col[x_tag]]))
            if recs_of is not None and x_tag != "x100":
                kb_e, _ = _k_best_from_energies(recs_of.get(comp_id, {}), C.STEM_OF_XTAG[x_tag])
                if kb_e is not None and kb_e != kb:
                    errors.append(f"{comp_id}/{x_tag}: csv k_best={kb} != paired-argmin {kb_e}")
            for phase in ("O3", "P3"):
                add(comp_id, "top30", x_tag, phase, kb, float(r[csv_de_col[x_tag]]), counts, acct)

    # ---- anchors HOST_v1 / HEO_v1 (16 rows) - k_best recomputed from the checkpoints (the
    # anchors_x4_raw.csv columns, when present, are cross-checked by the caller)
    for comp_id, (counts, acct) in anchors_acct.items():
        for x_tag in ["x100", "x081", "x067", "x052"]:
            kb, de = None, None
            if x_tag != "x100":
                if recs_of is None:                       # --dry-run: k_best needs the checkpoints
                    kb = -1
                    for phase in ("O3", "P3"):
                        add(comp_id, "anchor", x_tag, phase, kb, de, counts, acct)
                        rows[-1]["note"] = "dry-run placeholder: k_best from checkpoints"
                    continue
                kb, de = _k_best_from_energies(recs_of.get(comp_id, {}), C.STEM_OF_XTAG[x_tag])
                if kb is None:
                    errors.append(f"{comp_id}/{x_tag}: no paired O3/P3 samples in checkpoints"); continue
            elif recs_of is not None:
                ro, rp = recs_of.get(comp_id, {}).get("O3_pris"), recs_of.get(comp_id, {}).get("P3_pris")
                de = C.de_mev_fu(ro["energy"], rp["energy"]) if ro and rp else None
            for phase in ("O3", "P3"):
                add(comp_id, "anchor", x_tag, phase, kb, de, counts, acct)
    return rows, errors


def verify_and_export(rows, W, out_dir):
    """spec section 3-3: for every row pull the CIF, check Na count vs tag and phase vs
    prismatic order, hash it, and write the POSCAR.  Collects ALL failures, then aborts."""
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar
    errors, seen = [], {}
    for row in rows:
        key = (row["comp_id"], row["tag"])
        if key in seen:                                   # pilot rows reuse top30 structures
            row["struct_hash"] = seen[key]; continue
        fp = W.relaxed_cif_path(*key)
        if fp is None:
            errors.append(f"{key}: relaxed CIF not found in cache"); continue
        st = Structure.from_file(fp)
        n_na = sum(1 for s in st if s.specie.symbol == "Na")
        if n_na != row["na_count"]:
            errors.append(f"{key}: {n_na} Na, tag says {row['na_count']}"); continue
        mean_p, _ = C.prismatic_order(st)
        ph = C.phase_of_p_order(mean_p)
        if ph != row["phase"]:
            errors.append(f"{key}: prismatic_order {mean_p:+.3f} reads {ph}, tag says {row['phase']}")
            continue
        row["struct_hash"] = seen[key] = C.struct_hash(st)
        dst = os.path.join(out_dir, row["poscar"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Poscar(st).write_file(dst)
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top30", default=None, help="top30_v3.csv (default: $HEO_WORKDIR/results/top30_v3.csv)")
    ap.add_argument("--anchors-raw", default=None, help="anchors_x4_raw.csv for k_best cross-check")
    ap.add_argument("--out", default="dft_bundle")
    ap.add_argument("--n-top", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="no cache access: enumerate rows only")
    ap.add_argument("--no-tar", action="store_true")
    args = ap.parse_args()

    W = None
    recs_of = None
    if not args.dry_run:
        import heo_worker as W                            # RunPod only: creates $HEO_WORKDIR dirs
        _crosscheck_lifted_functions(W)
        df = W.merged_energies(check_model=True)          # invariant 6/10 enforced here
        recs_of = W.recs_by_comp(df)
        print(f"[ok] checkpoints merged: {len(df)} rows, model {W.MODEL_TAG}")

    top30_path = args.top30 or (f"{W.WORKDIR}/results/top30_v3.csv" if W else None)
    assert top30_path and os.path.exists(top30_path), f"top30 csv not found: {top30_path}"
    top30 = pd.read_csv(top30_path)
    assert len(top30) >= args.n_top, f"top30 csv has {len(top30)} rows < {args.n_top}"

    # anchors: HOST_v1 / HEO_v1 only (user decision 2026-09-18); accounting via the same greedy
    anchor_counts = {"HOST_v1": {"Ni": 14, "Mn": 13},
                     "HEO_v1": {"Ni": 3, "Cu": 3, "Mg": 3, "Fe": 4, "Co": 4, "Mn": 3, "Ti": 3, "Sn": 3, "Sb": 1}}
    if W is not None:
        for cid, cnt in anchor_counts.items():
            assert cnt == W.ANCHORS[cid], f"anchor counts drifted from heo_worker.ANCHORS: {cid}"
    anchors_acct = {cid: (cnt, C.charge_account(cnt)) for cid, cnt in anchor_counts.items()}
    print("[ok] anchor accounting:", {c: a for c, (_, a) in anchors_acct.items()})

    rows, errors = enumerate_rows(top30, anchors_acct, recs_of, args.n_top)

    # cross-check anchor k_best against anchors_x4_raw.csv when available
    ax_path = args.anchors_raw or (f"{W.WORKDIR}/results/anchors_x4_raw.csv" if W else None)
    if ax_path and os.path.exists(ax_path):
        ax = pd.read_csv(ax_path)
        idc = "comp_id" if "comp_id" in ax.columns else ax.columns[0]
        for row in rows:
            if row["set"] != "anchor" or row["x_tag"] == "x100": continue
            col = {"x081": "k_best_x081", "x067": "k_best_x067", "x052": "k_best"}[row["x_tag"]]
            hit = ax[ax[idc] == row["comp_id"]]
            if len(hit) and col in ax.columns and pd.notna(hit.iloc[0][col]):
                if int(float(hit.iloc[0][col])) != int(row["k_best"]):
                    errors.append(f"{row['comp_id']}/{row['x_tag']}: anchors_x4_raw k_best="
                                  f"{int(float(hit.iloc[0][col]))} != checkpoint argmin {row['k_best']}")

    # ---- E_Na + pilot rows
    rows.append({
        "row_id": "REF__E_Na", "set": "E_Na", "comp_id": "REF", "phase": "", "x_tag": "",
        "na_count": 2, "k_best": "", "struct_hash": "", "mlip_E": "", "mlip_dE": "",
        "priority": C.PRIORITY["E_Na"], "status": "pending", "fw_id": "", "task_id": "",
        "note": "bcc Na 2-atom cell, built by runner (spec 5); always vasp_std + MP k-density",
        "vasp_mode": "std", "attempts": 0, "magmom_json": json.dumps({"Na": 0.0}),
        "acct_json": "{}", "mlip_cell": "", "poscar": "", "tag": None,
    })
    top1 = top30.iloc[0]["comp_id"]
    base = {r["row_id"]: r for r in rows}
    for mode in ("std", "gam"):
        for phase in ("O3", "P3"):
            for x_tag in PILOT_XTAGS:
                src = base.get(f"{top1}__{phase}_{x_tag}")
                assert src is not None, f"pilot source row missing: {top1} {phase} {x_tag}"
                p = dict(src)
                p.update(row_id=f"{src['row_id']}__{mode}", set="pilot", vasp_mode=mode,
                         priority=C.PRIORITY["pilot"], note="")
                rows.append(p)
    # one designated custodian-recovery test (spec section 4): runner injects NELM=8
    for r in rows:
        if r["row_id"] == f"{top1}__O3_x100__std":
            r["note"] = "custodian_test"

    # ---- verify structures + write POSCARs (skipped in --dry-run)
    os.makedirs(args.out, exist_ok=True)
    if not args.dry_run:
        errors += verify_and_export([r for r in rows if r["tag"]], W, args.out)

    if errors:
        print(f"\nABORT: {len(errors)} verification failures (spec 3-3):", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        sys.exit(1)

    # ---- manifest + bundle
    for r in rows:
        r.pop("tag", None)
    mdf = pd.DataFrame(rows, columns=C.MANIFEST_COLUMNS)
    assert mdf.row_id.is_unique
    mdf.to_csv(os.path.join(args.out, "manifest.csv.tmp"), index=False)
    os.replace(os.path.join(args.out, "manifest.csv.tmp"), os.path.join(args.out, "manifest.csv"))

    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("dft_common.py", "runner.py", "worker.sbatch", "submit.sh", "sync_manifest.py",
               "collect.py", "audit.py", "orb_static.py", "config.yaml", "RUNBOOK.md"):
        src = os.path.join(here, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))

    n = mdf.groupby("set").size().to_dict()
    print(f"\nmanifest: {len(mdf)} rows {n}")
    expect = {"top30": 8 * args.n_top, "anchor": 16, "E_Na": 1, "pilot": 8}
    assert n == expect or args.n_top != 30, f"row counts {n} != {expect}"

    if not args.no_tar:
        tgz = args.out.rstrip("/") + ".tgz"
        with tarfile.open(tgz, "w:gz") as t:
            t.add(args.out, arcname=os.path.basename(args.out.rstrip("/")))
        print(f"bundle: {tgz}  -> scp to the lab server, then follow RUNBOOK.md")


if __name__ == "__main__":
    main()
