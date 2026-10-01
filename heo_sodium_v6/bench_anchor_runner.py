#!/usr/bin/env python3
"""bench_anchor_runner.py - one model's anchor 4-point benchmark (MODEL_BENCHMARK_SPEC gate 1 + XRD).

Runs the cell-5 (anchor gate) + cell-5d (x4 pilot) computation of heo_na_runpod_v4.ipynb for ONE
uMLIP adapter, with the gates RECORDED instead of asserted: a model failing a gate is benchmark
data, not a crash.  Launched once per model by heo_bench_runpod_v1.ipynb:

    HEO_WORKDIR=<bench>/runs/<adapter>  HEO_BASE_WORKDIR=<bench>/shared_base \
    CUDA_VISIBLE_DEVICES=<k>  <venv-python>  bench_anchor_runner.py --adapter <key>

Contracts kept from heo_worker.py:
  - structures: W.ANCHORS + W.build_items (paired vacancies, comp_rng determinism); the SQS cell
    MUST already exist in the shared base (asserted) - all 12 models start from the SAME cells
    (spec section 10-1) and a missing cache must never fall through to mock random placement.
  - checkpoint rows: W.CKPT_FIELDS with model = this adapter's tag; append + resume by (comp_id, tag).
  - relax conditions, uniform for every model (spec section 10-1): ASE FIRE + FrechetCellFilter,
    fmax = W.FMAX, max_steps = W.MAX_STEPS.  (Production v4 used torch-sim FIRE for ORB - the
    notebook's consistency cell quantifies that path difference.)
  - E_Na: this model's own bcc Na (spec section 10-3).  The PBE range check is recorded, not
    asserted (non-MP-currency contrast models are allowed to differ - spec section 10-6).

Engines (2026-09-30, HEO_BENCH_ENGINE, set per model by the notebook from install_report.json):
  - torchsim (default): the v3/v4 production engine - torch-sim FIRE + Frechet cell filter, batched
    by InFlightAutoBatcher, same fmax/max_steps, force convergence exactly as production
    (atomic forces only; HEO_TS_CELL_CONV=1 adds the cell forces).  Checkpoint tag = <tag>-ts.
  - ase: the 09-17 path (one structure at a time).  Checkpoint tag = <tag>.
  The tags differ so rows from the two engines never mix inside one model's aggregation; a
  torchsim model that fails to build falls back to ase and says so in anchor_bench.json.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

# The worker must import light (no torch, no model) and must never relax anything itself here:
# relaxation happens below through the ASE path.  mock keeps W's import side-effect free.
os.environ["HEO_MODE"] = "mock"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heo_worker as W          # noqa: E402
import bench_models             # noqa: E402
import xrd_tools                # noqa: E402

CKPT = f"{W.WORKDIR}/checkpoints/energies_bench.csv"
TIMES = f"{W.WORKDIR}/checkpoints/bench_times.csv"     # main() switches to bench_times_ts.csv
RESULTS = f"{W.WORKDIR}/results"
TS_CHUNK = int(os.environ.get("HEO_TS_CHUNK", 256))    # structures per optimize() call = resume grain
TS_CELL_CONV = os.environ.get("HEO_TS_CELL_CONV", "0") == "1"
TS_MAX_ATOMS = int(os.environ.get("HEO_TS_MAX_ATOMS", 500_000))


# ============================================================ relax engine (uniform ASE path)
def _frechet_filter(atoms):
    try:
        from ase.filters import FrechetCellFilter
    except ImportError:                     # ase < 3.23
        from ase.constraints import FrechetCellFilter
    return FrechetCellFilter(atoms)


def ase_relax(atoms, calc, fmax, max_steps):
    from ase.optimize import FIRE
    atoms.calc = calc
    opt = FIRE(_frechet_filter(atoms), logfile=None)
    conv = bool(opt.run(fmax=fmax, steps=max_steps))
    return atoms, conv, int(opt.nsteps)


def _load_done(tag_model):
    """(comp_id, tag) already relaxed BY THIS ENGINE+MODEL tag - rows of the other engine don't count."""
    done = set()
    if os.path.exists(CKPT):
        try:
            df = pd.read_csv(CKPT)
            df = df[df.model == tag_model]
            done.update(zip(df.comp_id, df.tag))
        except Exception:
            pass
    return done


def _append_row(row):
    hdr = not os.path.exists(CKPT)
    with open(CKPT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=W.CKPT_FIELDS)
        if hdr:
            w.writeheader()
        w.writerow(row)


def _append_time(comp_id, tag, seconds, nsteps):
    hdr = not os.path.exists(TIMES)
    with open(TIMES, "a", newline="") as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(["comp_id", "tag", "seconds", "nsteps"])
        w.writerow([comp_id, tag, round(seconds, 2), nsteps])


def relax_items(items, calc, adapter, tag_model, counts_of):
    """Relax every (comp_id, tag, Structure) not in the checkpoint; one CSV row per structure.
    A single failing structure is skipped with a log line, never kills the run."""
    from pymatgen.io.ase import AseAtomsAdaptor
    done = _load_done(tag_model)
    todo = [(cid, tg, st) for cid, tg, st in items if (cid, tg) not in done]
    print(f"[{adapter}] {len(items) - len(todo)} cached, {len(todo)} to relax", flush=True)
    n_fail = 0
    for i, (cid, tg, st) in enumerate(todo):
        t0 = time.time()
        try:
            if adapter == "mock":
                if cid == "REF":                 # mock Na reference: the worker's fixed value
                    E, V, conv, rst = W._E_NA_MOCK * len(st), st.volume, True, st
                else:
                    E, V, conv, rst = W.relax_mock(cid, tg, st, counts_of.get(cid, {}))
                nst = 0
                cell = rst.lattice.matrix
                natoms = len(rst)
                p_ord = W._p_order(rst)
                st_out = rst
            else:
                at = AseAtomsAdaptor.get_atoms(st)
                at, conv, nst = ase_relax(at, calc, W.FMAX, W.MAX_STEPS)
                E = float(at.get_potential_energy())
                V = float(at.get_volume())
                cell = at.cell[:]
                natoms = len(at)
                st_out = AseAtomsAdaptor.get_structure(at)
                p_ord = W._p_order(st_out)
            _append_row(dict(comp_id=cid, tag=tg, energy=float(E), natoms=int(natoms),
                             volume=float(V), P_order=p_ord, conv=bool(conv),
                             model=tag_model, cell=W._cell_json(cell)))
            _append_time(cid, tg, time.time() - t0, nst)
            if cid not in ("HULL", "REF"):
                st_out.to(filename=f"{W.RELAXED_DIR}/{cid}__{tg}.cif")
        except Exception as e:                    # noqa: BLE001 - skip-and-continue, like relax_batch
            n_fail += 1
            print(f"[{adapter}] SKIP {cid}/{tg}: {type(e).__name__}: {str(e)[:150]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[{adapter}] {i + 1}/{len(todo)} relaxed", flush=True)
    return len(todo), n_fail


# ============================================================ relax engine 2 (torch-sim, batched)
def ts_relax(ts_model, atoms_list, autobatch, scaler=None):
    """One torch-sim optimize() over atoms_list with the production settings (heo_worker.relax_batch):
    FIRE + Frechet cell filter, force_tol = W.FMAX, max_steps = W.MAX_STEPS.
    Returns (final_state, converged[bool array], memory scaler to reuse for the next chunk)."""
    import torch
    import torch_sim as ts
    from torch_sim.optimizers.cell_filters import CellFilter
    conv_fn = ts.generate_force_convergence_fn(force_tol=W.FMAX, include_cell_forces=TS_CELL_CONV)
    state = ts.io.atoms_to_state(atoms_list, device=ts_model.device, dtype=ts_model.dtype)
    ab = False
    if autobatch and torch.device(ts_model.device).type == "cuda":   # memory probe is CUDA-only
        ab = ts.InFlightAutoBatcher(ts_model, memory_scales_with=ts_model.memory_scales_with,
                                    max_memory_scaler=scaler, max_atoms_to_try=TS_MAX_ATOMS)
    final = ts.optimize(system=state, model=ts_model, optimizer=ts.optimizers.Optimizer.fire,
                        convergence_fn=conv_fn, max_steps=W.MAX_STEPS,
                        init_kwargs={"cell_filter": CellFilter.frechet}, autobatcher=ab)
    conv = conv_fn(final).detach().cpu().numpy().astype(bool)   # optimize() also returns max_steps hits
    return final, conv, (ab.max_memory_scaler if ab else scaler)


def ts_relax_items(items, ts_model, adapter, tag_model):
    """Batched counterpart of relax_items with the same checkpoint contract (one CSV row + one CIF per
    structure, resume by (comp_id, tag) under this tag_model).  Failure handling is heo_worker's
    2026-09-04 fix: a failed chunk is retried structure by structure AFTER the exception object is
    released and the allocator emptied, so one bad cell or an OOM never drops the whole chunk."""
    import gc
    import torch
    import torch_sim as ts
    from pymatgen.io.ase import AseAtomsAdaptor

    def _free_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    done = _load_done(tag_model)
    todo = [(cid, tg, st) for cid, tg, st in items if (cid, tg) not in done]
    print(f"[{adapter}] torch-sim engine: {len(items) - len(todo)} cached, {len(todo)} to relax "
          f"(chunks of {TS_CHUNK})", flush=True)
    n_fail = 0
    scaler = float(os.environ["HEO_TS_MEMORY_SCALER"]) if os.environ.get("HEO_TS_MEMORY_SCALER") else None
    for c0 in range(0, len(todo), TS_CHUNK):
        chunk = todo[c0:c0 + TS_CHUNK]
        atoms = [AseAtomsAdaptor.get_atoms(st) for *_, st in chunk]
        _free_gpu()
        t0 = time.time()
        batch_err, results = None, []
        try:
            final, conv, scaler = ts_relax(ts_model, atoms, True, scaler)
            energies = [float(x) for x in final.energy.detach().cpu().numpy().ravel()]
            rel = ts.io.state_to_atoms(final)
            # the batch engine regroups structures internally; atom counts are a cheap fingerprint
            # that catches a reordering before it silently mislabels energies (as production)
            assert [len(a) for a in rel] == [len(a) for a in atoms], \
                "torch-sim returned structures in a different order -> energies would be mislabelled"
            results = list(zip(chunk, energies, rel, conv))
            final = None
        except AssertionError:
            raise
        except (ValueError, RuntimeError) as err:
            batch_err = f"{type(err).__name__}: {str(err)[:150]}"
        if batch_err is not None:
            final = None
            _free_gpu()                       # the exception (and its traceback) is out of scope here
            print(f"[{adapter}] [batch fallback] {batch_err} -> per-structure torch-sim", flush=True)
            for item, at in zip(chunk, atoms):
                try:
                    f1, c1, _ = ts_relax(ts_model, [at], False)
                    results.append((item, float(f1.energy.detach().cpu().numpy().ravel()[0]),
                                    ts.io.state_to_atoms(f1)[0], bool(c1[0])))
                    f1 = None
                except Exception as err2:                 # noqa: BLE001 - skip-and-continue
                    n_fail += 1
                    print(f"[{adapter}] SKIP {item[0]}/{item[1]}: {type(err2).__name__}: "
                          f"{str(err2)[:150]}", flush=True)
                    err2 = None
                    _free_gpu()
        dt = (time.time() - t0) / max(len(results), 1)   # amortised wall time per structure
        for (cid, tg, _), en, at, cv in results:
            st_out = AseAtomsAdaptor.get_structure(at)
            _append_row(dict(comp_id=cid, tag=tg, energy=float(en), natoms=len(at),
                             volume=float(at.get_volume()), P_order=W._p_order(st_out), conv=bool(cv),
                             model=tag_model, cell=W._cell_json(at.cell[:])))
            _append_time(cid, tg, dt, -1)                 # per-structure step counts are not exposed
            if cid not in ("HULL", "REF"):
                st_out.to(filename=f"{W.RELAXED_DIR}/{cid}__{tg}.cif")
        n_conv = sum(bool(r[3]) for r in results)
        print(f"[{adapter}] {min(c0 + TS_CHUNK, len(todo))}/{len(todo)} relaxed "
              f"({n_conv}/{len(results)} converged, {dt:.2f} s/structure amortised)", flush=True)
    return len(todo), n_fail


# ============================================================ aggregation (cells 5 + 5d, recorded)
def aggregate(adapter, tag_model, e_na, status):
    df = pd.read_csv(CKPT)
    df = df[df.model == tag_model]
    W._check_tag_natoms(df)                     # invariant 10 still guards the cache key
    R = W.recs_by_comp(df, list(W.ANCHORS))
    floor = W.DELTA_FLOOR_MEV_FU
    W.set_E_Na(e_na)

    rows = []
    for aid in W.ANCHORS:
        rec = R.get(aid, {})
        if "O3_pris" not in rec or "P3_pris" not in rec:
            status["warnings"].append(f"{aid}: pristine pair missing -> anchor dropped from aggregation")
            continue
        m = W.metrics_from(rec)
        m["anchor"], m["group"] = aid, aid.split("_")[0]
        for s in W.X4_STEMS:
            mx = W.metrics_x(rec, s)
            if not mx:
                status["warnings"].append(f"{aid}: no complete {s} pair")
            m.update(mx)
        rows.append(m)
    if not rows:
        status["status"] = "no_anchor_metrics"
        return None, None
    A4 = pd.DataFrame(rows).set_index("anchor")
    A4.to_csv(f"{RESULTS}/anchors_x4_raw.csv")

    stems = W.X_GRID_DESC
    dE_col = {"pris": "dE_pris", "x081": "dE_x081", "x067": "dE_x067", "desod": "dE_desod"}
    sg_col = {"x081": "sigma_vac_x081", "x067": "sigma_vac_x067", "desod": "sigma_vac"}
    G = {}
    for grp, sub in A4.groupby("group"):
        g = {"group": grp}
        verdicts, E_nat, src = [], {}, {}
        for s in stems:
            if dE_col[s] not in sub.columns or sub[dE_col[s]].isna().all():
                verdicts.append("missing")
                continue
            dE = float(sub[dE_col[s]].mean())
            g[dE_col[s]] = dE
            if s == "pris":
                v = W.judge_phase(dE, floor)
                g["delta_pris"] = floor
                E_nat[s], src[s] = float(sub.E_pris_O3.mean()), "O3"
            else:
                delta = float(max(sub[sg_col[s]].max(), floor)) if sg_col[s] in sub.columns else floor
                g[f"delta_{s}"] = delta
                v = W.judge_phase(dE, delta)
                eo_c = "E_desod_O3" if s == "desod" else f"E_O3_{s}"
                ep_c = "E_desod_P3" if s == "desod" else f"E_P3_{s}"
                if eo_c in sub.columns and ep_c in sub.columns:
                    E_nat[s], src[s] = W.e_nat_of(v, float(sub[eo_c].mean()), float(sub[ep_c].mean()))
            g[f"verdict_{s}"] = v
            verdicts.append(v)
        g.update(W.x_star_bracket(verdicts))
        if np.isfinite(e_na):
            g.update(W.v_segments(E_nat, e_na))
        g.update(W.formation_x(E_nat))
        g["E_nat_src"] = "/".join(src.get(s, "?") for s in stems)
        g["V_avg"] = float(sub.V_avg.mean()) if "V_avg" in sub.columns else float("nan")
        g["sigma_vac"] = float(sub.sigma_vac.max()) if "sigma_vac" in sub.columns else float("nan")
        g["n_phase_flip"] = int(sub.n_phase_flip.sum()) if "n_phase_flip" in sub.columns else -1
        G[grp] = g
    X4 = pd.DataFrame(G.values()).set_index("group")
    X4.to_csv(f"{RESULTS}/anchors_x4.csv")

    # ---- cell-5 gates, RECORDED (assert -> boolean) ----
    gates, diag = {}, {}
    if "HOST" in G:
        h = G["HOST"]
        gates["gate_host_V_window"] = bool(2.5 <= h.get("V_avg", float("nan")) <= 3.4) \
            if np.isfinite(h.get("V_avg", float("nan"))) else None
        gates["gate_host_glide"] = (h.get("verdict_desod") == "P3") if "verdict_desod" in h else None
        gates["gate_host_xstar_in_window"] = h.get("x_star_class") in ("transition", "amb_boundary")
    if "HOST" in G and "HEO" in G:
        gap_pris = float(G["HEO"].get("dE_pris", np.nan) - G["HOST"].get("dE_pris", np.nan))
        margin = float(max(A4.groupby("group").dE_pris.std(ddof=0).max(), floor))
        gates["gate_pris_direction"] = bool(gap_pris > margin) if np.isfinite(gap_pris) else None
        diag["gap_pris"], diag["margin_pris"] = gap_pris, margin
        for s in ("x081", "x067", "desod"):
            c = dE_col[s]
            if c in X4.columns and {"HEO", "HOST"} <= set(X4.index):
                diag[f"gap_{s}"] = float(X4.loc["HEO", c] - X4.loc["HOST", c])
    vs_cols = [c for c in X4.columns if c.startswith("V_seg_")]
    if vs_cols:
        gates["gate_vseg_positive"] = bool((X4[vs_cols] > 0).all().all())  # invariant 9
    status["gates"], status["diag"] = gates, diag

    # ---- change-A decomposition on the anchors (5d block) ----
    dv_rows = []
    for aid in W.ANCHORS:
        rec = R.get(aid, {})
        if "O3_pris" not in rec:
            continue
        cp = W.cell_of(aid, "O3_pris", rec["O3_pris"])
        for s in ("x081", "x067", "desod"):
            kb_col = "k_best" if s == "desod" else f"k_best_{s}"
            if kb_col not in A4.columns or pd.isna(A4.loc[aid, kb_col]):
                continue
            kb = int(A4.loc[aid, kb_col])
            grp = aid.split("_")[0]
            v = X4.loc[grp, f"verdict_{s}"] if f"verdict_{s}" in X4.columns else "P3"
            ph = "O3" if v == "O3" else "P3"
            cd = W.cell_of(aid, f"{ph}_{s}_{kb}", rec.get(f"{ph}_{s}_{kb}"))
            if cp is None or cd is None:
                continue
            r = W.dV_report(cp, cd)
            r.update(anchor=aid, x=round(W.X_OF[s], 4), phase=ph)
            dv_rows.append(r)
    if dv_rows:
        DV = pd.DataFrame(dv_rows)
        DV.to_csv(f"{RESULTS}/anchors_x4_dV.csv", index=False)
        # promote the group-mean decomposition into the cross-model table (spec section 4 lists
        # dV/dA/dh_perp/vm_strain among the Tier R observables): geometry is the second axis on
        # which models can disagree even when their dE signs match.
        DV["group"] = DV.anchor.str.split("_").str[0]
        for stem in ("x067", "desod"):
            xs = round(W.X_OF[stem], 4)
            gm = DV[DV.x == xs].groupby("group")[["dV_pct", "dA_pct", "dh_perp_pct", "vm_strain"]].mean()
            for grp in gm.index.intersection(X4.index):
                for c in gm.columns:
                    X4.loc[grp, f"{c}_{stem}"] = float(gm.loc[grp, c])
        X4.to_csv(f"{RESULTS}/anchors_x4.csv")
    return A4, X4


# ============================================================ top-N target aggregation (per comp)
def aggregate_targets(tag_model, targets, e_na, status):
    """Per-composition 4-point metrics for the top-N targets: the cell-5d quantities WITHOUT
    variant averaging (a screening composition has no anchor-style variants).  One row per comp
    -> results/targets_x4.csv; this is the table the cross-model cell compares (rho_ddE, verdict
    agreement, MAE_dE) - 'does model M reproduce the ORB top-N?' in the pipeline's own language."""
    df = pd.read_csv(CKPT)
    df = df[df.model == tag_model]
    R = W.recs_by_comp(df, list(targets.comp_id))
    floor = W.DELTA_FLOOR_MEV_FU
    W.set_E_Na(e_na)
    dE_col = {"pris": "dE_pris", "x081": "dE_x081", "x067": "dE_x067", "desod": "dE_desod"}
    sg_col = {"x081": "sigma_vac_x081", "x067": "sigma_vac_x067", "desod": "sigma_vac"}
    rows = []
    for r in targets.itertuples():
        rec = R.get(r.comp_id, {})
        if "O3_pris" not in rec or "P3_pris" not in rec:
            continue
        m = W.metrics_from(rec)
        for s in W.X4_STEMS:
            m.update(W.metrics_x(rec, s))
        g = {"comp_id": r.comp_id}
        verdicts, E_nat = [], {}
        for s in W.X_GRID_DESC:
            dE = m.get(dE_col[s])
            if dE is None or not np.isfinite(dE):
                verdicts.append("missing")
                continue
            g[dE_col[s]] = dE
            if s == "pris":
                v = W.judge_phase(dE, floor)
                E_nat[s] = m["E_pris_O3"]
            else:
                delta = float(max(m.get(sg_col[s], floor), floor))
                g[f"delta_{s}"] = delta
                v = W.judge_phase(dE, delta)
                eo = m.get("E_desod_O3" if s == "desod" else f"E_O3_{s}")
                ep = m.get("E_desod_P3" if s == "desod" else f"E_P3_{s}")
                if eo is not None and ep is not None:
                    E_nat[s], _ = W.e_nat_of(v, eo, ep)
            g[f"verdict_{s}"] = v
            verdicts.append(v)
        for k in ("ddE", "sigma_vac", "V_avg", "n_phase_flip"):
            g[k] = m.get(k)
        g.update(W.x_star_bracket(verdicts))
        if np.isfinite(e_na):
            g.update(W.v_segments(E_nat, e_na))
        g.update(W.formation_x(E_nat))
        cp = W.cell_of(r.comp_id, "O3_pris", rec.get("O3_pris"))
        for s in ("x081", "x067", "desod"):
            kb = m.get("k_best" if s == "desod" else f"k_best_{s}")
            v = g.get(f"verdict_{s}")
            if kb is None or v is None or cp is None:
                continue
            ph = "O3" if v == "O3" else "P3"
            cd = W.cell_of(r.comp_id, f"{ph}_{s}_{int(kb)}", rec.get(f"{ph}_{s}_{int(kb)}"))
            if cd is None:
                continue
            for k2, v2 in W.dV_report(cp, cd).items():
                g[f"{k2}_{s}"] = v2
        rows.append(g)
    if not rows:
        status["warnings"].append("targets: no complete 4-point rows -> targets_x4.csv skipped")
        return None
    T = pd.DataFrame(rows).set_index("comp_id")
    T.to_csv(f"{RESULTS}/targets_x4.csv")
    st = status.setdefault("targets", {})
    st["n"] = int(len(T))
    if "verdict_desod" in T.columns:
        st["verdict_desod"] = {k: int(v) for k, v in T.verdict_desod.value_counts().items()}
    if "ddE" in T.columns:
        st["ddE_median"] = float(T.ddE.median())
    return T


# ============================================================ XRD (XRD_SPEC, on this model's cells)
def run_xrd(adapter, tag_model, status, extra_counts=None):
    df = pd.read_csv(CKPT)
    df = df[df.model == tag_model]
    counts_all = dict(W.ANCHORS)
    if extra_counts:
        counts_all.update(extra_counts)                   # top-N targets join the anchors
    recs_of = W.recs_by_comp(df, list(counts_all))
    tm_set = set(W.OXI) | {e for c in counts_all.values() for e in c}

    def loader(comp_id, tag):
        fp = f"{W.RELAXED_DIR}/{comp_id}__{tag}.cif"
        if not os.path.exists(fp):
            return None
        from pymatgen.core import Structure
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Structure.from_file(fp)

    xdf, _refs = xrd_tools.run_xrd(list(counts_all), counts_all, recs_of, loader, RESULTS,
                                   W.X_STEMS, W.X_OF, n_vac=W.N_VAC_SAMPLES, tm_set=tm_set)
    if xdf is None or xdf.empty:
        status["warnings"].append("xrd: no relaxed CIFs found -> skipped")
        return None
    xdf.to_csv(f"{RESULTS}/anchors_x4_xrd.csv", index=False)
    host = xrd_tools.host_direction_test(xdf)
    status["xrd"] = {
        "dir_ok_003": host["xrd_dir_ok_003"], "dir_ok_110": host["xrd_dir_ok_110"],
        "tt_003_track": host["tt_003_track"], "tt_110_track": host["tt_110_track"],
        "d003_audit_max_resid": float(np.nanmax(xdf.chk_d003_resid)) if len(xdf) else None,
        "d003_audit_bad_rows": int((xdf.chk_d003_resid > xrd_tools.D003_TOL).sum()),
        "letter_agree_rate": float(xdf.xrd_agree_letter.dropna().mean())
        if "xrd_agree_letter" in xdf.columns and xdf.xrd_agree_letter.notna().any() else None,
    }
    return xdf


# ============================================================ Ehull extension (v4 cell-8 definitions)
# The hull is IN-MODEL on both sides, exactly as production: the target compound AND the MP
# competitor structures are relaxed by the same model, so no cross-currency energies ever meet.
# Competitor STRUCTURES (from the shared cache, MP geometries) are identical for every model.
def _ehull_inputs(status):
    """(enabled, targets_df, competitor_items).  Needs shared_base/bench_targets.csv and
    shared_base/hull_cache/competitors.json, both written by notebook cell 4."""
    shared = W.BASE_WORKDIR or ""
    tfp, hfp = f"{shared}/bench_targets.csv", f"{shared}/hull_cache/competitors.json"
    if not (shared and os.path.exists(tfp)):
        status["warnings"].append("targets: shared_base/bench_targets.csv missing -> top-N skipped "
                                  "(run notebook cell 4 with HEO_BENCH_TOPN>0)")
        return False, None, []
    targets = pd.read_csv(tfp)
    comp_items = []
    # HEO_BENCH_EHULL=0 turns off ONLY the hull side (competitor relaxations + Ehull table);
    # the top-N targets themselves still run - they are the benchmark's core comparison set.
    if os.environ.get("HEO_BENCH_EHULL", "1") == "0":
        status["warnings"].append("ehull: competitor relaxation disabled (HEO_BENCH_EHULL=0); "
                                  "top-N targets still run, Ehull table skipped")
    elif os.path.exists(hfp):
        import math
        from pymatgen.core import Structure
        pools = [frozenset(json.loads(sc)) | {"Na", "O"} for sc in targets.site_counts]
        raw = json.load(open(hfp))["entries"]
        for mid, sd in raw.items():
            st = Structure.from_dict(sd)
            els = {str(el) for el in st.composition.elements}
            if not any(els <= p for p in pools):
                continue                          # not a competitor of any target's element set
            if len(st) < 24:                      # v1 CELL 7-FIX: keep relax cells homogeneous
                n = math.ceil((24 / len(st)) ** (1.0 / 3.0)); st = st * (n, n, n)
            comp_items.append(("HULL", mid, st))
        comp_items.sort(key=lambda x: len(x[2]))
    return True, targets, comp_items


def compute_ehull(tag_model, targets, status):
    """v4 cell-8 LP hull (verified there against pymatgen to 1e-6) on this model's own energies."""
    from scipy.optimize import linprog
    from pymatgen.core import Composition, Structure
    df = pd.read_csv(CKPT); df = df[df.model == tag_model]
    kT = W.K_B * W.T_SYNTH
    if MOCK:
        rows = []
        for r in targets.itertuples():
            cnt = json.loads(r.site_counts)
            eh = W._mock_params(r.comp_id, cnt)["Ehull"]
            rows.append(dict(comp_id=r.comp_id, Ehull=eh, Ehull_eff=eh - kT * r.sconf_per_atom_kB,
                             T_star=eh / (W.K_B * r.sconf_per_atom_kB)))
        E = pd.DataFrame(rows)
    else:
        hd = df[df.comp_id == "HULL"].set_index("tag")
        if not len(hd):
            status["warnings"].append("ehull: no relaxed competitors in the checkpoint -> skipped")
            return None
        shared = W.BASE_WORKDIR
        raw = json.load(open(f"{shared}/hull_cache/competitors.json"))["entries"]
        frac, epa = [], []
        for mid in hd.index:
            if mid not in raw:
                continue
            comp = Structure.from_dict(raw[mid]).composition
            frac.append({str(el): comp.get_atomic_fraction(el) for el in comp.elements})
            epa.append(float(hd.loc[mid, "energy"]) / float(hd.loc[mid, "natoms"]))
        epa = np.array(epa)
        sub_cache = {}

        def hull_epa(comp, key):
            if key not in sub_cache:
                els = sorted(key)
                idx = [i for i, f in enumerate(frac) if set(f) <= key]
                A = np.array([[frac[i].get(el, 0.0) for i in idx] for el in els])
                sub_cache[key] = (idx, A, els)
            idx, A, els = sub_cache[key]
            b = np.array([comp.get_atomic_fraction(el) for el in els])
            res = linprog(epa[idx], A_eq=A, b_eq=b, bounds=(0, None), method="highs")
            if not res.success:
                return float("nan")
            return float(res.fun)

        R = W.recs_by_comp(df[~df.comp_id.isin(["HULL", "REF"])])
        rows = []
        for r in targets.itertuples():
            rec = R.get(r.comp_id, {})
            if "O3_pris" not in rec:
                continue
            cnt = json.loads(r.site_counts)
            comp = Composition({**cnt, "Na": W.N_NA, "O": W.N_O})
            h = hull_epa(comp, frozenset(str(el) for el in comp.elements))
            if not np.isfinite(h):
                status["warnings"].append(f"ehull: LP failed for {r.comp_id}")
                continue
            eh = max(0.0, rec["O3_pris"]["energy"] / comp.num_atoms - h)
            row = dict(comp_id=r.comp_id, Ehull=eh, Ehull_eff=eh - kT * r.sconf_per_atom_kB,
                       T_star=eh / (W.K_B * r.sconf_per_atom_kB))
            if "P3_pris" in rec:
                row["dE_pris"] = (rec["P3_pris"]["energy"] - rec["O3_pris"]["energy"]) * 1000 / W.N_FU
            rows.append(row)
        E = pd.DataFrame(rows)
        status.setdefault("ehull", {})["n_competitors"] = int(len(hd))
    if not len(E):
        status["warnings"].append("ehull: no target rows -> skipped")
        return None
    if MOCK:
        R = W.recs_by_comp(pd.read_csv(CKPT).query("model == @tag_model"))
        E["dE_pris"] = [((R.get(c, {}).get("P3_pris", {}).get("energy", np.nan)
                          - R.get(c, {}).get("O3_pris", {}).get("energy", np.nan)) * 1000 / W.N_FU)
                        for c in E.comp_id]
    E.to_csv(f"{RESULTS}/bench_ehull.csv", index=False)
    st = status.setdefault("ehull", {})
    st.update(n_targets=int(len(E)),
              n_survivors=int(((E.Ehull_eff < W.EHULL_EFF_CUT)
                               & (E.get("dE_pris", pd.Series(np.nan, index=E.index))
                                  > W.DELTA_FLOOR_MEV_FU)).sum()),
              ehull_eff_median=float(E.Ehull_eff.median()))
    return E


MOCK = False   # set in main() so compute_ehull knows the adapter kind


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=sorted(bench_models.ADAPTERS))
    ap.add_argument("--skip-xrd", action="store_true")
    args = ap.parse_args()
    ad = bench_models.ADAPTERS[args.adapter]
    tag_model = ad["tag"]
    os.makedirs(RESULTS, exist_ok=True)

    status = {"adapter": args.adapter, "model_id": ad["model_id"], "name": ad["name"],
              "tag": tag_model, "currency": ad["currency"], "status": "ok",
              "warnings": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def _finish(code):
        status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(status, open(f"{RESULTS}/anchor_bench.json", "w"), indent=1, default=str)
        print(f"[{args.adapter}] -> {RESULTS}/anchor_bench.json  status={status['status']}", flush=True)
        sys.exit(code)

    # ---- 0. the shared SQS cache must exist: never let mock random placement sneak in
    missing = [aid for aid in W.ANCHORS if not os.path.exists(W.sqs_path(aid))]
    if missing:
        status["status"] = f"sqs_missing: {missing} -> run the notebook SQS pre-generation cell first"
        _finish(1)

    # ---- 1. calculator
    calc = None
    if args.adapter != "mock":
        t0 = time.time()
        try:
            calc = bench_models.build(args.adapter)
            status["t_load_s"] = round(time.time() - t0, 1)
        except Exception as e:                    # noqa: BLE001
            status["status"] = f"adapter_error: {type(e).__name__}"
            status["error"] = str(e)
            status["traceback"] = traceback.format_exc()[-3000:]
            _finish(1)

    # ---- 1b. engine: torch-sim batch model on top of the ASE calculator, else the ASE path
    global TIMES
    engine = os.environ.get("HEO_BENCH_ENGINE", "torchsim").strip().lower()
    if args.adapter == "mock":
        engine = "ase"                                    # mock relaxes through heo_worker.relax_mock
    if engine not in ("torchsim", "ase"):
        status["warnings"].append(f"engine: unknown HEO_BENCH_ENGINE={engine!r} -> ase")
        engine = "ase"
    ts_model = None
    if engine == "torchsim":
        t0 = time.time()
        try:
            ts_model = bench_models.build_ts(args.adapter, calc)
            status["t_load_ts_s"] = round(time.time() - t0, 1)
        except Exception as e:                    # noqa: BLE001 - recorded; the ASE path still runs
            status["warnings"].append(f"engine: torch-sim model build failed ({type(e).__name__}: "
                                      f"{str(e)[:300]}) -> ASE engine")
            engine = "ase"
    if engine == "torchsim":
        tag_model = ad["tag"] + "-ts"
        TIMES = f"{W.WORKDIR}/checkpoints/bench_times_ts.csv"
        status["time_basis"] = "amortised batch wall time per structure (nsteps not exposed)"
    status["engine"], status["tag"] = engine, tag_model
    print(f"[{args.adapter}] engine={engine} tag={tag_model}", flush=True)

    # ---- 2. structures: 4 anchors x 32 (pris 2 + desod/x081/x067 10 each) + Na bcc reference
    counts_of = dict(W.ANCHORS)
    items = [it for aid, c in W.ANCHORS.items()
             for which in ("all", "x4") for it in W.build_items(aid, c, which)]
    from pymatgen.core import Structure, Lattice
    na = Structure(Lattice.cubic(4.29), ["Na", "Na"], [[0, 0, 0], [.5, .5, .5]]) * (3, 3, 3)
    items.append(("REF", "Na_bcc", na))

    # ---- 2b. Ehull extension (optional): top-N pristine pairs + in-model competitor references
    global MOCK
    MOCK = args.adapter == "mock"
    ehull_on, targets, comp_items = _ehull_inputs(status)
    # HEO_BENCH_TARGET_WHICH=x4 (default): the top-N targets get the FULL 4-point grid (32 cells
    # each, same treatment as production screening) so every pipeline observable - dE(x), ddE,
    # x*, V_seg, dV - can be compared across models on the top-N itself (user 2026-09-17).
    # "pris" restricts targets to the pristine pair (enough for Ehull and dE_pris only).
    target_which = os.environ.get("HEO_BENCH_TARGET_WHICH", "x4")
    if ehull_on:
        n_missing = 0
        for r in targets.itertuples():
            if not os.path.exists(W.sqs_path(r.comp_id)):
                n_missing += 1
                continue
            counts_of[r.comp_id] = json.loads(r.site_counts)
            items.extend(W.build_items(r.comp_id, counts_of[r.comp_id], "pris"))
            if target_which == "x4":
                items.extend(W.build_items(r.comp_id, counts_of[r.comp_id], "desod"))
                items.extend(W.build_items(r.comp_id, counts_of[r.comp_id], "x4"))
        if n_missing:
            status["warnings"].append(f"targets: {n_missing} without a shared SQS cell -> dropped")
        if not MOCK:
            items.extend(comp_items)
        per = 32 if target_which == "x4" else 2
        print(f"[{args.adapter}] targets: {len(targets) - n_missing} comps x {per} cells "
              f"({target_which}) + {0 if MOCK else len(comp_items)} hull competitor references", flush=True)

    print(f"[{args.adapter}] {len(items)} structures total "
          f"({len(W.ANCHORS)} anchors x 32 + Na ref{' + ehull set' if ehull_on else ''}) "
          f"| fmax {W.FMAX} | max_steps {W.MAX_STEPS}", flush=True)

    # ---- 3. relax (resume-safe)
    try:
        if engine == "torchsim":
            n_run, n_fail = ts_relax_items(items, ts_model, args.adapter, tag_model)
        else:
            n_run, n_fail = relax_items(items, calc, args.adapter, tag_model, counts_of)
        status["n_relaxed_now"], status["n_failed"] = n_run - n_fail, n_fail
    except Exception as e:                        # noqa: BLE001
        status["status"] = f"relax_error: {type(e).__name__}"
        status["error"] = str(e)
        status["traceback"] = traceback.format_exc()[-3000:]
        _finish(1)

    # ---- 4. E_Na (own currency; range check recorded, not asserted - spec 10-3 / 10-6)
    df = pd.read_csv(CKPT)
    ref = df[(df.comp_id == "REF") & (df.tag == "Na_bcc") & (df.model == tag_model)]
    if len(ref):
        e_na = float(ref.iloc[0].energy) / int(ref.iloc[0].natoms)
        status["E_Na"] = e_na
        if not (-1.6 < e_na < -1.0):
            status["warnings"].append(f"E_Na {e_na:.3f} eV/atom outside the PBE range (~-1.31) "
                                      "-> different currency (expected for contrast models)")
    else:
        e_na = float("nan")
        status["warnings"].append("Na reference missing -> V_avg/V_seg are NaN")

    # ---- 5. aggregate + gates
    try:
        _A4, X4 = aggregate(args.adapter, tag_model, e_na, status)
    except Exception as e:                        # noqa: BLE001
        status["status"] = f"aggregate_error: {type(e).__name__}"
        status["error"] = str(e)
        status["traceback"] = traceback.format_exc()[-3000:]
        _finish(1)
    if X4 is not None:
        status["groups"] = {g: {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in row.items()} for g, row in X4.to_dict("index").items()}

    # ---- 6. timing / convergence
    if os.path.exists(TIMES):
        t = pd.read_csv(TIMES)
        status["t_relax_median_s"] = float(t.seconds.median())
        status["t_relax_total_h"] = float(t.seconds.sum() / 3600)
    sub = df[(df.model == tag_model) & (~df.comp_id.isin(["REF", "HULL"]))]
    status["n_rows"] = int(len(sub))
    status["conv_rate"] = float(sub.conv.mean()) if len(sub) else None

    # ---- 6b. top-N per-composition 4-point table (the cross-model comparison input)
    if ehull_on and target_which == "x4":
        try:
            aggregate_targets(tag_model, targets, e_na, status)
        except Exception as e:                    # noqa: BLE001
            status["warnings"].append(f"targets aggregation failed: {type(e).__name__}: {str(e)[:200]}")

    # ---- 6c. Ehull table (v4 LP hull on this model's own energies)
    if ehull_on and (MOCK or comp_items):
        try:
            compute_ehull(tag_model, targets, status)
        except Exception as e:                    # noqa: BLE001
            status["warnings"].append(f"ehull failed: {type(e).__name__}: {str(e)[:200]}")

    # ---- 7. XRD (third phase reader; pure post-processing)
    if not args.skip_xrd:
        try:
            extra = ({r.comp_id: json.loads(r.site_counts) for r in targets.itertuples()}
                     if ehull_on and target_which == "x4" else None)
            run_xrd(args.adapter, tag_model, status, extra_counts=extra)
        except Exception as e:                    # noqa: BLE001
            status["warnings"].append(f"xrd failed: {type(e).__name__}: {str(e)[:200]}")

    _finish(0)


if __name__ == "__main__":
    main()
