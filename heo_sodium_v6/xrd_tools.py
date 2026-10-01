"""xrd_tools.py - simulated XRD post-processing of MLIP-relaxed structures (XRD_SPEC.md).

Pure post-processing: reads relaxed structures, never moves an atom, never re-relaxes,
only ADDS columns/files (XRD_SPEC section 9).  Cost is CPU minutes.

Three jobs (section 0):
  1. translate dA/dh numbers into (110)/(003) peak shifts - the experimentalist's language
  2. cross-check: d_003 must equal h_perp/3, an independent audit of the det-based decomposition
  3. a third phase reader (diffraction fingerprint) next to the letter classifier and dE

The superlattice trap (section 1): an SQS cell has dopants in ONE arrangement, which produces
superlattice peaks a real disordered solid solution does not have.  Fix: occupancy averaging
(virtual crystal) - keep the relaxed geometry, replace every TM site by the fractional TM
composition and every Na site by {Na: x}.  pymatgen then averages the scattering factors.
"""
import math
import os

import numpy as np

WAVELENGTH   = 1.5406          # CuKa, same as the Komaba 2012 reference experiment
TT_RANGE     = (10.0, 80.0)    # 2-theta window, degrees
TT_STEP      = 0.01
FWHM_DEG     = 0.15            # broadening SETTING, not physics - report it as such (section 2)
ETA_PV       = 0.5             # pseudo-Voigt mixing
SIM_TT_LO, SIM_TT_HI = 15.0, 75.0   # fingerprint window (section 4)
AMB_MARGIN   = 0.05            # |sim_O3 - sim_P3| below this -> "AMB"
D003_TOL     = 0.005           # Angstrom; |d_003(peak) - h_perp/3| beyond this = cache/cell error
PEAK_WIN     = 0.3             # degrees; search window around the expected 2-theta


# ============================================================ occupancy averaging (section 1)
def occupancy_average(struct, tm_counts, na_count, tm_set, n_na_sites=27):
    """Species -> fractional occupancies; geometry untouched.  tm_counts: {'Ni': 6, ...} site counts,
    na_count: how many Na the cell HOLDS at this x (vacancies become fractional occupancy)."""
    from pymatgen.core import Composition
    s = struct.copy()
    tm_frac = Composition(tm_counts).fractional_composition.as_dict()
    x = na_count / float(n_na_sites)
    for i, site in enumerate(s):
        el = site.specie.symbol
        if el in tm_set:
            s.replace(i, tm_frac)
        elif el == "Na":
            s.replace(i, {"Na": x})
    return s


# ============================================================ pattern (section 2)
_TT_GRID = np.arange(TT_RANGE[0], TT_RANGE[1] + TT_STEP / 2, TT_STEP)

def _pseudo_voigt(tt, tt0, fwhm=FWHM_DEG, eta=ETA_PV):
    g = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss = np.exp(-0.5 * ((tt - tt0) / g) ** 2)
    lor = 1.0 / (1.0 + ((tt - tt0) / (fwhm / 2.0)) ** 2)
    return eta * lor + (1.0 - eta) * gauss

def pattern_curve(s_avg):
    """Occupancy-averaged structure -> (tt_grid, I_curve normalised to max 100, raw peak list).
    Raw peaks: list of (two_theta, intensity, d_spacing)."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    calc = XRDCalculator(wavelength="CuKa")
    pat = calc.get_pattern(s_avg, two_theta_range=TT_RANGE, scaled=True)
    curve = np.zeros_like(_TT_GRID)
    for tt0, inten, d in zip(pat.x, pat.y, pat.d_hkls):
        lo = np.searchsorted(_TT_GRID, tt0 - 8 * FWHM_DEG)
        hi = np.searchsorted(_TT_GRID, tt0 + 8 * FWHM_DEG)
        curve[lo:hi] += inten * _pseudo_voigt(_TT_GRID[lo:hi], tt0)
    if curve.max() > 0:
        curve *= 100.0 / curve.max()
    peaks = list(zip([float(v) for v in pat.x], [float(v) for v in pat.y],
                     [float(v) for v in pat.d_hkls]))
    return _TT_GRID, curve, peaks


# ============================================================ peak tracking (section 3)
def cell_metrics(cell):
    """Duplicate of heo_worker.cell_metrics (pure math): V, in-plane area, tilt-immune height."""
    cell = np.asarray(cell, dtype=float)
    a, b, _c = cell
    V = abs(np.linalg.det(cell))
    A_par = np.linalg.norm(np.cross(a, b))
    return V, A_par, V / A_par

def bragg_tt(d):
    s = WAVELENGTH / (2.0 * d)
    if not 0 < s < 1:
        return float("nan")
    return 2.0 * math.degrees(math.asin(s))

def _peak_in_window(tt_grid, curve, tt_expect, win=PEAK_WIN):
    """Max peak near the expected position, refined to sub-grid precision by a 3-point parabola:
    at low angle the 0.01-degree grid alone quantises d_003 by ~0.004 A, which would trip the
    0.005 A audit (section 3) with no physical error behind it."""
    if not np.isfinite(tt_expect):
        return float("nan"), float("nan")
    m = (tt_grid >= tt_expect - win) & (tt_grid <= tt_expect + win)
    if not m.any() or curve[m].max() <= 0:
        return float("nan"), float("nan")
    idx = np.flatnonzero(m)
    j = idx[np.argmax(curve[idx])]
    tt0, i0 = float(tt_grid[j]), float(curve[j])
    if 0 < j < len(curve) - 1:
        y1, y2, y3 = curve[j - 1], curve[j], curve[j + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom < 0:                              # a genuine maximum
            shift = 0.5 * (y1 - y3) / denom
            if abs(shift) <= 1.0:
                tt0 += shift * TT_STEP
    return tt0, i0

def _n_maxima_in_window(tt_grid, curve, tt_expect, win=PEAK_WIN, rel_height=0.3):
    """Local maxima above rel_height * window-max: 2+ means the peak is split (section 3)."""
    if not np.isfinite(tt_expect):
        return 0
    m = (tt_grid >= tt_expect - win) & (tt_grid <= tt_expect + win)
    c = curve[m]
    if len(c) < 3 or c.max() <= 0:
        return 0
    is_max = (c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:]) & (c[1:-1] > rel_height * c.max())
    return int(is_max.sum())

def _raw_peak_near(peaks, tt_expect, win=PEAK_WIN):
    """Strongest RAW (delta) reflection within the window: exact Bragg position, no broadening.
    The broadened-curve maximum can shift by ~0.01 deg when a neighbouring reflection overlaps
    asymmetrically inside the pseudo-Voigt tails - cosmetic, not physical - so the section-3
    audit is done on the raw list (d(00l) = h_perp/l is a lattice identity and must hold exactly)."""
    if not np.isfinite(tt_expect):
        return None
    cand = [(inten, tt, d) for tt, inten, d in peaks if abs(tt - tt_expect) <= win]
    if not cand:
        return None
    inten, tt, d = max(cand)
    return tt, inten, d

def track_peaks(tt_grid, curve, cell, peaks=None):
    """(110)/(003)/(104) positions expected from the lattice matrix vs found in the pattern.
    Supercell hkl indices differ from the experimental hexagonal cell, so everything is matched
    through d-spacings computed from the exact cell decomposition:
        a_eff = sqrt(2*A_par / (9*sqrt(3)))   (3x3 in-plane supercell)
        c_eff = h_perp                        (3-slab cell height)
    Positions/d come from the raw reflection list when available (exact), intensities and the
    110-split flag from the broadened curve (what a diffractogram would show)."""
    V, A_par, h_perp = cell_metrics(cell)
    a_eff = math.sqrt(2.0 * A_par / (9.0 * math.sqrt(3.0)))
    c_eff = h_perp
    d003, d110 = c_eff / 3.0, a_eff / 2.0
    inv_d104_sq = (4.0 / 3.0) * 1.0 / a_eff ** 2 + 16.0 / c_eff ** 2
    d104 = 1.0 / math.sqrt(inv_d104_sq)
    out = {"a_eff": a_eff, "c_eff": c_eff}
    vals = {}
    for name, d_exp in (("003", d003), ("110", d110), ("104", d104)):
        tt_exp = bragg_tt(d_exp)
        raw = _raw_peak_near(peaks, tt_exp) if peaks else None
        if raw is not None:
            tt_f, i_f, d_f = raw[0], raw[1], raw[2]
        else:
            tt_f, i_f = _peak_in_window(tt_grid, curve, tt_exp)
            d_f = WAVELENGTH / (2 * math.sin(math.radians(tt_f / 2))) if np.isfinite(tt_f) else float("nan")
        vals[name] = (tt_f, i_f, d_f)
        out[f"tt_{name}"] = tt_f
    out["d_003"], out["d_110"] = vals["003"][2], vals["110"][2]
    i003, i104 = vals["003"][1], vals["104"][1]
    out["I_104_rel"] = (i104 / i003) if (np.isfinite(i104) and np.isfinite(i003) and i003 > 0) else float("nan")
    # the audit at the heart of section 3: the pattern and the det decomposition must agree
    out["chk_d003_resid"] = abs(out["d_003"] - h_perp / 3.0) if np.isfinite(out["d_003"]) else float("nan")
    out["xrd_split_110"] = _n_maxima_in_window(tt_grid, curve, bragg_tt(d110)) >= 2
    return out


# ============================================================ phase fingerprint (section 4)
def _to_dspace_curve(tt_grid, curve, d003_anchor):
    """Resample I(2-theta) onto a fixed log-d grid after shift-normalising so the (003) of every
    pattern lands on the same coordinate: lattices differ between compositions, so similarity
    must compare PATTERN SHAPE, not absolute peak positions."""
    d = WAVELENGTH / (2.0 * np.sin(np.radians(tt_grid / 2.0)))
    m = (tt_grid >= SIM_TT_LO) & (tt_grid <= SIM_TT_HI)
    logd = np.log(d[m]) - math.log(d003_anchor)          # shift-normalise by the (003) d-spacing
    grid = np.linspace(-2.2, 0.35, 2000)                 # covers d/d003 from ~0.11 to ~1.4
    return np.interp(grid, logd[::-1], curve[m][::-1], left=0.0, right=0.0)

def cosine_sim(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))

def make_reference(tt_grid, curve, d003):
    return _to_dspace_curve(tt_grid, curve, d003)

def fingerprint(tt_grid, curve, d003, ref_O3, ref_P3):
    v = _to_dspace_curve(tt_grid, curve, d003)
    sim_o3, sim_p3 = cosine_sim(v, ref_O3), cosine_sim(v, ref_P3)
    if not (np.isfinite(sim_o3) and np.isfinite(sim_p3)):
        phase = "n/a"
    elif abs(sim_o3 - sim_p3) < AMB_MARGIN:
        phase = "AMB"
    else:
        phase = "O3" if sim_o3 > sim_p3 else "P3"
    return {"xrd_sim_O3": sim_o3, "xrd_sim_P3": sim_p3, "xrd_phase": phase}


# ============================================================ driver
def xrd_one(struct, tm_counts, na_count, tm_set, cell=None):
    """One relaxed structure -> occupancy average -> pattern + peak tracking.
    cell defaults to the structure's own lattice (identical when the CIF round-trips)."""
    s_avg = occupancy_average(struct, tm_counts, na_count, tm_set)
    tt, curve, peaks = pattern_curve(s_avg)
    cell = np.asarray(cell if cell is not None else struct.lattice.matrix, dtype=float)
    row = track_peaks(tt, curve, cell, peaks=peaks)
    return tt, curve, peaks, row


def run_xrd(comp_ids, counts_of, recs_of, loader, out_dir, x_stems, x_of, n_vac=5,
            tm_set=None, ref_comp_candidates=("HOST_v1", "HOST_v2"), verbose=True):
    """XRD table over many compositions.  Long format: one row per (comp_id, stem, phase).

    comp_ids   iterable of composition ids (anchors and/or screening comps)
    counts_of  {comp_id: {'Ni': 6, ...}} TM site counts
    recs_of    {comp_id: {tag: rec}} from the energy checkpoints - k_best/P_order/cell come from here
    loader     callable (comp_id, tag) -> pymatgen Structure or None (relaxed CIF lookup)
    out_dir    npz patterns land in {out_dir}/xrd/{comp_id}/{stem}_{phase}_{kbest|avg5}.npz
    x_stems    {'pris': 0, 'x081': 5, ...} Na removed per stem (heo_worker.X_STEMS)
    x_of       {'pris': 1.0, ...} (heo_worker.X_OF)

    Reference fingerprints (section 4): the first available candidate's pristine O3/P3 pattern.
    Returns (DataFrame, refs_dict)."""
    import pandas as pd
    if tm_set is None:
        tm_set = set()
        for c in counts_of.values():
            tm_set |= set(c)

    def _tags(stem, phase):
        if stem == "pris":
            return [f"{phase}_pris"]
        return [f"{phase}_{stem}_{k}" for k in range(n_vac)]

    def _kbest(comp_id, stem):
        recs = recs_of.get(comp_id, {})
        pairs = []
        for k in range(n_vac):
            o, p = recs.get(f"O3_{stem}_{k}"), recs.get(f"P3_{stem}_{k}")
            if o and p:
                pairs.append((p["energy"] - o["energy"], k))
        return min(pairs)[1] if pairs else None

    # ---- reference fingerprints from the HOST pristine relaxed structures
    refs, ref_d003 = {}, {}
    for cand in ref_comp_candidates:
        if cand not in counts_of:
            continue
        ok = True
        for phase in ("O3", "P3"):
            st = loader(cand, f"{phase}_pris")
            if st is None:
                ok = False
                break
            tt, curve, _pk, row = xrd_one(st, counts_of[cand], 27 - x_stems["pris"], tm_set)
            refs[phase] = make_reference(tt, curve, row["d_003"])
            ref_d003[phase] = row["d_003"]
        if ok:
            if verbose:
                print(f"[xrd] reference fingerprints from {cand} pristine O3/P3 "
                      f"(d003 O3 {ref_d003.get('O3', float('nan')):.4f} A)")
            break
        refs = {}
    if not refs and verbose:
        print("[xrd] WARNING: no reference structure found -> xrd_phase columns will be n/a")

    rows = []
    for ci, comp_id in enumerate(comp_ids):
        counts = counts_of.get(comp_id)
        if counts is None:
            continue
        for stem, n_rm in x_stems.items():
            na_count = 27 - n_rm
            kb = None if stem == "pris" else _kbest(comp_id, stem)
            for phase in ("O3", "P3"):
                tag_kb = f"{phase}_pris" if stem == "pris" else (None if kb is None else f"{phase}_{stem}_{kb}")
                if tag_kb is None:
                    continue
                st = loader(comp_id, tag_kb)
                if st is None:
                    continue
                rec = recs_of.get(comp_id, {}).get(tag_kb, {})
                cell = rec.get("cell")
                if isinstance(cell, str) and cell.startswith("["):
                    import json as _json
                    cell = np.asarray(_json.loads(cell), dtype=float)
                elif not isinstance(cell, np.ndarray):
                    cell = None
                tt, curve, peaks, row = xrd_one(st, counts, na_count, tm_set, cell=cell)
                row.update(comp_id=comp_id, stem=stem, x=round(x_of[stem], 4), phase=phase,
                           k_best=(kb if stem != "pris" else -1))
                if refs:
                    row.update(fingerprint(tt, curve, row["d_003"], refs["O3"], refs["P3"]))
                    p_ord = rec.get("P_order", float("nan"))
                    if np.isfinite(p_ord) and row["xrd_phase"] in ("O3", "P3"):
                        row["xrd_agree_letter"] = row["xrd_phase"] == ("P3" if p_ord > 0 else "O3")
                    else:
                        row["xrd_agree_letter"] = None
                # 5-sample average curve (vacancy stems only)
                curves = [curve]
                if stem != "pris":
                    for k in range(n_vac):
                        if k == kb:
                            continue
                        st_k = loader(comp_id, f"{phase}_{stem}_{k}")
                        if st_k is not None:
                            _t, c_k, _p, _r = xrd_one(st_k, counts, na_count, tm_set)
                            curves.append(c_k)
                row["n_avg"] = len(curves)
                avg = np.mean(curves, axis=0)
                d = f"{out_dir}/xrd/{comp_id}"
                os.makedirs(d, exist_ok=True)
                suffix = "kbest" if stem != "pris" else "single"
                np.savez_compressed(f"{d}/{stem}_{phase}_{suffix}.npz", tt=tt, I=curve,
                                    peaks=np.array(peaks))
                if len(curves) > 1:
                    np.savez_compressed(f"{d}/{stem}_{phase}_avg{len(curves)}.npz", tt=tt, I=avg)
                rows.append(row)
        if verbose and (ci + 1) % 10 == 0:
            print(f"[xrd] {ci + 1}/{len(list(comp_ids)) if hasattr(comp_ids, '__len__') else '?'} compositions")
    df = pd.DataFrame(rows)
    return df, refs


# ============================================================ HOST pilot direction test (section 5)
def host_direction_test(df, host_ids=("HOST_v1", "HOST_v2"), stems_desc=("pris", "x081", "x067", "desod")):
    """Komaba 2012 qualitative directions on the HOST waterfall (absolute 2-theta agreement is NOT
    expected - 0 K, currency offset; directions and ordering only):
      (003) moves to LOWER angle as x decreases (interlayer expansion)
      (110) moves to HIGHER angle as x decreases (in-plane contraction)
    Tested on the O3 branch (the discharge path).  Returns dict of booleans + the tracks."""
    sub = df[df.comp_id.isin(host_ids) & (df.phase == "O3")]
    if sub.empty:
        return {"xrd_dir_ok_003": None, "xrd_dir_ok_110": None, "tt_003_track": {}, "tt_110_track": {}}
    t003, t110 = {}, {}
    for stem in stems_desc:
        s = sub[sub.stem == stem]
        if len(s):
            t003[stem] = float(np.nanmean(s.tt_003))
            t110[stem] = float(np.nanmean(s.tt_110))
    seq3 = [t003[s] for s in stems_desc if s in t003 and np.isfinite(t003[s])]
    seq1 = [t110[s] for s in stems_desc if s in t110 and np.isfinite(t110[s])]
    tol = 0.005                              # degrees; below this two positions are a tie
    ok3 = all(b <= a + tol for a, b in zip(seq3, seq3[1:])) if len(seq3) >= 2 else None
    ok1 = all(b >= a - tol for a, b in zip(seq1, seq1[1:])) if len(seq1) >= 2 else None
    return {"xrd_dir_ok_003": ok3, "xrd_dir_ok_110": ok1,
            "tt_003_track": t003, "tt_110_track": t110}
