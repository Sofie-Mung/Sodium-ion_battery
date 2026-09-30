"""sse_worker.py - Li6PS5Cl dual-doping MLIP screening: every function of the pipeline.

Implements SSE_PIPELINE_SPEC.md (2026-09-27).  The notebook `sse_screening.ipynb` only
orchestrates; all physics, structure generation, relaxation, hull, MD, kinisi analysis,
aggregation, figure (pptx) and DFT export code lives here.

Worker usage (one process per GPU, spawned by the notebook launcher):
    CUDA_VISIBLE_DEVICES=<k> python sse_worker.py --task relax --model orb_v3 \
        --jobs outputs/jobs/<name>.json --worker-id <k> --n-workers <N> --config outputs/config.json
    CUDA_VISIBLE_DEVICES=<k> python sse_worker.py --task md    --model orb_v3 ...
    python sse_worker.py --task analyze --jobs ... --worker-id <k> --n-workers <N>   (CPU only)

Absolute rules (spec section 0) enforced in code:
  * the two models never share a hull or a ratio (every function takes an explicit model name
    and every energy row carries `model`);
  * dopants are substituted INSIDE the 2x2x2 supercell (no unit-cell substitution + tiling);
  * the S/Cl disorder count is a single constant for every composition;
  * invariants INV-1..INV-9 are hard asserts that abort the whole run;
  * no sigma is ever written without a confidence interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================ constants
VALENCE = {"Li": 1, "Na": 1, "Ag": 1, "Cu": 1, "Mg": 2, "Zn": 2, "Ca": 2,
           "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3, "Ge": 4, "P": 5, "S": -2, "Cl": -1}
LI_SITE_POOL = ["Na", "Ag", "Cu", "Mg", "Zn", "Ca", "Al", "Ga", "In", "Sc", "Y"]   # spec 4.3
REDOX_RISK = {"Cu"}
K_B_EV = 8.617333262e-5          # eV/K
E_CHARGE = 1.602176634e-19       # C
K_B_J = 1.380649e-23             # J/K
N_LI_HOST, N_P_HOST, N_S_HOST, N_CL_HOST, N_ATOMS_HOST = 192, 32, 160, 32, 416
N_UC_IN_SUPERCELL = 8
REQUIRED_REFERENCE_FORMULAS = ["Li2S", "LiCl", "Li3PS4", "P2S5", "Li4GeS4", "GeS2"]
DOPANT_REFERENCE_FORMULAS = {   # sulfide, chloride of every Li-site dopant (spec 7.4-1)
    "Na": ["Na2S", "NaCl"], "Ag": ["Ag2S", "AgCl"], "Cu": ["Cu2S", "CuCl"],
    "Mg": ["MgS", "MgCl2"], "Zn": ["ZnS", "ZnCl2"], "Ca": ["CaS", "CaCl2"],
    "Al": ["Al2S3", "AlCl3"], "Ga": ["Ga2S3", "GaCl3"], "In": ["In2S3", "InCl3"],
    "Sc": ["Sc2S3", "ScCl3"], "Y": ["Y2S3", "YCl3"],
}

DEFAULT_CONFIG = {
    "seed": 20260927,
    "workdir": ".",
    "host_cif": "inputs/Li6PS5Cl.cif",
    "exp_reference_csv": "inputs/exp_reference.csv",
    "supercell": [2, 2, 2],
    # anion disorder: fraction of 4c sites occupied by Cl (same for ALL compositions)
    "scl_disorder_4c_cl_fraction": 0.375,      # 12 of 32 swaps in 2x2x2
    # Li 48h pair detection
    "li48h_pair_cutoff": 2.0,                  # Angstrom; verified with a distance histogram (5.2)
    # if the CIF has Li on 24g (doublet midpoint) instead of 48h, the pair sampling of spec 5.2 is
    # impossible.  The loader refuses such a file unless the user explicitly allows it here.
    "allow_24g_host": False,
    # P-site cap
    "max_ge_per_supercell": 16,
    # structure generation
    "n_configs_pilot": 8,                      # 4 random + 2 clustered + 2 dispersed
    "n_configs_stage2": 5,                     # 3 random + 1 clustered + 1 dispersed
    "min_interatomic_dist": 1.6,
    "interstitial_min_cation_dist": 2.0,
    "interstitial_min_anion_dist": 1.9,
    "vacancy_near_radius": 4.0,
    # relax
    "relax_fmax": 0.02,
    "relax_max_steps": 1000,
    "relax_cell": True,
    "relax_chunk": 16,                         # structures per torch-sim optimize call
    # config selection for MD
    "t_syn": 800.0,
    "n_configs_md_pilot": 2,                   # USER DECISION 2026-09-28 (was 3): 4-8 h budget on 4 GPUs
    "n_configs_md_stage2_short": 2,
    # MD
    "md_timestep_fs": 2.0,                     # USER DECISION 2026-09-28 (was 1.0)
    "md_temperatures_pilot": [600, 750, 900],  # USER DECISION 2026-09-28 (was 500/600/700); 600 K stays the gate temperature
    "md_npt_ps": 10.0,                         # USER DECISION 2026-09-28 (was 20)
    "md_nvt_ps_pilot": 100.0,                  # USER DECISION 2026-09-28 (was 2000): screening resolution ~1.5x in sigma ratio
    "md_nvt_ps_stage2_short": 500.0,
    "md_stage2_short_T": 600,
    "md_save_every_fs": 100.0,
    "md_thermostat": "nose_hoover",            # fallback: langevin with friction <= 0.002 /fs
    "md_langevin_friction_per_fs": 0.002,
    "md_batch_size": 8,                        # systems per ts.integrate call on one GPU (OOM -> halves automatically)
    "md_keep_h5": False,                       # torch-sim h5 is converted to npz; keep the h5 too?
    "md_nondiffusive_msd_A2": 20.0,
    # analysis
    "kinisi_start_dt_ps": 2.0,
    "kinisi_n_dt": 120,
    "kinisi_n_samples": 1000,
    "kinisi_n_walkers": 32,
    "kinisi_n_burn": 500,
    "kinisi_n_thin": 10,
    "ci_level": 0.95,
    "t_extrapolate": 300.0,
    "n_mc_samples": 4000,                      # Monte-Carlo draws for Arrhenius / ratio propagation
    # hull
    "hull_mp_ehull_max": 0.05,
    "hull_min_ref_atoms": 24,                  # tiny reference cells are expanded (autobatcher stability)
    # models
    "models": ["orb_v3"],                      # USER DECISION 2026-09-28: ORB only (SevenNet 0.23 ns/day, OOM at batch 4)
    "orb_checkpoint": "orb_v3_conservative_inf_mpa",
    "orb_precision": "float32-high",
    "sevenn_checkpoint": "7net-omni",
    "sevenn_modal": "mpa",
    "sevenn_fallback_checkpoint": "7net-mf-ompa",
    "stab_threshold_eV": None,                 # G-stab: set explicitly after Stage I (user-confirmed)
    "auto_terminate": True,
}

MC_MODES_PILOT = ["random"] * 4 + ["clustered"] * 2 + ["dispersed"] * 2
MC_MODES_STAGE2 = ["random"] * 3 + ["clustered"] * 1 + ["dispersed"] * 1


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """Return DEFAULT_CONFIG updated with a JSON file and/or an override dict."""
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        cfg.update(json.load(open(path)))
    if overrides:
        cfg.update(overrides)
    cfg["supercell"] = tuple(int(x) for x in cfg["supercell"])
    return cfg


def wpath(cfg: dict, *parts: str) -> str:
    """Path inside the workdir; parent directories are created."""
    p = Path(cfg["workdir"], *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def derived_seed(base: int, *keys) -> int:
    """Deterministic 31-bit seed derived from the master seed and any hashable keys."""
    h = hashlib.sha256(("|".join([str(base)] + [str(k) for k in keys])).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def file_md5(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def atomic_write_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ============================================================================ compositions
def comp_key_counts(comp: dict) -> tuple[dict, dict]:
    li_site = {k: int(v) for k, v in (comp.get("li_site") or {}).items() if int(v) > 0}
    p_site = {k: int(v) for k, v in (comp.get("p_site") or {}).items() if int(v) > 0}
    return li_site, p_site


def n_li_of(comp: dict) -> int:
    """Charge ledger (spec 1.2): N_Li = 192 - sum(n_i k_i) + sum((5 - v_j) g_j)."""
    li_site, p_site = comp_key_counts(comp)
    n = N_LI_HOST
    n -= sum(VALENCE[el] * k for el, k in li_site.items())
    n += sum((5 - VALENCE[el]) * g for el, g in p_site.items())
    return int(n)


def li_delta(comp: dict) -> int:
    """delta = N_Li - (192 - sum k_i): <0 vacancies, >0 interstitial insertions."""
    li_site, _ = comp_key_counts(comp)
    return n_li_of(comp) - (N_LI_HOST - sum(li_site.values()))


def valence_combo(comp: dict) -> str:
    li_site, _ = comp_key_counts(comp)
    vals = sorted(VALENCE[el] for el, k in li_site.items() for _ in range(max(1, k // 8)))
    return "(" + ",".join(str(v) for v in vals) + ")" if vals else "()"


def comp_summary_row(comp: dict) -> dict:
    li_site, p_site = comp_key_counts(comp)
    return dict(
        composition_id=comp["id"], stage=comp.get("stage", ""), role=comp.get("role", ""),
        li_site=json.dumps(li_site, sort_keys=True), p_site=json.dumps(p_site, sort_keys=True),
        n_li=n_li_of(comp), n_li_per_uc=n_li_of(comp) / N_UC_IN_SUPERCELL, delta=li_delta(comp),
        li_group=n_li_of(comp), valence_combo=valence_combo(comp),
        flag_redox_risk=any(el in REDOX_RISK for el in li_site),
        n_ge=sum(p_site.values()),
    )


def pilot_compositions() -> list[dict]:
    """The 9 Stage I compositions (spec 4.2), supercell counts."""
    return [
        {"id": "A", "li_site": {}, "p_site": {}, "stage": "I", "role": "host anchor"},
        {"id": "F", "li_site": {}, "p_site": {"Ge": 16}, "stage": "I", "role": "Ge trend (gate G-a)"},
        {"id": "N0", "li_site": {"Na": 24}, "p_site": {}, "stage": "I", "role": "monovalent, uncompensated"},
        {"id": "N2", "li_site": {"Na": 24}, "p_site": {"Ge": 16}, "stage": "I", "role": "monovalent, compensated"},
        {"id": "M0", "li_site": {"Mg": 24}, "p_site": {}, "stage": "I", "role": "divalent, uncompensated"},
        {"id": "M2", "li_site": {"Mg": 24}, "p_site": {"Ge": 16}, "stage": "I", "role": "divalent, compensated"},
        {"id": "L0", "li_site": {"Al": 24}, "p_site": {}, "stage": "I", "role": "trivalent, uncompensated"},
        {"id": "L2", "li_site": {"Al": 24}, "p_site": {"Ge": 16}, "stage": "I", "role": "trivalent, compensated"},
        {"id": "D", "li_site": {"Mg": 4}, "p_site": {"Ge": 8}, "stage": "I", "role": "iso-Li with A (gate G-c)"},
    ]


PILOT_LEDGER_EXPECTED = {   # spec 4.2 table: (192 - sum k, N_Li, delta)
    "A": (192, 192, 0), "F": (192, 208, 16), "N0": (168, 168, 0), "N2": (168, 184, 16),
    "M0": (168, 144, -24), "M2": (168, 160, -8), "L0": (168, 120, -48), "L2": (168, 136, -32),
    "D": (188, 192, 4),
}


def check_pilot_ledger() -> pd.DataFrame:
    """Unit test of the charge ledger against the spec 4.2 table (spec 12-2)."""
    rows = []
    for comp in pilot_compositions():
        li_site, _ = comp_key_counts(comp)
        got = (N_LI_HOST - sum(li_site.values()), n_li_of(comp), li_delta(comp))
        exp = PILOT_LEDGER_EXPECTED[comp["id"]]
        assert got == exp, f"charge ledger mismatch for {comp['id']}: got {got}, expected {exp}"
        rows.append(dict(composition_id=comp["id"], li_after_sub=got[0], n_li=got[1], delta=got[2]))
    return pd.DataFrame(rows)


def stage2_compositions(ge_levels=(0, 8, 16)) -> tuple[list[dict], list[dict]]:
    """Stage II compositions (spec 4.3).  Returns (single-dopant baselines, mixtures)."""
    from itertools import combinations
    baselines, mixtures = [], []
    for el in LI_SITE_POOL:
        for g in ge_levels:
            baselines.append({"id": f"S_{el}_Ge{g}", "li_site": {el: 24}, "p_site": {"Ge": g} if g else {},
                              "stage": "II-baseline", "role": "single-dopant baseline"})
    for trio in combinations(LI_SITE_POOL, 3):
        for g in ge_levels:
            mixtures.append({"id": f"T_{'-'.join(trio)}_Ge{g}", "li_site": {el: 8 for el in trio},
                             "p_site": {"Ge": g} if g else {}, "stage": "II", "role": "ternary 1:1:1"})
    for a in ["Na", "Ag", "Cu"]:
        for b in LI_SITE_POOL:
            if b == a:
                continue
            for g in ge_levels:
                mixtures.append({"id": f"B_{a}2-{b}_Ge{g}", "li_site": {a: 16, b: 8},
                                 "p_site": {"Ge": g} if g else {}, "stage": "II", "role": "binary 2:1 (monovalent-rich)"})
    assert len(baselines) == 33 and len(mixtures) == 585, (len(baselines), len(mixtures))
    return baselines, mixtures


def chemsys_elements(comp: dict) -> list[str]:
    li_site, p_site = comp_key_counts(comp)
    return sorted({"Li", "P", "S", "Cl"} | set(li_site) | set(p_site))


def s_syn_denominator_terms(comp: dict) -> list[tuple[str, float]]:
    """(baseline composition id, weight k_i/K) pairs for the geometric-mean denominator (spec 8.5)."""
    li_site, p_site = comp_key_counts(comp)
    K = sum(li_site.values())
    g = p_site.get("Ge", 0)
    return [(f"S_{el}_Ge{g}", k / K) for el, k in li_site.items()]


# ============================================================================ host structure
@dataclass
class Host:
    """Ordered host template with site labels (unit cell and supercell)."""
    unit: object                      # pymatgen Structure, ordered, Li on ALL candidate sites
    labels_unit: list[str]
    supercell: object                 # pymatgen Structure template (Li candidates + framework)
    labels: list[str]
    li_site_mode: str                 # "48h_pairs" | "24g"
    li_pairs: np.ndarray              # (n_pairs, 2) template indices; for 24g: (192, 1)
    sym_ops_frac: list                # framework symmetry operations (unit-cell fractional)
    a: float
    pair_stats: dict = field(default_factory=dict)
    source_md5: str = ""


def _wyckoff_labels(struct, symprec=1e-3) -> tuple[list[str], int, str]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(struct, symprec=symprec)
    ds = sga.get_symmetry_dataset()
    eq = ds.equivalent_atoms
    counts = {}
    for e in eq:
        counts[e] = counts.get(e, 0) + 1
    labels = [f"{counts[eq[i]]}{ds.wyckoffs[i]}" for i in range(len(struct))]
    return labels, sga.get_space_group_number(), sga.get_space_group_symbol()


def load_host(cfg: dict, log_dir: str | None = None) -> Host:
    """Read the user CIF, verify F-43m, label Wyckoff sites, build the ordered template supercell.

    Spec 5.1/5.2.  Coordinates are never hard-coded; everything comes from the CIF.
    """
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cif = os.path.join(cfg["workdir"], cfg["host_cif"]) if not os.path.isabs(cfg["host_cif"]) else cfg["host_cif"]
    if not os.path.exists(cif):
        raise FileNotFoundError(f"host CIF not found: {cif} -> the user must provide inputs/Li6PS5Cl.cif")
    raw = Structure.from_file(cif)
    labels_raw, sg_num, sg_sym = _wyckoff_labels(raw)
    assert sg_num == 216, f"host space group is {sg_sym} ({sg_num}), expected F-43m (216)"
    lat = raw.lattice
    assert abs(lat.a - lat.b) < 1e-3 and abs(lat.a - lat.c) < 1e-3 and all(abs(x - 90) < 1e-2 for x in lat.angles), \
        "host cell is not the cubic conventional cell"
    a = float(lat.a)

    # ---- collect sites by role.  Partial occupancies (48h 0.5, mixed 4a/4c) are resolved here.
    species_unit, frac_unit, labels_unit = [], [], []
    li_label = None
    for site, lab in zip(raw, labels_raw):
        els = {str(sp.element) if hasattr(sp, "element") else str(sp): occ for sp, occ in site.species.items()}
        if "P" in els:
            assert lab == "4b", f"P found on {lab}, expected 4b"
            species_unit.append("P"); labels_unit.append("P_4b")
        elif "Li" in els:
            assert lab in ("48h", "24g"), f"Li found on Wyckoff {lab}; only 48h (pairs) or 24g (midpoints) are supported"
            li_label = lab
            species_unit.append("Li"); labels_unit.append(f"Li_{lab}")
        elif lab == "16e":
            assert "S" in els, f"16e site is {els}, expected S"
            species_unit.append("S"); labels_unit.append("S_16e")
        elif lab == "4a":                       # ordered reference: Cl on 4a
            assert set(els) <= {"S", "Cl"}, f"4a site is {els}"
            species_unit.append("Cl"); labels_unit.append("Cl_4a")
        elif lab == "4c":                       # ordered reference: free S on 4c
            assert set(els) <= {"S", "Cl"}, f"4c site is {els}"
            species_unit.append("S"); labels_unit.append("S_4c")
        else:
            raise AssertionError(f"unexpected site {els} on Wyckoff {lab}")
        frac_unit.append(np.array(site.frac_coords) % 1.0)
    counts = pd.Series(labels_unit).value_counts().to_dict()
    n_li_unit = counts.get("Li_48h", 0) + counts.get("Li_24g", 0)
    assert counts.get("P_4b") == 4 and counts.get("S_16e") == 16 and counts.get("Cl_4a") == 4 \
        and counts.get("S_4c") == 4, f"framework site counts wrong: {counts}"
    if li_label == "48h":
        assert n_li_unit == 48, f"expected 48 Li 48h candidate sites per unit cell, got {n_li_unit}"
        mode = "48h_pairs"
    else:
        assert n_li_unit == 24, f"expected 24 Li 24g sites per unit cell, got {n_li_unit}"
        mode = "24g"
        if not cfg.get("allow_24g_host", False):
            raise AssertionError(
                "The host CIF places Li on 24g (the midpoint of the 48h doublet), so the 48h pair "
                "sampling of spec 5.2 cannot be performed.  Provide an experimental CIF with Li on 48h "
                "(partial occupancy 0.5), or set CONFIG['allow_24g_host']=True to accept the ordered "
                "24g host (deviation from spec 5.2 - user decision).")
    unit = Structure(Lattice.cubic(a), species_unit, frac_unit)

    # ---- framework symmetry operations (for labelling interstitial candidates)
    fw_idx = [i for i, l in enumerate(labels_unit) if not l.startswith("Li")]
    framework = Structure(unit.lattice, [species_unit[i] for i in fw_idx], [frac_unit[i] for i in fw_idx])
    sga_fw = SpacegroupAnalyzer(framework, symprec=1e-3)
    assert sga_fw.get_space_group_number() == 216, "ordered framework (Cl 4a / S 4c) lost F-43m symmetry"
    ops = sga_fw.get_symmetry_operations(cartesian=False)

    # ---- template supercell (explicit image loop so that labels follow the sites)
    nx, ny, nz = cfg["supercell"]
    assert (nx, ny, nz) == (2, 2, 2), "the spec fixes the 2x2x2 supercell (416 atoms)"
    sp_s, fr_s, lab_s = [], [], []
    for i in range(len(unit)):
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    sp_s.append(species_unit[i])
                    fr_s.append((frac_unit[i] + np.array([ix, iy, iz])) / np.array([nx, ny, nz]))
                    lab_s.append(labels_unit[i])
    supercell = Structure(Lattice.cubic(a * nx), sp_s, fr_s)

    # ---- Li pairs
    li_idx = np.array([i for i, l in enumerate(lab_s) if l.startswith("Li")])
    pair_stats = {}
    if mode == "48h_pairs":
        dm = supercell.lattice.get_all_distances(supercell.frac_coords[li_idx], supercell.frac_coords[li_idx])
        np.fill_diagonal(dm, np.inf)
        cutoff = float(cfg["li48h_pair_cutoff"])
        hist, edges = np.histogram(dm[np.triu_indices(len(li_idx), 1)], bins=np.arange(0.0, 6.05, 0.05))
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(log_dir, "li48h_pair_hist.txt"), "w") as f:
                f.write("# Li 48h - Li 48h distance histogram in the 2x2x2 template (all 384 candidate sites)\n")
                f.write(f"# pair cutoff = {cutoff} A\n# d_low(A) d_high(A) count\n")
                for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
                    if h:
                        f.write(f"{lo:.2f} {hi:.2f} {h}\n")
        nn = np.sort(dm, axis=1)
        d_intra_max = float(nn[:, 0].max()); d_inter_min = float(nn[:, 1].min())
        assert d_intra_max < cutoff < d_inter_min, (
            f"li48h_pair_cutoff={cutoff} does not separate intra-pair (max {d_intra_max:.3f} A) from "
            f"inter-pair (min {d_inter_min:.3f} A) distances -> inspect logs/li48h_pair_hist.txt")
        partner = np.argmin(dm, axis=1)
        assert all(partner[partner[i]] == i for i in range(len(li_idx))), "48h pairing is not mutual"
        pairs = sorted({tuple(sorted((int(li_idx[i]), int(li_idx[partner[i]])))) for i in range(len(li_idx))})
        li_pairs = np.array(pairs, dtype=int)
        assert len(li_pairs) == N_LI_HOST, f"expected 192 pairs, got {len(li_pairs)}"
        pair_stats = dict(d_intra_min=float(nn[:, 0].min()), d_intra_max=d_intra_max, d_inter_min=d_inter_min)
    else:
        li_pairs = li_idx.reshape(-1, 1)
        assert len(li_pairs) == N_LI_HOST
        dm = supercell.lattice.get_all_distances(supercell.frac_coords[li_idx], supercell.frac_coords[li_idx])
        np.fill_diagonal(dm, np.inf)
        pair_stats = dict(d_li_li_min=float(dm.min()))

    return Host(unit=unit, labels_unit=labels_unit, supercell=supercell, labels=lab_s, li_site_mode=mode,
                li_pairs=li_pairs, sym_ops_frac=ops, a=a, pair_stats=pair_stats, source_md5=file_md5(cif))


# ============================================================================ structure generation
def _min_dist_to(lattice, frac_pts, frac_targets) -> np.ndarray:
    """Minimum PBC distance from each of frac_pts to any of frac_targets (inf if no targets)."""
    if len(frac_targets) == 0:
        return np.full(len(frac_pts), np.inf)
    return lattice.get_all_distances(np.asarray(frac_pts), np.asarray(frac_targets)).min(axis=1)


def _greedy_dispersed(lattice, frac, n, rng) -> list[int]:
    """Greedy max-min selection of n indices (maximises the minimum pairwise distance)."""
    chosen = [int(rng.integers(len(frac)))]
    dmin = lattice.get_all_distances(frac, frac[chosen]).min(axis=1)
    while len(chosen) < n:
        dmin[chosen] = -np.inf
        cand = np.flatnonzero(dmin >= dmin.max() - 1e-6)
        nxt = int(rng.choice(cand))
        chosen.append(nxt)
        dmin = np.minimum(dmin, lattice.get_all_distances(frac, frac[[nxt]]).ravel())
    return chosen


def _greedy_clustered(lattice, frac, n, rng, anchors=None) -> list[int]:
    """Pick n indices close to anchors (or, without anchors, close to each other)."""
    if anchors is not None and len(anchors):
        d = lattice.get_all_distances(frac, anchors).min(axis=1) + rng.uniform(0, 1e-3, len(frac))
        return [int(i) for i in np.argsort(d)[:n]]
    chosen = [int(rng.integers(len(frac)))]
    dmin = lattice.get_all_distances(frac, frac[chosen]).min(axis=1)
    while len(chosen) < n:
        dmin[chosen] = np.inf
        nxt = int(np.argmin(dmin + rng.uniform(0, 1e-3, len(frac))))
        chosen.append(nxt)
        dmin = np.minimum(dmin, lattice.get_all_distances(frac, frac[[nxt]]).ravel())
    return chosen


def interstitial_candidates(lattice, frac_anions, frac_cations, host: Host, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Tetrahedral interstitial candidates from the Delaunay tessellation of the anion sublattice (spec 5.5).

    Returns fractional coordinates (supercell) and Wyckoff-style labels under the host framework
    symmetry ('16e', '4x', '24x', '48h', '96i', ...).
    """
    from scipy.spatial import Delaunay
    frac_anions = np.asarray(frac_anions) % 1.0
    images = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    frac_img = (frac_anions[None, :, :] + images[:, None, :]).reshape(-1, 3)
    cart_img = lattice.get_cartesian_coords(frac_img)
    tri = Delaunay(cart_img)
    centers = cart_img[tri.simplices].mean(axis=1)
    fc = lattice.get_fractional_coords(centers)
    inside = np.all((fc >= -1e-9) & (fc < 1 - 1e-9), axis=1)
    fc = fc[inside] % 1.0
    # de-duplicate (the same tetrahedron appears from several image sets)
    keys = {}
    for p in fc:
        keys.setdefault(tuple(np.round(p, 4)), p)
    fc = np.array(list(keys.values()))
    # geometric filters
    d_cat = _min_dist_to(lattice, fc, frac_cations)
    d_an = _min_dist_to(lattice, fc, frac_anions)
    keep = (d_cat >= cfg["interstitial_min_cation_dist"]) & (d_an >= cfg["interstitial_min_anion_dist"])
    fc = fc[keep]
    # labels by orbit multiplicity under the framework space group (unit-cell fractional coordinates)
    ncell = np.array(cfg["supercell"])
    labels = []
    for p in fc:
        pu = (p * ncell) % 1.0
        imgs = []
        for op in host.sym_ops_frac:
            q = op.operate(pu) % 1.0
            if not any(np.allclose(q, r, atol=2e-3) or np.allclose((q - r + 0.5) % 1.0 - 0.5, 0, atol=2e-3) for r in imgs):
                imgs.append(q)
        m = len(imgs)
        labels.append({16: "16e", 48: "48h", 96: "96i"}.get(m, f"{m}x"))
    return fc, labels


def build_config(host: Host, comp: dict, mode: str, seed: int, cfg: dict):
    """Generate one doped supercell configuration (spec 5.2-5.5) and check the invariants (5.6).

    Returns (ase.Atoms with arrays['site_label'], meta dict).
    """
    from ase import Atoms
    rng = np.random.default_rng(seed)
    S = host.supercell
    lat = S.lattice
    labels = list(host.labels)
    species = [str(s) for s in S.species]
    li_site, p_site = comp_key_counts(comp)
    K = sum(li_site.values())
    n_ge = sum(p_site.values())
    assert n_ge <= cfg["max_ge_per_supercell"], f"P-site dopants {n_ge} > cap {cfg['max_ge_per_supercell']} (INV-8)"
    assert set(p_site) <= {"Ge"}, f"only Ge is defined as a P-site dopant, got {list(p_site)}"

    # 5.2 Li occupancy: one Li per pair (or the 24g site itself)
    occupied = [int(pair[rng.integers(len(pair))]) for pair in host.li_pairs]
    assert len(occupied) == N_LI_HOST

    # 5.3 S/Cl disorder: swap exactly n_swap Cl(4a) <-> S(4c)
    idx_4a = [i for i, l in enumerate(labels) if l == "Cl_4a"]
    idx_4c = [i for i, l in enumerate(labels) if l == "S_4c"]
    n_swap = int(round(cfg["scl_disorder_4c_cl_fraction"] * len(idx_4c)))
    sw_a = rng.choice(idx_4a, n_swap, replace=False)
    sw_c = rng.choice(idx_4c, n_swap, replace=False)
    for i in sw_a:
        species[i] = "S"
    for i in sw_c:
        species[i] = "Cl"

    # 5.4 P-site dopants (random placement of Ge on P sites)
    idx_p = [i for i, l in enumerate(labels) if l == "P_4b"]
    ge_sites = [int(i) for i in rng.choice(idx_p, n_ge, replace=False)] if n_ge else []
    for el, g in p_site.items():
        for i in ge_sites[:g]:
            species[i] = el
    frac_ge = S.frac_coords[ge_sites] if ge_sites else np.zeros((0, 3))

    # 5.4 Li-site dopants
    frac_occ = S.frac_coords[occupied]
    if K:
        if mode == "random":
            sel = [int(i) for i in rng.choice(len(occupied), K, replace=False)]
        elif mode == "clustered":
            sel = _greedy_clustered(lat, frac_occ, K, rng, anchors=frac_ge if len(frac_ge) else None)
        elif mode == "dispersed":
            sel = _greedy_dispersed(lat, frac_occ, K, rng)
        else:
            raise ValueError(mode)
        metal_list = [el for el, k in li_site.items() for _ in range(k)]
        rng.shuffle(metal_list)
        dopant_sites = {occupied[j]: metal_list[n] for n, j in enumerate(sel)}
    else:
        dopant_sites = {}
    for i, el in dopant_sites.items():
        species[i] = el
    li_regular = [i for i in occupied if i not in dopant_sites]          # Li on regular sites, before vacancies
    frac_dop = S.frac_coords[list(dopant_sites)] if dopant_sites else np.zeros((0, 3))
    frac_all_dop = np.vstack([frac_dop, frac_ge]) if (len(frac_dop) or len(frac_ge)) else np.zeros((0, 3))

    # 5.4 / 5.5 Li adjustment
    delta = li_delta(comp)
    vacancies, interstitials, inter_labels = [], [], []
    if delta < 0:
        n_vac = -delta
        frac_reg = S.frac_coords[li_regular]
        d_dop = _min_dist_to(lat, frac_reg, frac_all_dop)
        if mode == "random":
            order = rng.permutation(len(li_regular))
        elif mode == "clustered":
            near = d_dop <= cfg["vacancy_near_radius"]
            order = np.concatenate([rng.permutation(np.flatnonzero(near)), rng.permutation(np.flatnonzero(~near))])
        else:
            order = np.argsort(-(d_dop + rng.uniform(0, 1e-3, len(d_dop))))
        vacancies = [li_regular[j] for j in order[:n_vac]]
    li_regular_final = [i for i in li_regular if i not in set(vacancies)]

    # assemble the atom list: framework + regular Li + dopants, then interstitials
    keep = [i for i, l in enumerate(labels) if not l.startswith("Li")] + sorted(li_regular_final) + sorted(dopant_sites)
    sp_out = [species[i] for i in keep]
    fr_out = [S.frac_coords[i] for i in keep]
    lab_out = [labels[i] for i in keep]
    if delta > 0:
        anion_fr = np.array([S.frac_coords[i] for i in keep if species[i] in ("S", "Cl")])
        cation_fr = np.array([S.frac_coords[i] for i in keep if species[i] not in ("S", "Cl")])
        cand, cand_lab = interstitial_candidates(lat, anion_fr, cation_fr, host, cfg)
        if len(cand) < delta:
            raise AssertionError(f"{comp['id']}: only {len(cand)} interstitial candidates for {delta} insertions")
        pref = [j for j in rng.permutation(len(cand)) if cand_lab[j] == "16e"]
        rest = [j for j in rng.permutation(len(cand)) if cand_lab[j] != "16e"]
        chosen_fr = []
        for j in pref + rest:
            if len(chosen_fr) == delta:
                break
            if chosen_fr and _min_dist_to(lat, [cand[j]], chosen_fr)[0] < cfg["interstitial_min_cation_dist"]:
                continue
            chosen_fr.append(cand[j]); inter_labels.append(cand_lab[j])
        if len(chosen_fr) < delta:
            raise AssertionError(f"{comp['id']}: could not place {delta} mutually separated interstitial Li "
                                 f"(placed {len(chosen_fr)} of {len(cand)} candidates)")
        for p, l in zip(chosen_fr, inter_labels):
            sp_out.append("Li"); fr_out.append(np.asarray(p)); lab_out.append(f"Li_int_{l}")
        interstitials = list(range(len(keep), len(keep) + delta))

    fr_out = np.array(fr_out) % 1.0
    atoms = Atoms(symbols=sp_out, scaled_positions=fr_out, cell=lat.matrix, pbc=True)
    atoms.set_array("site_label", np.array(lab_out, dtype="U16"))

    # ---- metadata
    d_dop_ge = float(lat.get_all_distances(frac_dop, frac_ge).min()) if (len(frac_dop) and len(frac_ge)) else float("nan")
    frac_vac = S.frac_coords[vacancies] if vacancies else np.zeros((0, 3))
    d_dop_vac = float(lat.get_all_distances(frac_all_dop, frac_vac).min()) if (len(frac_all_dop) and len(frac_vac)) else float("nan")
    n_vac_near = int((_min_dist_to(lat, frac_vac, frac_all_dop) <= cfg["vacancy_near_radius"]).sum()) if len(frac_vac) else 0
    meta = dict(
        composition_id=comp["id"], config_mode=mode, seed=int(seed), n_li=n_li_of(comp), delta=int(delta),
        n_vacancy=len(vacancies), n_vacancy_near=n_vac_near, n_vacancy_far=len(vacancies) - n_vac_near,
        n_interstitial=len(interstitials), n_interstitial_16e=sum(1 for l in inter_labels if l == "16e"),
        n_interstitial_other=sum(1 for l in inter_labels if l != "16e"),
        interstitial_labels=",".join(inter_labels), n_scl_swaps=n_swap,
        d_dopant_ge_min=d_dop_ge, d_dopant_vacancy_min=d_dop_vac, li_site_mode=host.li_site_mode,
        n_atoms=len(atoms),
    )
    check_invariants(atoms, comp, meta, cfg)
    return atoms, meta


def check_invariants(atoms, comp: dict, meta: dict, cfg: dict) -> None:
    """INV-1..INV-9 (spec 5.6).  Any failure raises AssertionError -> the whole pipeline stops."""
    syms = atoms.get_chemical_symbols()
    labels = atoms.get_array("site_label")
    li_site, p_site = comp_key_counts(comp)
    cnt = pd.Series(syms).value_counts().to_dict()
    n_swap = int(round(cfg["scl_disorder_4c_cl_fraction"] * N_CL_HOST))
    # INV-1
    q = sum(VALENCE[s] for s in syms)
    assert q == 0, f"INV-1 charge sum = {q} != 0 ({comp['id']})"
    # INV-2
    assert cnt.get("Li", 0) == n_li_of(comp), f"INV-2 Li count {cnt.get('Li', 0)} != {n_li_of(comp)} ({comp['id']})"
    # INV-3
    assert cnt.get("P", 0) + sum(cnt.get(el, 0) for el in p_site) == N_P_HOST, f"INV-3 P-site count ({comp['id']})"
    # INV-4
    assert cnt.get("S", 0) == N_S_HOST and cnt.get("Cl", 0) == N_CL_HOST, f"INV-4 S/Cl counts {cnt} ({comp['id']})"
    # INV-5
    for el, k in li_site.items():
        assert cnt.get(el, 0) == k, f"INV-5 {el}: {cnt.get(el, 0)} != {k} ({comp['id']})"
    for el, g in p_site.items():
        assert cnt.get(el, 0) == g, f"INV-5 {el}: {cnt.get(el, 0)} != {g} ({comp['id']})"
    extra = set(cnt) - {"Li", "P", "S", "Cl"} - set(li_site) - set(p_site)
    assert not extra, f"INV-5 unexpected elements {extra} ({comp['id']})"
    # INV-6
    n_cl_4c = sum(1 for s, l in zip(syms, labels) if l == "S_4c" and s == "Cl")
    n_s_4a = sum(1 for s, l in zip(syms, labels) if l == "Cl_4a" and s == "S")
    assert n_cl_4c == n_swap and n_s_4a == n_swap, f"INV-6 disorder swaps {n_cl_4c}/{n_s_4a} != {n_swap} ({comp['id']})"
    # INV-7
    from ase.neighborlist import neighbor_list
    d = neighbor_list("d", atoms, cfg["min_interatomic_dist"] + 0.5)
    dmin = float(d.min()) if len(d) else float("inf")
    assert dmin >= cfg["min_interatomic_dist"], f"INV-7 min distance {dmin:.3f} A < {cfg['min_interatomic_dist']} ({comp['id']})"
    meta["min_dist"] = dmin
    # INV-8
    assert sum(p_site.values()) <= cfg["max_ge_per_supercell"], f"INV-8 ({comp['id']})"
    # INV-9
    expected = N_ATOMS_HOST - meta["n_vacancy"] + meta["n_interstitial"]
    assert len(atoms) == expected, f"INV-9 atoms {len(atoms)} != {expected} ({comp['id']})"


def config_modes(stage: str) -> list[str]:
    return list(MC_MODES_PILOT if stage == "I" else MC_MODES_STAGE2)


def generate_configs(host: Host, comp: dict, stage: str, cfg: dict, seed_salt: str = "") -> list[tuple]:
    """All configurations of one composition: list of (config_id, atoms, meta)."""
    out = []
    for k, mode in enumerate(config_modes(stage)):
        seed = derived_seed(cfg["seed"], "config", comp["id"], k, seed_salt)
        atoms, meta = build_config(host, comp, mode, seed, cfg)
        cid = f"c{k}"
        meta["config_id"] = cid
        out.append((cid, atoms, meta))
    return out


def write_atoms(atoms, path: str, info: dict | None = None) -> None:
    from ase.io import write
    a = atoms.copy()
    if info:
        a.info.update({k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write(path, a, format="extxyz")


def read_atoms(path: str):
    from ase.io import read
    return read(path, format="extxyz")


# ============================================================================ models
@dataclass
class ModelHandle:
    name: str                 # "orb_v3" | "sevenn" | "mock"
    tag: str                  # checkpoint description recorded in the manifest
    backend: str              # "torchsim" | "ase"
    ts_model: object = None
    ase_calc: object = None
    device: str = "cpu"
    dtype: object = None
    notes: dict = field(default_factory=dict)


def load_model(name: str, cfg: dict, device: str | None = None) -> ModelHandle:
    """Load one MLIP.  Never guesses: the actual checkpoint / backend used is returned in `tag`/`notes`."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mock":
        from torch_sim.models.lennard_jones import LennardJonesModel
        m = LennardJonesModel(sigma=2.6, epsilon=0.05, cutoff=6.0, device=device, dtype=torch.float32,
                              compute_forces=True, compute_stress=True)
        return ModelHandle(name, "mock-lennard-jones(sigma=2.6,eps=0.05)", "torchsim", ts_model=m,
                           device=device, dtype=torch.float32, notes={"warning": "wiring test only"})
    if name == "orb_v3":
        from orb_models.forcefield import pretrained
        import orb_models
        from torch_sim.models.orb import OrbModel
        fn = getattr(pretrained, cfg["orb_checkpoint"])
        loaded = fn(device=device, precision=cfg.get("orb_precision", "float32-high"))
        raw, adapter = loaded if isinstance(loaded, tuple) else (loaded, None)
        try:
            m = OrbModel(model=raw, atoms_adapter=adapter, device=device, dtype=torch.float32)
            how = "OrbModel(model, atoms_adapter)"
        except TypeError:
            m = OrbModel(model=raw, device=device, dtype=torch.float32)
            how = "OrbModel(model)"
        return ModelHandle(name, cfg["orb_checkpoint"], "torchsim", ts_model=m, device=device, dtype=torch.float32,
                           notes={"orb_models_version": getattr(orb_models, "__version__", "?"), "wrapper": how,
                                  "precision": cfg.get("orb_precision", "float32-high")})
    if name == "sevenn":
        import sevenn
        ver = getattr(sevenn, "__version__", "?")
        errors = {}
        for ckpt in (cfg["sevenn_checkpoint"], cfg.get("sevenn_fallback_checkpoint")):
            if not ckpt:
                continue
            try:
                from sevenn.torchsim import SevenNetModel
                m = SevenNetModel(model=ckpt, modal=cfg["sevenn_modal"], device=device)
                return ModelHandle(name, f"{ckpt}[modal={cfg['sevenn_modal']}]", "torchsim", ts_model=m, device=device,
                                   dtype=torch.float32, notes={"sevenn_version": ver, "wrapper": "sevenn.torchsim.SevenNetModel",
                                                                "tried": errors})
            except Exception as e:                       # noqa: BLE001 - record and try the ASE path
                errors[f"torchsim:{ckpt}"] = f"{type(e).__name__}: {str(e)[:200]}"
            try:
                from sevenn.calculator import SevenNetCalculator
                calc = SevenNetCalculator(model=ckpt, modal=cfg["sevenn_modal"], device=device)
                return ModelHandle(name, f"{ckpt}[modal={cfg['sevenn_modal']}]", "ase", ase_calc=calc, device=device,
                                   dtype=torch.float32, notes={"sevenn_version": ver, "wrapper": "sevenn.calculator.SevenNetCalculator (ASE fallback)",
                                                                "tried": errors})
            except Exception as e:                       # noqa: BLE001
                errors[f"ase:{ckpt}"] = f"{type(e).__name__}: {str(e)[:200]}"
        raise RuntimeError(f"SevenNet could not be loaded: {json.dumps(errors, indent=1)}")
    raise ValueError(f"unknown model {name}")


def energy_forces(handle: ModelHandle, atoms) -> tuple[float, np.ndarray]:
    """Single-point energy (eV) and forces (eV/A) with either backend."""
    if handle.backend == "torchsim":
        import torch_sim as ts
        state = ts.io.atoms_to_state([atoms], device=handle.device, dtype=handle.dtype)
        out = handle.ts_model(state)
        return float(out["energy"].detach().cpu().numpy().ravel()[0]), out["forces"].detach().cpu().numpy()
    a = atoms.copy(); a.calc = handle.ase_calc
    return float(a.get_potential_energy()), a.get_forces()


# ============================================================================ relaxation
def _reattach_arrays(src, dst):
    for k in src.arrays:
        if k not in ("numbers", "positions") and k not in dst.arrays:
            dst.set_array(k, src.arrays[k].copy())
    dst.info.update(src.info)
    return dst


def relax_batch(handle: ModelHandle, atoms_list: list, cfg: dict, relax_cell: bool | None = None) -> list[dict]:
    """Relax a list of ASE Atoms.  Returns dicts (atoms, energy, converged, fmax, n_atoms).

    torch-sim: FIRE + Frechet cell filter, autobatched; a failed batch falls back to per-structure runs.
    ASE fallback (SevenNet without torch-sim support): FIRE + FrechetCellFilter.
    """
    relax_cell = cfg["relax_cell"] if relax_cell is None else relax_cell
    fmax, max_steps = float(cfg["relax_fmax"]), int(cfg["relax_max_steps"])
    results = []
    if handle.backend == "torchsim":
        import gc
        import torch
        import torch_sim as ts
        from torch_sim.optimizers.cell_filters import CellFilter

        def _opt(batch, autob):
            state = ts.io.atoms_to_state(batch, device=handle.device, dtype=handle.dtype)
            init_kwargs = {"cell_filter": CellFilter.frechet} if relax_cell else {}
            return ts.optimize(system=state, model=handle.ts_model, optimizer=ts.Optimizer.fire,
                               convergence_fn=ts.generate_force_convergence_fn(force_tol=fmax),
                               max_steps=max_steps, init_kwargs=init_kwargs, autobatcher=autob)

        def _collect(final, batch):
            out = ts.io.state_to_atoms(final)
            assert [len(a) for a in out] == [len(a) for a in batch], "torch-sim returned systems in a different order"
            E = final.energy.detach().cpu().numpy().ravel()
            F = final.forces.detach().cpu().numpy()
            sysidx = final.system_idx.detach().cpu().numpy()
            rows = []
            for j, (src, rel) in enumerate(zip(batch, out)):
                fm = float(np.linalg.norm(F[sysidx == j], axis=1).max())
                rows.append(dict(atoms=_reattach_arrays(src, rel), energy=float(E[j]), converged=bool(fm <= fmax * 1.0001),
                                 fmax=fm, n_atoms=len(rel)))
            return rows

        gc.collect()
        if handle.device.startswith("cuda"):
            torch.cuda.empty_cache()
        err = None
        try:
            use_autob = len(atoms_list) > 1 and str(handle.device).startswith("cuda")   # memory estimation needs a GPU
            results = _collect(_opt(atoms_list, use_autob), atoms_list)
        except AssertionError:
            raise
        except Exception as e:                            # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
        if err is not None:
            gc.collect()
            if handle.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"[relax] batch failed ({err}) -> per-structure fallback", flush=True)
            results = []
            for a in atoms_list:
                try:
                    results.extend(_collect(_opt([a], False), [a]))
                except Exception as e2:                   # noqa: BLE001
                    results.append(dict(atoms=None, energy=float("nan"), converged=False, fmax=float("nan"),
                                        n_atoms=len(a), error=f"{type(e2).__name__}: {str(e2)[:200]}"))
                    gc.collect()
                    if handle.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        return results
    # ---- ASE path
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    for a0 in atoms_list:
        a = a0.copy(); a.calc = handle.ase_calc
        try:
            target = FrechetCellFilter(a) if relax_cell else a
            opt = FIRE(target, logfile=None)
            conv = bool(opt.run(fmax=fmax, steps=max_steps))
            fm = float(np.linalg.norm(a.get_forces(), axis=1).max())
            results.append(dict(atoms=_reattach_arrays(a0, a.copy()), energy=float(a.get_potential_energy()),
                                converged=conv, fmax=fm, n_atoms=len(a)))
        except Exception as e:                            # noqa: BLE001
            results.append(dict(atoms=None, energy=float("nan"), converged=False, fmax=float("nan"),
                                n_atoms=len(a0), error=f"{type(e).__name__}: {str(e)[:200]}"))
    return results


def relax_job_paths(cfg: dict, model: str, key: str) -> tuple[str, str]:
    return (wpath(cfg, "outputs", "structures", "relaxed", model, f"{key}.extxyz"),
            wpath(cfg, "outputs", "structures", "relaxed", model, f"{key}.json"))


def run_relax_jobs(handle: ModelHandle, jobs: list[dict], cfg: dict) -> None:
    """jobs: [{key, input (extxyz path), relax_cell?}]; writes relaxed extxyz + json per job (resume-safe)."""
    todo = [j for j in jobs if not os.path.exists(relax_job_paths(cfg, handle.name, j["key"])[1])]
    print(f"[relax:{handle.name}] {len(todo)} of {len(jobs)} jobs to do", flush=True)
    chunk = int(cfg["relax_chunk"])
    todo.sort(key=lambda j: j.get("n_atoms", 0))
    t0 = time.time()
    for i0 in range(0, len(todo), chunk):
        batch = todo[i0:i0 + chunk]
        atoms_in = [read_atoms(j["input"]) for j in batch]
        same_cell_flag = {bool(j.get("relax_cell", cfg["relax_cell"])) for j in batch}
        assert len(same_cell_flag) == 1, "mixed relax_cell flags in one chunk"
        res = relax_batch(handle, atoms_in, cfg, relax_cell=same_cell_flag.pop())
        for j, r in zip(batch, res):
            p_xyz, p_json = relax_job_paths(cfg, handle.name, j["key"])
            row = dict(key=j["key"], model=handle.name, model_tag=handle.tag, backend=handle.backend,
                       energy=r["energy"], n_atoms=r["n_atoms"],
                       e_per_atom=r["energy"] / r["n_atoms"] if np.isfinite(r["energy"]) else float("nan"),
                       converged=r["converged"], fmax=r["fmax"], error=r.get("error", ""))
            if r["atoms"] is not None:
                a = r["atoms"]
                row.update(a=float(a.cell.lengths()[0]), b=float(a.cell.lengths()[1]), c=float(a.cell.lengths()[2]),
                           volume=float(a.get_volume()))
                write_atoms(a, p_xyz, info={"energy": r["energy"], "model": handle.name})
            atomic_write_json(row, p_json)
        print(f"[relax:{handle.name}] {min(i0 + chunk, len(todo))}/{len(todo)} | {(time.time() - t0) / 60:.1f} min", flush=True)


def collect_relax_results(cfg: dict, model: str, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k in keys:
        _, pj = relax_job_paths(cfg, model, k)
        if os.path.exists(pj):
            rows.append(json.load(open(pj)))
    return pd.DataFrame(rows)


# ============================================================================ hull / stability
def mp_light_pool_cache(cfg: dict) -> str:
    return wpath(cfg, "outputs", "hull_cache", "mp_near_hull_light.json")


def fetch_mp_near_hull_pool(cfg: dict, pool_elements: list[str]) -> dict:
    """One light MP query: every entry with E_hull <= hull_mp_ehull_max whose elements lie inside the pool.

    Cached (keyed by the pool and the threshold).  Structures are fetched separately, on demand.
    """
    cache = mp_light_pool_cache(cfg)
    pool = sorted(set(pool_elements))
    key = "-".join(pool) + f"@{cfg['hull_mp_ehull_max']}"
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("key") == key:
            return d
    api_key = os.environ.get("MP_API_KEY", "")
    assert api_key, "export MP_API_KEY before building the hull"
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(energy_above_hull=(0.0, float(cfg["hull_mp_ehull_max"])),
                                            num_elements=(1, 6),
                                            fields=["material_id", "elements", "formula_pretty", "energy_above_hull"])
    pset = set(pool)
    entries = {}
    for d in docs:
        els = sorted(str(e) for e in d.elements)
        if set(els) <= pset:
            entries[str(d.material_id)] = dict(elements=els, formula=str(d.formula_pretty),
                                               mp_e_above_hull=float(d.energy_above_hull))
    out = dict(key=key, pool=pool, n_docs=len(docs), entries=entries, fetched=time.strftime("%Y-%m-%d %H:%M"))
    atomic_write_json(out, cache)
    return out


def fetch_mp_structures(cfg: dict, mp_ids: list[str]) -> dict:
    """material_id -> pymatgen Structure dict, cached on disk."""
    cache = wpath(cfg, "outputs", "hull_cache", "mp_structures.json")
    have = json.load(open(cache)) if os.path.exists(cache) else {}
    need = [m for m in mp_ids if m not in have]
    if need:
        api_key = os.environ.get("MP_API_KEY", "")
        assert api_key, "export MP_API_KEY before building the hull"
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            for i0 in range(0, len(need), 200):
                for d in mpr.materials.summary.search(material_ids=need[i0:i0 + 200], fields=["material_id", "structure"]):
                    have[str(d.material_id)] = d.structure.as_dict()
        atomic_write_json(have, cache)
    return {m: have[m] for m in mp_ids if m in have}


def reference_ids_for(pool: dict, elements: list[str]) -> list[str]:
    eset = set(elements)
    return sorted(m for m, e in pool["entries"].items() if set(e["elements"]) <= eset)


def assert_required_references(pool: dict, elements: list[str]) -> list[str]:
    """Spec 7.4-1: the listed reference phases must be present for the chemsys."""
    from pymatgen.core import Composition
    ids = reference_ids_for(pool, elements)
    formulas = {Composition(pool["entries"][m]["formula"]).reduced_formula for m in ids}
    required = list(REQUIRED_REFERENCE_FORMULAS) if "Ge" in elements else ["Li2S", "LiCl", "Li3PS4", "P2S5"]
    for el in elements:
        required += DOPANT_REFERENCE_FORMULAS.get(el, [])
    missing = [f for f in required if Composition(f).reduced_formula not in formulas]
    assert not missing, f"required hull references missing from MP near-hull pool for {elements}: {missing}"
    return ids


def prepare_reference_inputs(cfg: dict, mp_ids: list[str]) -> list[dict]:
    """Write reference-phase input structures (expanded to >= hull_min_ref_atoms) and return relax jobs."""
    from pymatgen.core import Structure
    structs = fetch_mp_structures(cfg, mp_ids)
    jobs = []
    for mid in mp_ids:
        st = Structure.from_dict(structs[mid])
        if len(st) < cfg["hull_min_ref_atoms"]:
            n = math.ceil((cfg["hull_min_ref_atoms"] / len(st)) ** (1 / 3))
            st = st * (n, n, n)
        p = wpath(cfg, "outputs", "structures", "references_input", f"{mid}.extxyz")
        if not os.path.exists(p):
            from pymatgen.io.ase import AseAtomsAdaptor
            a = AseAtomsAdaptor.get_atoms(st)
            a.set_array("site_label", np.array(["ref"] * len(a), dtype="U16"))
            write_atoms(a, p, info={"mp_id": mid})
        jobs.append(dict(key=f"REF__{mid}", input=p, n_atoms=len(st), relax_cell=True))
    return jobs


def build_phase_diagram(cfg: dict, model: str, mp_ids: list[str]):
    """PhaseDiagram from the MLIP-relaxed reference phases of ONE model (spec 7.4-2/3)."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    entries = []
    for mid in mp_ids:
        _, pj = relax_job_paths(cfg, model, f"REF__{mid}")
        if not os.path.exists(pj):
            continue
        r = json.load(open(pj))
        if not np.isfinite(r["energy"]):
            continue
        a = read_atoms(relax_job_paths(cfg, model, f"REF__{mid}")[0])
        entries.append(PDEntry(Composition(a.get_chemical_formula()), r["energy"], name=mid,
                               attribute={"model": model, "converged": r["converged"]}))
    assert entries, "no relaxed reference phases"
    return PhaseDiagram(entries)


def e_above_hull(pd_obj, atoms, energy: float) -> float:
    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.core import Composition
    entry = PDEntry(Composition(atoms.get_chemical_formula()), energy)
    _, eh = pd_obj.get_decomp_and_e_above_hull(entry, allow_negative=True)
    return float(eh)


def configurational_entropy_eV_per_K(comp: dict) -> float:
    """dS_conf = -k_B sum N_s ln x_s over the 192 regular Li sites (Li, each metal, vacancies); spec 8.4."""
    li_site, _ = comp_key_counts(comp)
    K = sum(li_site.values())
    delta = li_delta(comp)
    n_vac = -delta if delta < 0 else 0
    n_li_reg = N_LI_HOST - K - n_vac
    counts = [n_li_reg, n_vac] + list(li_site.values())
    s = 0.0
    for n in counts:
        if n > 0:
            x = n / N_LI_HOST
            s -= n * math.log(x)
    return K_B_EV * s


# ============================================================================ molecular dynamics
def md_dir(cfg: dict, model: str, key: str) -> str:
    p = Path(cfg["workdir"], "outputs", "trajectories", model, key)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def unwrap_positions(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Unwrap a [n_frames, n_atoms, 3] cartesian trajectory (fixed cell) by minimum image between frames."""
    inv = np.linalg.inv(cell)
    frac = pos @ inv
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    out = np.concatenate([frac[:1], frac[:1] + np.cumsum(d, axis=0)], axis=0)
    return out @ cell


def _h5_to_arrays(h5_path: str):
    """Read a torch-sim trajectory: positions [n,N,3], cells [n or 1,3,3], numbers [N], steps [n]."""
    from torch_sim.trajectory import TorchSimTrajectory
    with TorchSimTrajectory(h5_path, mode="r") as tr:
        pos = np.asarray(tr.get_array("positions"))
        steps = np.asarray(tr.get_steps("positions"))
        n = len(pos)
        cells = np.array([tr.get_atoms(i).cell[:] for i in (range(n) if n <= 400 else np.linspace(0, n - 1, 400).astype(int))])
        numbers = tr.get_atoms(0).get_atomic_numbers()
    return pos, cells, numbers, steps


def run_md_batch(handle: ModelHandle, jobs: list[dict], cfg: dict) -> None:
    """Run NPT (20 ps) -> mean cell -> NVT production for a batch of jobs on one device (spec 7.5).

    job: {key, model, composition_id, config_id, T, input (relaxed extxyz), npt_ps, nvt_ps, seed}
    Output per job (in outputs/trajectories/<model>/<key>/):
        npt.h5 (torch-sim), nvt.npz (unwrapped positions float32, cell, numbers, times_fs), result.json
    """
    if handle.backend == "ase":
        for j in jobs:
            run_md_ase(handle, j, cfg)
        return
    import torch
    import torch_sim as ts
    from torch_sim.trajectory import TrajectoryReporter

    dt_fs = float(cfg["md_timestep_fs"]); dt_ps = dt_fs / 1000.0
    save_every = int(round(cfg["md_save_every_fs"] / dt_fs))
    atoms_in = [read_atoms(j["input"]) for j in jobs]
    temps = [float(j["T"]) for j in jobs] if len(jobs) > 1 else float(jobs[0]["T"])   # per-system temperatures (K); plain floats avoid device mismatches
    seed = derived_seed(cfg["seed"], "md", *[j["key"] for j in jobs])
    torch.manual_seed(seed)
    dirs = [md_dir(cfg, handle.name, j["key"]) for j in jobs]
    n_npt = int(round(jobs[0]["npt_ps"] * 1000 / dt_fs)); n_nvt = int(round(jobs[0]["nvt_ps"] * 1000 / dt_fs))
    assert all(int(round(j["npt_ps"] * 1000 / dt_fs)) == n_npt and int(round(j["nvt_ps"] * 1000 / dt_fs)) == n_nvt for j in jobs), \
        "all jobs of one MD batch must share npt_ps / nvt_ps"
    thermostat = cfg["md_thermostat"]
    t0 = time.time()

    # ---------------- NPT
    npt_files = [os.path.join(d, "npt.h5") for d in dirs]
    state = ts.io.atoms_to_state(atoms_in, device=handle.device, dtype=handle.dtype)
    state.rng = seed
    rep = TrajectoryReporter(npt_files, state_frequency=save_every, state_kwargs={"variable_cell": True})
    npt_integrator = ts.Integrator.npt_nose_hoover_isotropic if thermostat == "nose_hoover" else ts.Integrator.npt_langevin_isotropic
    try:
        state = ts.integrate(system=state, model=handle.ts_model, integrator=npt_integrator,
                             n_steps=n_npt, temperature=temps, timestep=dt_ps, trajectory_reporter=rep,
                             external_pressure=0.0)
    finally:
        rep.close()                                       # an exception (e.g. OOM) must not leave the h5 files open for the retry
    t_npt = time.time() - t0
    final_npt = ts.io.state_to_atoms(state)

    # ---------------- mean cell over the second half of NPT, rescale, NVT
    nvt_inputs = []
    mean_cells = []
    for a_in, a_npt, f in zip(atoms_in, final_npt, npt_files):
        _, cells, _, steps = _h5_to_arrays(f)
        half = len(cells) // 2
        mean_cell = cells[half:].mean(axis=0)
        mean_cells.append(mean_cell)
        a = a_npt.copy()
        a.set_cell(mean_cell, scale_atoms=True)
        a.wrap()
        nvt_inputs.append(_reattach_arrays(a_in, a))
    t1 = time.time()
    nvt_h5 = [os.path.join(d, "nvt.h5") for d in dirs]
    state = ts.io.atoms_to_state(nvt_inputs, device=handle.device, dtype=handle.dtype)
    state.rng = seed + 1
    rep = TrajectoryReporter(nvt_h5, state_frequency=save_every, state_kwargs={"variable_cell": False})
    integ_kwargs = {}
    if thermostat == "nose_hoover":
        integrator = ts.Integrator.nvt_nose_hoover
    else:
        integrator = ts.Integrator.nvt_langevin
        integ_kwargs["gamma"] = float(cfg["md_langevin_friction_per_fs"]) * 1000.0     # 1/ps
    try:
        state = ts.integrate(system=state, model=handle.ts_model, integrator=integrator, n_steps=n_nvt,
                             temperature=temps, timestep=dt_ps, trajectory_reporter=rep, **integ_kwargs)
    finally:
        rep.close()
    t_nvt = time.time() - t1

    # ---------------- convert to npz (unwrapped), write result.json
    for j, d, f, mc, a_in in zip(jobs, dirs, nvt_h5, mean_cells, atoms_in):
        pos, cells, numbers, steps = _h5_to_arrays(f)
        cell = np.asarray(mc, dtype=float)
        pos_u = unwrap_positions(pos.astype(np.float64), cell)
        jump = np.abs(np.diff(pos_u, axis=0)).max() if len(pos_u) > 1 else 0.0
        times_fs = steps.astype(float) * dt_fs
        np.savez(os.path.join(d, "nvt.npz"), positions=pos_u.astype(np.float32), cell=cell, numbers=numbers,
                 times_fs=times_fs, site_label=a_in.get_array("site_label"))
        li = numbers == 3
        msd_last = float(((pos_u[-1, li] - pos_u[0, li]) ** 2).sum(axis=1).mean())
        res = dict(key=j["key"], model=handle.name, model_tag=handle.tag, backend=handle.backend,
                   composition_id=j["composition_id"], config_id=j["config_id"], T=float(j["T"]),
                   thermostat=thermostat, barostat="npt_" + thermostat + "_isotropic", timestep_fs=dt_fs,
                   save_every_fs=cfg["md_save_every_fs"], npt_ps=j["npt_ps"], nvt_ps=j["nvt_ps"],
                   n_frames=int(len(pos)), seed=int(seed), batch_size=len(jobs),
                   cell=cell.tolist(), volume_A3=float(abs(np.linalg.det(cell))), n_li=int(li.sum()),
                   msd_li_last_A2=msd_last, max_frame_jump_A=float(jump),
                   wall_npt_s=t_npt, wall_nvt_s=t_nvt,
                   ns_per_day_nvt_batch=(n_nvt * dt_fs * 1e-6) * len(jobs) / (t_nvt / 86400.0),
                   ns_per_day_nvt_per_system=(n_nvt * dt_fs * 1e-6) / (t_nvt / 86400.0),
                   haven_ratio_assumed=1.0)
        atomic_write_json(res, os.path.join(d, "result.json"))
        if not cfg.get("md_keep_h5", False):
            for p in (f, os.path.join(d, "npt.h5")):
                try:
                    os.remove(p)
                except OSError:
                    pass
    print(f"[md:{handle.name}] batch of {len(jobs)} done: NPT {t_npt / 60:.1f} min, NVT {t_nvt / 60:.1f} min "
          f"({(n_nvt * dt_fs * 1e-6) * len(jobs) / (t_nvt / 86400.0):.2f} ns/day aggregate)", flush=True)


def run_md_ase(handle: ModelHandle, j: dict, cfg: dict) -> None:
    """ASE fallback MD (Nose-Hoover chain NVT, isotropic MTK NPT) for a backend without torch-sim support."""
    from ase import units
    from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    dt_fs = float(cfg["md_timestep_fs"]); save_every = int(round(cfg["md_save_every_fs"] / dt_fs))
    d = md_dir(cfg, handle.name, j["key"])
    a_in = read_atoms(j["input"]); a = a_in.copy(); a.calc = handle.ase_calc
    seed = derived_seed(cfg["seed"], "md", j["key"])
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(a, temperature_K=float(j["T"]), rng=rng)
    n_npt = int(round(j["npt_ps"] * 1000 / dt_fs)); n_nvt = int(round(j["nvt_ps"] * 1000 / dt_fs))
    t0 = time.time()
    cells = []
    dyn = IsotropicMTKNPT(a, timestep=dt_fs * units.fs, temperature_K=float(j["T"]), pressure_au=0.0,
                          tdamp=100 * dt_fs * units.fs, pdamp=1000 * dt_fs * units.fs)
    dyn.attach(lambda: cells.append(a.cell[:].copy()), interval=save_every)
    dyn.run(n_npt)
    t_npt = time.time() - t0
    half = len(cells) // 2
    cell = np.mean(cells[half:], axis=0)
    a.set_cell(cell, scale_atoms=True); a.wrap()
    pos, steps = [], []
    dyn = NoseHooverChainNVT(a, timestep=dt_fs * units.fs, temperature_K=float(j["T"]), tdamp=100 * dt_fs * units.fs)
    dyn.attach(lambda: (pos.append(a.get_positions().copy()), steps.append(dyn.nsteps)), interval=save_every)
    t1 = time.time()
    dyn.run(n_nvt)
    t_nvt = time.time() - t1
    pos = np.array(pos); steps = np.array(steps)
    pos_u = unwrap_positions(pos, cell)
    numbers = a.get_atomic_numbers()
    np.savez(os.path.join(d, "nvt.npz"), positions=pos_u.astype(np.float32), cell=cell, numbers=numbers,
             times_fs=steps.astype(float) * dt_fs, site_label=a_in.get_array("site_label"))
    li = numbers == 3
    res = dict(key=j["key"], model=handle.name, model_tag=handle.tag, backend="ase",
               composition_id=j["composition_id"], config_id=j["config_id"], T=float(j["T"]),
               thermostat="nose_hoover_chain(ase)", barostat="IsotropicMTKNPT(ase)", timestep_fs=dt_fs,
               save_every_fs=cfg["md_save_every_fs"], npt_ps=j["npt_ps"], nvt_ps=j["nvt_ps"], n_frames=int(len(pos)),
               seed=int(seed), batch_size=1, cell=cell.tolist(), volume_A3=float(abs(np.linalg.det(cell))),
               n_li=int(li.sum()), msd_li_last_A2=float(((pos_u[-1, li] - pos_u[0, li]) ** 2).sum(axis=1).mean()),
               max_frame_jump_A=float(np.abs(np.diff(pos_u, axis=0)).max()) if len(pos_u) > 1 else 0.0,
               wall_npt_s=t_npt, wall_nvt_s=t_nvt,
               ns_per_day_nvt_batch=(n_nvt * dt_fs * 1e-6) / (t_nvt / 86400.0),
               ns_per_day_nvt_per_system=(n_nvt * dt_fs * 1e-6) / (t_nvt / 86400.0), haven_ratio_assumed=1.0)
    atomic_write_json(res, os.path.join(d, "result.json"))


def run_md_jobs(handle: ModelHandle, jobs: list[dict], cfg: dict) -> None:
    todo = [j for j in jobs if not os.path.exists(os.path.join(md_dir(cfg, handle.name, j["key"]), "result.json"))]
    print(f"[md:{handle.name}] {len(todo)} of {len(jobs)} jobs to do", flush=True)
    bs = max(1, int(cfg["md_batch_size"]))
    # group by (npt_ps, nvt_ps) so that a batch shares its step counts
    groups = {}
    for j in todo:
        groups.setdefault((j["npt_ps"], j["nvt_ps"]), []).append(j)
    for _, js in groups.items():
        for i0 in range(0, len(js), bs):
            _run_md_split(handle, js[i0:i0 + bs], cfg)


def _run_md_split(handle: ModelHandle, batch: list[dict], cfg: dict) -> None:
    """Run one MD batch; on failure (typically CUDA OOM) retry the two halves, down to single jobs."""
    failed = False
    try:
        run_md_batch(handle, batch, cfg)
    except Exception as e:                                # noqa: BLE001
        print(f"[md:{handle.name}] batch of {len(batch)} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if len(batch) == 1:
            atomic_write_json(dict(key=batch[0]["key"], error=f"{type(e).__name__}: {str(e)[:300]}"),
                              os.path.join(md_dir(cfg, handle.name, batch[0]["key"]), "result.json"))
        else:
            failed = True
    if failed:
        # retry outside the except block: the traceback would otherwise keep the failed batch's GPU tensors alive
        import gc
        gc.collect()
        _empty_cuda_cache()
        h = (len(batch) + 1) // 2
        print(f"[md:{handle.name}] -> retrying as batches of {h} and {len(batch) - h}", flush=True)
        _run_md_split(handle, batch[:h], cfg)
        _run_md_split(handle, batch[h:], cfg)


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                     # noqa: BLE001
        pass


# ============================================================================ kinisi analysis
def _kinisi_dt_fs(times_fs: np.ndarray, n_dt: int) -> np.ndarray:
    """Log-spaced subset of the available time intervals (multiples of the frame interval)."""
    step = times_fs[1] - times_fs[0]
    n_frames = len(times_fs)
    ks = np.unique(np.round(np.geomspace(1, n_frames - 1, n_dt)).astype(int))
    return ks * step


def analyze_trajectory(npz_path: str, T: float, cfg: dict, seed: int, out_json: str | None = None) -> dict:
    """kinisi tracer diffusion of Li + Nernst-Einstein sigma with posterior samples (spec 7.6)."""
    import scipp as sc
    from ase import Atoms
    from kinisi.diffusion_analyzer import DiffusionAnalyzer

    z = np.load(npz_path)
    pos, cell, numbers, times = z["positions"].astype(float), z["cell"], z["numbers"], z["times_fs"]
    frames = [Atoms(numbers=numbers, positions=p, cell=cell, pbc=True) for p in pos]
    dt_fs_frame = float(times[1] - times[0])
    dt = sc.array(dims=["time interval"], values=_kinisi_dt_fs(times, int(cfg["kinisi_n_dt"])), unit="fs")
    an = DiffusionAnalyzer.from_ase(frames, specie="Li", time_step=sc.scalar(dt_fs_frame, unit="fs"),
                                    step_skip=sc.scalar(1, unit="dimensionless"), dt=dt, progress=False)
    msd_t = an.dt.values.astype(float)            # fs
    msd = an.da.values.astype(float)              # A^2
    start_fs = float(cfg["kinisi_start_dt_ps"]) * 1000.0
    total_fs = float(times[-1] - times[0])
    if start_fs >= 0.5 * total_fs:                 # short trajectories (smoke test): fit the last half
        start_fs = 0.25 * total_fs
    rs = np.random.RandomState(seed % (2 ** 31))
    an.diffusion(sc.scalar(start_fs, unit="fs"), n_samples=int(cfg["kinisi_n_samples"]), n_walkers=int(cfg["kinisi_n_walkers"]),
                 n_burn=int(cfg["kinisi_n_burn"]), n_thin=int(cfg["kinisi_n_thin"]), progress=False, random_state=rs)
    D = np.asarray(an.D.values, dtype=float)      # cm^2/s
    fit = msd_t >= start_fs
    slope = float(np.polyfit(np.log(msd_t[fit]), np.log(np.maximum(msd[fit], 1e-12)), 1)[0]) if fit.sum() >= 3 else float("nan")
    n_li = int((numbers == 3).sum()); V_m3 = float(abs(np.linalg.det(cell))) * 1e-30
    sig = n_li * E_CHARGE ** 2 * (D * 1e-4) / (V_m3 * K_B_J * T) * 1e-2   # S/cm
    lo, hi = (1 - cfg["ci_level"]) / 2, 1 - (1 - cfg["ci_level"]) / 2
    sp = sig[sig > 0]
    res = dict(T=float(T), n_li=n_li, volume_A3=V_m3 * 1e30, n_frames=int(len(pos)), t_total_ps=total_fs / 1000.0,
               start_dt_ps=start_fs / 1000.0, D_star=float(np.median(D)), D_star_mean=float(D.mean()),
               D_star_ci_low=float(np.quantile(D, lo)), D_star_ci_high=float(np.quantile(D, hi)),
               sigma_Scm=float(np.median(sig)), sigma_ci_low=float(np.quantile(sig, lo)), sigma_ci_high=float(np.quantile(sig, hi)),
               ln_sigma_mean=float(np.log(sp).mean()) if len(sp) else float("nan"),
               ln_sigma_sd=float(np.log(sp).std()) if len(sp) > 1 else float("nan"),
               frac_negative_D=float((D <= 0).mean()),
               msd_last_A2=float(msd[-1]), flag_nondiffusive=bool(msd[-1] < cfg["md_nondiffusive_msd_A2"]),
               msd_loglog_slope=slope, haven_ratio_assumed=1.0, ci_level=cfg["ci_level"], seed=int(seed))
    if out_json:
        atomic_write_json(res, out_json)
        np.savez(out_json.replace(".json", "_posterior.npz"), D_samples=D.astype(np.float32),
                 msd_t_fs=msd_t.astype(np.float32), msd_A2=msd.astype(np.float32))
    return res


def run_analyze_jobs(jobs: list[dict], cfg: dict) -> None:
    """jobs: [{npz, T, out, seed}] (CPU); resume-safe."""
    for j in jobs:
        if os.path.exists(j["out"]):
            continue
        try:
            analyze_trajectory(j["npz"], float(j["T"]), cfg, int(j["seed"]), out_json=j["out"])
            print(f"[analyze] {j['out']}", flush=True)
        except Exception as e:                            # noqa: BLE001
            traceback.print_exc()
            atomic_write_json(dict(error=f"{type(e).__name__}: {str(e)[:300]}", T=j["T"]), j["out"])


def synthetic_brownian_check(cfg: dict, D_true_cm2s: float = 1e-5, n_atoms: int = 64, n_frames: int = 400,
                             dt_fs: float = 100.0, seed: int = 1) -> dict:
    """Spec 12-4: kinisi must recover a known D from a synthetic random walk (within its CI)."""
    from ase import Atoms
    import scipp as sc
    from kinisi.diffusion_analyzer import DiffusionAnalyzer
    rng = np.random.default_rng(seed)
    L = 30.0
    D_A2fs = D_true_cm2s * 1e16 / 1e15          # cm^2/s -> A^2/fs
    sd = math.sqrt(2 * D_A2fs * dt_fs)
    steps = rng.normal(0, sd, size=(n_frames - 1, n_atoms, 3))
    pos = np.concatenate([rng.uniform(0, L, (1, n_atoms, 3)), np.cumsum(steps, axis=0) + rng.uniform(0, L, (1, n_atoms, 3))])
    frames = [Atoms(numbers=[3] * n_atoms + [16] * 4, positions=np.vstack([p, np.array([[1, 1, 1], [15, 15, 15], [1, 15, 1], [15, 1, 15]])]),
                    cell=np.eye(3) * L, pbc=True) for p in pos]
    times = np.arange(n_frames) * dt_fs
    dt = sc.array(dims=["time interval"], values=_kinisi_dt_fs(times, 60), unit="fs")
    an = DiffusionAnalyzer.from_ase(frames, specie="Li", time_step=sc.scalar(dt_fs, unit="fs"),
                                    step_skip=sc.scalar(1, unit="dimensionless"), dt=dt, progress=False)
    an.diffusion(sc.scalar(2000.0, unit="fs"), n_samples=800, n_walkers=32, n_burn=300, n_thin=5, progress=False,
                 random_state=np.random.RandomState(seed))
    D = np.asarray(an.D.values)
    lo, hi = np.quantile(D, 0.025), np.quantile(D, 0.975)
    ok = bool(lo <= D_true_cm2s <= hi)
    return dict(D_true=D_true_cm2s, D_median=float(np.median(D)), ci_low=float(lo), ci_high=float(hi), recovered=ok)


# ============================================================================ aggregation / statistics
def _z(ci_level: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 + ci_level / 2))


def combine_configs_ln_sigma(rows: list[dict]) -> dict:
    """Equal-weight combination of configs in ln sigma: var = mean(within) + between (spec 7.6-3)."""
    mus = np.array([r["ln_sigma_mean"] for r in rows], dtype=float)
    sds = np.array([r["ln_sigma_sd"] for r in rows], dtype=float)
    ok = np.isfinite(mus) & np.isfinite(sds)
    if ok.sum() == 0:
        return dict(mu=float("nan"), var=float("nan"), n=0)
    mus, sds = mus[ok], sds[ok]
    var = float(np.mean(sds ** 2) + (np.var(mus, ddof=1) if len(mus) > 1 else 0.0))
    return dict(mu=float(mus.mean()), var=var, n=int(len(mus)))


def boltzmann_weights(energies_eV: np.ndarray, T: float) -> np.ndarray:
    e = np.asarray(energies_eV, dtype=float)
    w = np.exp(-(e - e.min()) / (K_B_EV * T))
    return w / w.sum()


def lognormal_summary(mu: float, var: float, ci_level: float) -> dict:
    z = _z(ci_level); sd = math.sqrt(var) if np.isfinite(var) else float("nan")
    return dict(median=math.exp(mu), ci_low=math.exp(mu - z * sd), ci_high=math.exp(mu + z * sd)) if np.isfinite(mu) else \
        dict(median=float("nan"), ci_low=float("nan"), ci_high=float("nan"))


def ratio_summary(a: dict, b: dict, ci_level: float) -> dict:
    """a / b for two ln-sigma Gaussians (independent)."""
    mu = a["mu"] - b["mu"]; var = a["var"] + b["var"]
    s = lognormal_summary(mu, var, ci_level)
    return dict(ratio=s["median"], ratio_ci_low=s["ci_low"], ratio_ci_high=s["ci_high"], ln_ratio_mu=mu, ln_ratio_var=var)


def arrhenius_mc(per_T: dict[float, dict], t_extrap: float, n_mc: int, seed: int) -> dict:
    """Monte-Carlo Arrhenius fit.  per_T: {T: {"mu": ln sigma mean, "var": ...}} (sigma in S/cm).

    ln(sigma T) = ln A - Ea/(k T).  Returns samples-derived Ea and ln sigma(t_extrap) (mu/var) and CI.
    """
    Ts = np.array(sorted(per_T), dtype=float)
    if len(Ts) < 2 or any(not np.isfinite(per_T[T]["mu"]) for T in Ts):
        return dict(Ea=float("nan"), Ea_ci_low=float("nan"), Ea_ci_high=float("nan"),
                    ln_sigma_extrap_mu=float("nan"), ln_sigma_extrap_var=float("nan"))
    rng = np.random.default_rng(seed)
    x = 1.0 / (K_B_EV * Ts)
    Y = np.stack([rng.normal(per_T[T]["mu"], math.sqrt(per_T[T]["var"]), n_mc) + math.log(T) for T in Ts], axis=1)  # ln(sigma T)
    xm = x.mean(); Sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm)[None, :] * (Y - Y.mean(axis=1, keepdims=True))).sum(axis=1) / Sxx
    intercept = Y.mean(axis=1) - slope * xm
    Ea = -slope
    ln_sig = intercept + slope / (K_B_EV * t_extrap) - math.log(t_extrap)
    return dict(Ea=float(np.median(Ea)), Ea_ci_low=float(np.quantile(Ea, 0.025)), Ea_ci_high=float(np.quantile(Ea, 0.975)),
                ln_sigma_extrap_mu=float(ln_sig.mean()), ln_sigma_extrap_var=float(ln_sig.var()))


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, quadratic: bool = False) -> dict:
    """Weighted least squares with parameter standard errors (for gate G-lin)."""
    X = np.column_stack([np.ones_like(x), x] + ([x ** 2] if quadratic else []))
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    resid = y - X @ beta
    dof = max(len(x) - X.shape[1], 1)
    s2 = float((w * resid ** 2).sum() / dof)
    cov = s2 * np.linalg.inv(XtW @ X)
    se = np.sqrt(np.diag(cov))
    ybar = (w * y).sum() / w.sum()
    r2 = 1 - (w * resid ** 2).sum() / (w * (y - ybar) ** 2).sum()
    return dict(beta=beta.tolist(), se=se.tolist(), r2=float(r2))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


# ============================================================================ figures (python-pptx, native charts)
def _add_error_bars(series, plus: list[float], minus: list[float]) -> None:
    """Custom y error bars on a python-pptx series (not exposed by the API -> XML)."""
    from lxml import etree
    from pptx.oxml.ns import qn
    ser = series._element
    eb = etree.SubElement(ser, qn("c:errBars"))
    etree.SubElement(eb, qn("c:errDir")).set("val", "y")
    etree.SubElement(eb, qn("c:errBarType")).set("val", "both")
    etree.SubElement(eb, qn("c:errValType")).set("val", "cust")
    etree.SubElement(eb, qn("c:noEndCap")).set("val", "0")
    for tag, vals in (("c:plus", plus), ("c:minus", minus)):
        node = etree.SubElement(eb, qn(tag))
        lit = etree.SubElement(node, qn("c:numLit"))
        etree.SubElement(lit, qn("c:formatCode")).text = "General"
        etree.SubElement(lit, qn("c:ptCount")).set("val", str(len(vals)))
        for i, v in enumerate(vals):
            pt = etree.SubElement(lit, qn("c:pt")); pt.set("idx", str(i))
            etree.SubElement(pt, qn("c:v")).text = f"{0.0 if not np.isfinite(v) else v:.6g}"
    # errBars must precede c:xVal/c:yVal/c:cat/c:val in the schema: move it right after c:marker/c:spPr
    ser.remove(eb)
    anchor = None
    for cand in ("c:xVal", "c:cat", "c:val", "c:yVal"):
        anchor = ser.find(qn(cand))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(eb)
    else:
        ser.append(eb)


def _slide_with_title(prs, title: str, note: str):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    slide.notes_slide.notes_text_frame.text = note
    return slide


def _scatter_chart(slide, series: dict[str, tuple[list, list]], x_title: str, y_title: str, y_log: bool = False,
                   err: dict[str, tuple[list, list]] | None = None):
    from pptx.chart.data import XyChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.util import Inches
    cd = XyChartData()
    for name, (xs, ys) in series.items():
        s = cd.add_series(name)
        for x, y in zip(xs, ys):
            if np.isfinite(x) and np.isfinite(y):
                s.add_data_point(float(x), float(y))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(0.5), Inches(1.5), Inches(9), Inches(5.5), cd)
    ch = gf.chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.RIGHT; ch.legend.include_in_layout = False
    ch.category_axis.has_title = True; ch.category_axis.axis_title.text_frame.text = x_title
    ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = y_title
    if y_log:
        from lxml import etree
        from pptx.oxml.ns import qn
        scaling = ch.value_axis._element.find(qn("c:scaling"))
        lb = etree.SubElement(scaling, qn("c:logBase")); lb.set("val", "10")
        scaling.remove(lb); scaling.insert(0, lb)
    if err:
        for s in ch.plots[0].series:
            if s.name in err:
                plus, minus = err[s.name]
                _add_error_bars(s, plus, minus)
    return ch


def _bar_chart(slide, categories: list[str], series: dict[str, list[float]], y_title: str,
               err: dict[str, tuple[list, list]] | None = None, y_log: bool = False):
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.util import Inches
    cd = CategoryChartData(); cd.categories = categories
    for name, vals in series.items():
        cd.add_series(name, [float(v) if np.isfinite(v) else None for v in vals])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.5), Inches(9), Inches(5.5), cd)
    ch = gf.chart
    ch.has_legend = len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.RIGHT; ch.legend.include_in_layout = False
    ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = y_title
    if y_log:
        from lxml import etree
        from pptx.oxml.ns import qn
        scaling = ch.value_axis._element.find(qn("c:scaling"))
        lb = etree.SubElement(scaling, qn("c:logBase")); lb.set("val", "10")
        scaling.remove(lb); scaling.insert(0, lb)
    if err:
        for s in ch.plots[0].series:
            if s.name in err:
                _add_error_bars(s, *err[s.name])
    return ch


def make_figures(cfg: dict, summary_single: pd.DataFrame | None, model_sel: pd.DataFrame | None,
                 multi: pd.DataFrame | None, top: pd.DataFrame | None, per_T: pd.DataFrame | None,
                 gates: dict | None, out_path: str) -> str:
    """figures.pptx (spec 10).  Every slide is a native, editable chart; the source CSV is in the notes."""
    from pptx import Presentation
    prs = Presentation()
    prs.slide_width, prs.slide_height = 9144000, 6858000
    ci = cfg["ci_level"]

    def _err(df, col):
        v = df[col].values.astype(float)
        lo = df[f"{col}_ci_low"].values.astype(float); hi = df[f"{col}_ci_high"].values.astype(float)
        return (list(np.maximum(hi - v, 0)), list(np.maximum(v - lo, 0)))

    # 1 model selection
    if model_sel is not None and len(model_sel):
        s = _slide_with_title(prs, "1. Model selection: sigma(600 K) of A, F and sigma_F/sigma_A", "source: outputs/results_model_selection.csv")
        cats = list(model_sel["model"])
        _bar_chart(s, cats, {"sigma_A_600K (S/cm)": list(model_sel["sigma_A_600K"]), "sigma_F_600K (S/cm)": list(model_sel["sigma_F_600K"])},
                   "sigma (S/cm)", err={"sigma_A_600K (S/cm)": _err(model_sel, "sigma_A_600K"), "sigma_F_600K (S/cm)": _err(model_sel, "sigma_F_600K")}, y_log=True)
        s2 = _slide_with_title(prs, "1b. Model check: ratio sigma_F/sigma_A (600 K)", "source: outputs/results_model_selection.csv")
        _bar_chart(s2, cats, {"ratio_F_A_600K": list(model_sel["ratio_F_A_600K"])},
                   "sigma_F / sigma_A", err={"ratio_F_A_600K": _err(model_sel, "ratio_F_A_600K")})
    # 2 compensation recovery map, 3 Arrhenius, 4 G-lin
    if summary_single is not None and len(summary_single):
        for model, df in summary_single.groupby("model"):
            pil = df[df.composition_id.isin(PILOT_LEDGER_EXPECTED)]
            if not len(pil):
                continue
            s = _slide_with_title(prs, f"2. Compensation recovery map ({model}): sigma/sigma_A at 600 K vs N_Li per unit cell",
                                  "source: outputs/results_single_summary.csv; pairs X0->X2 are connected")
            ser, err = {}, {}
            for pair, nm in ((("N0", "N2"), "Na (1+)"), (("M0", "M2"), "Mg (2+)"), (("L0", "L2"), "Al (3+)")):
                sub = pil[pil.composition_id.isin(pair)].sort_values("n_li_per_uc")
                ser[nm] = (list(sub.n_li_per_uc), list(sub.ratio_vs_A_600K)); err[nm] = _err(sub, "ratio_vs_A_600K")
            for cid in ("A", "F", "D"):
                sub = pil[pil.composition_id == cid]
                ser[cid] = (list(sub.n_li_per_uc), list(sub.ratio_vs_A_600K)); err[cid] = _err(sub, "ratio_vs_A_600K")
            _scatter_chart(s, ser, "N_Li per unit cell", "sigma / sigma_A (600 K)", y_log=True, err=err)
            if per_T is not None and len(per_T):
                s3 = _slide_with_title(prs, f"3. Arrhenius ({model}): ln sigma vs 1000/T, pilot compositions", "source: outputs/results_single_perT.csv")
                pt = per_T[(per_T.model == model) & per_T.composition_id.isin(PILOT_LEDGER_EXPECTED)]
                ser3, err3 = {}, {}
                for cid, g in pt.groupby("composition_id"):
                    g = g.sort_values("T")
                    ser3[cid] = (list(1000.0 / g["T"]), list(np.log(g.sigma_Scm)))
                    err3[cid] = (list(np.log(g.sigma_ci_high) - np.log(g.sigma_Scm)), list(np.log(g.sigma_Scm) - np.log(g.sigma_ci_low)))
                _scatter_chart(s3, ser3, "1000 / T (1/K)", "ln sigma (S/cm)", err=err3)
            s4 = _slide_with_title(prs, f"4. G-lin ({model}): ln sigma(600 K) vs N_Li with weighted regression line", "source: outputs/results_single_summary.csv + outputs/gates.json")
            lin = pil[pil.composition_id.isin(["A", "F", "N0", "N2", "M0", "M2", "L0", "L2"])]
            ser4 = {"ln sigma(600 K)": (list(lin.n_li), list(np.log(lin.sigma_600K)))}
            err4 = {"ln sigma(600 K)": (list(np.log(lin.sigma_600K_ci_high) - np.log(lin.sigma_600K)), list(np.log(lin.sigma_600K) - np.log(lin.sigma_600K_ci_low)))}
            if gates and gates.get("G-lin", {}).get("model") == model and gates["G-lin"].get("beta_linear"):
                b0, b1 = gates["G-lin"]["beta_linear"][:2]
                xs = np.linspace(lin.n_li.min(), lin.n_li.max(), 20)
                ser4["weighted fit"] = (list(xs), list(b0 + b1 * xs))
            _scatter_chart(s4, ser4, "N_Li (supercell)", "ln sigma (S/cm)", err=err4)
    # 5 stability, 6 S_syn, 7 top candidates
    if multi is not None and len(multi):
        s5 = _slide_with_title(prs, "5. Stage II stability: de_hull_corr_vs_host (eV/atom) distribution", "source: outputs/results_multi.csv; threshold line = CONFIG['stab_threshold_eV']")
        vals = multi.de_hull_corr_vs_host.values.astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            hist, edges = np.histogram(vals, bins=30)
            cats = [f"{(a + b) / 2:.3f}" for a, b in zip(edges[:-1], edges[1:])]
            _bar_chart(s5, cats, {"count": list(hist.astype(float))}, "compositions")
        s6 = _slide_with_title(prs, "6. S_syn by Li group (short MD, 600 K)", "source: outputs/results_multi.csv")
        ok = multi[np.isfinite(multi.S_syn_short.astype(float))]
        if len(ok):
            _scatter_chart(s6, {"S_syn_short": (list(ok.li_group.astype(float)), list(ok.S_syn_short.astype(float)))},
                           "Li group (N_Li per supercell)", "S_syn (short MD)", y_log=True,
                           err={"S_syn_short": _err(ok, "S_syn_short")})
    if top is not None and len(top):
        s7 = _slide_with_title(prs, "7. Top candidates per Li group: S_syn (long MD) main vs contrast model", "source: outputs/top_candidates.csv")
        cats = [f"{r.composition_id} [{int(r.li_group)}]" for r in top.itertuples()]
        ser7 = {"S_syn_long (main)": list(top.S_syn_long.astype(float))}
        err7 = {"S_syn_long (main)": _err(top, "S_syn_long")}
        if "S_syn_contrast" in top and np.isfinite(top.S_syn_contrast.astype(float)).any():
            ser7["S_syn_contrast"] = list(top.S_syn_contrast.astype(float)); err7["S_syn_contrast"] = _err(top, "S_syn_contrast")
        _bar_chart(s7, cats, ser7, "S_syn", err=err7)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    return out_path


# ============================================================================ DFT hand-off
def export_dft_handoff(cfg: dict, main_model: str, pilot_ids: list[str], top_ids: list[str],
                       relaxed_keys: dict[str, list[str]], md_keys: dict[str, list[tuple[str, float]]],
                       ref_ids_by_model: dict[str, list[str]], manifest: dict, n_frames: int = 20) -> str:
    """dft_handoff/ (spec 11): POSCARs of the main-model relaxed configs, 20 evenly spaced NVT frames, references."""
    from ase.io import write
    root = Path(cfg["workdir"], "outputs", "dft_handoff")
    root.mkdir(parents=True, exist_ok=True)
    meta_all = {}
    for group, ids in (("pilot", pilot_ids), ("top_candidates", top_ids)):
        for cid in ids:
            d = root / group / cid
            d.mkdir(parents=True, exist_ok=True)
            for key in relaxed_keys.get(cid, []):
                p_xyz, p_json = relax_job_paths(cfg, main_model, key)
                if not os.path.exists(p_xyz):
                    continue
                a = read_atoms(p_xyz)
                k = key.split("__")[-1]
                write(str(d / f"relaxed_config_{k}.vasp"), a, format="vasp", sort=True, direct=True)
                meta_all[f"{group}/{cid}/relaxed_config_{k}"] = dict(json.load(open(p_json)), source_key=key)
            for key, T in md_keys.get(cid, []):
                npz = os.path.join(md_dir(cfg, main_model, key), "nvt.npz")
                if not os.path.exists(npz):
                    continue
                z = np.load(npz)
                idx = np.linspace(0, len(z["positions"]) - 1, n_frames).astype(int)
                from ase import Atoms
                frames = []
                for i in idx:
                    at = Atoms(numbers=z["numbers"], positions=z["positions"][i], cell=z["cell"], pbc=True)
                    at.wrap(); at.info["time_fs"] = float(z["times_fs"][i]); at.info["source_key"] = key
                    frames.append(at)
                write(str(d / f"md_frames_{int(T)}K.extxyz"), frames, format="extxyz")
    for model, ids in ref_ids_by_model.items():
        d = root / "references" / model
        d.mkdir(parents=True, exist_ok=True)
        for mid in ids:
            p_xyz, _ = relax_job_paths(cfg, model, f"REF__{mid}")
            if os.path.exists(p_xyz):
                write(str(d / f"{mid}.vasp"), read_atoms(p_xyz), format="vasp", sort=True, direct=True)
    atomic_write_json(meta_all, str(root / "metadata.json"))
    with open(root / "README.md", "w") as f:
        f.write("# DFT hand-off\n\nStructures only; no DFT is run by this pipeline.\n\n")
        f.write(f"* main model: {main_model} ({manifest.get('models', {}).get(main_model, {}).get('tag', '?')})\n")
        f.write(f"* master seed: {cfg['seed']}\n* pilot/<id>/relaxed_config_<k>.vasp: main-model relaxed configurations (POSCAR)\n")
        f.write("* pilot/<id>/md_frames_<T>K.extxyz: 20 evenly spaced NVT frames (wrapped)\n")
        f.write("* top_candidates/<id>/: same for the final Stage II candidates\n* references/<model>/: relaxed hull reference phases\n")
        f.write("* metadata.json: energies, checkpoint tags and the original config keys\n\n## Compositions\n\n")
        for c in pilot_compositions():
            f.write(f"* {c['id']}: li_site={c['li_site']} p_site={c['p_site']} N_Li={n_li_of(c)}\n")
    return str(root)



# ============================================================================ pipeline bookkeeping / aggregation
def relax_key(comp_id: str, cid: str) -> str:
    return f"{comp_id}__{cid}"


def md_key(comp_id: str, cid: str, T: float) -> str:
    return f"{comp_id}__{cid}__{int(round(T))}K"


def analysis_path(cfg: dict, model: str, key: str) -> str:
    return os.path.join(md_dir(cfg, model, key), "analysis.json")


def generated_path(cfg: dict, key: str, salt: str = "") -> str:
    sub = "generated" if not salt else f"generated_{salt}"
    return wpath(cfg, "outputs", "structures", sub, f"{key}.extxyz")


def ensure_generated(host: Host, comps: list[dict], stage: str, cfg: dict, salt: str = "", cid_prefix: str = "c") -> dict:
    """Generate (or load) every configuration of the given compositions.  Returns {key: meta}."""
    meta_path = wpath(cfg, "outputs", "structures", f"generated{('_' + salt) if salt else ''}_meta.json")
    metas = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    changed = False
    for comp in comps:
        for k, mode in enumerate(config_modes(stage)):
            cid = f"{cid_prefix}{k}"
            key = relax_key(comp["id"], cid)
            p = generated_path(cfg, key, salt)
            if key in metas and os.path.exists(p):
                continue
            seed = derived_seed(cfg["seed"], "config", comp["id"], k, salt)
            atoms, meta = build_config(host, comp, mode, seed, cfg)
            meta["config_id"] = cid; meta["key"] = key
            write_atoms(atoms, p, info=meta)
            metas[key] = meta; changed = True
    if changed:
        atomic_write_json(metas, meta_path)
    return metas


def relax_jobs_for(cfg: dict, metas: dict, keys: list[str], salt: str = "") -> list[dict]:
    return [dict(key=k, input=generated_path(cfg, k, salt), n_atoms=metas[k]["n_atoms"], relax_cell=cfg["relax_cell"]) for k in keys]


def select_md_configs(relax_df: pd.DataFrame, comp_id: str, n: int) -> list[str]:
    """The n lowest-energy CONVERGED configurations of a composition (spec 7.3)."""
    d = relax_df[(relax_df.composition_id == comp_id) & relax_df.converged.astype(bool)].sort_values("e_per_atom")
    return list(d.config_id.head(n))


def md_jobs_for(cfg: dict, model: str, comp_id: str, cids: list[str], temps: list[float], npt_ps: float, nvt_ps: float,
                relaxed_model: str | None = None) -> list[dict]:
    """MD jobs for one composition: relaxed structure of `relaxed_model` (default: same model)."""
    jobs = []
    for cid in cids:
        p_xyz, _ = relax_job_paths(cfg, relaxed_model or model, relax_key(comp_id, cid))
        for T in temps:
            jobs.append(dict(key=md_key(comp_id, cid, T), model=model, composition_id=comp_id, config_id=cid, T=float(T),
                             input=p_xyz, npt_ps=float(npt_ps), nvt_ps=float(nvt_ps)))
    return jobs


def analyze_jobs_for(cfg: dict, model: str, md_jobs: list[dict]) -> list[dict]:
    out = []
    for j in md_jobs:
        d = md_dir(cfg, model, j["key"])
        npz = os.path.join(d, "nvt.npz")
        if os.path.exists(npz):
            out.append(dict(npz=npz, T=j["T"], out=os.path.join(d, "analysis.json"), seed=derived_seed(cfg["seed"], "kinisi", model, j["key"])))
    return out


def relax_table(cfg: dict, model: str, metas: dict, keys: list[str]) -> pd.DataFrame:
    """Relaxation results joined with the generation metadata."""
    rows = []
    for k in keys:
        _, pj = relax_job_paths(cfg, model, k)
        if not os.path.exists(pj):
            continue
        r = json.load(open(pj)); m = metas[k]
        rows.append(dict(model=model, key=k, composition_id=m["composition_id"], config_id=m["config_id"], config_mode=m["config_mode"],
                         seed=m["seed"], converged=bool(r["converged"]), energy=r["energy"], n_atoms=r["n_atoms"], e_per_atom=r["e_per_atom"],
                         a=r.get("a", np.nan), b=r.get("b", np.nan), c=r.get("c", np.nan), volume=r.get("volume", np.nan), fmax=r["fmax"],
                         n_interstitial_16e=m["n_interstitial_16e"], n_interstitial_other=m["n_interstitial_other"],
                         n_vacancy_near=m["n_vacancy_near"], n_vacancy_far=m["n_vacancy_far"],
                         d_dopant_ge_min=m["d_dopant_ge_min"], d_dopant_vacancy_min=m["d_dopant_vacancy_min"], error=r.get("error", "")))
    return pd.DataFrame(rows)


def hull_table(cfg: dict, model: str, relax_df: pd.DataFrame, comps: list[dict], pool: dict, host_id: str = "A") -> pd.DataFrame:
    """e_above_hull / de_hull_vs_host per config, one PhaseDiagram per chemical system, ONE model (spec 7.4)."""
    comp_by_id = {c["id"]: c for c in comps}
    pds = {}
    rows = []
    for r in relax_df.itertuples():
        if not np.isfinite(r.energy):
            rows.append(dict(model=model, key=r.key, e_above_hull=np.nan)); continue
        comp = comp_by_id[r.composition_id]
        els = tuple(chemsys_elements(comp))
        if els not in pds:
            pds[els] = build_phase_diagram(cfg, model, reference_ids_for(pool, list(els)))
        a = read_atoms(relax_job_paths(cfg, model, r.key)[0])
        rows.append(dict(model=model, key=r.key, e_above_hull=e_above_hull(pds[els], a, float(r.energy))))
    df = pd.DataFrame(rows)
    out = relax_df.merge(df, on=["model", "key"], how="left")
    host = out[(out.composition_id == host_id) & out.converged]
    eh_host = float(host.e_above_hull.min()) if len(host) else np.nan
    out["e_above_hull_host"] = eh_host
    out["de_hull_vs_host"] = out.e_above_hull - eh_host
    return out


def results_single_table(cfg: dict, model: str, comps: list[dict], hull_df: pd.DataFrame, temps: list[float],
                         stage_label: str) -> pd.DataFrame:
    """results_single.csv rows for one model (spec 7.7): one row per (composition, config, T)."""
    rows = []
    for comp in comps:
        d = hull_df[hull_df.composition_id == comp["id"]]
        conv = d[d.converged]
        w = boltzmann_weights(conv.energy.values, cfg["t_syn"]) if len(conv) else np.array([])
        wmap = dict(zip(conv.config_id, w))
        for r in d.itertuples():
            base = dict(model=model, composition_id=comp["id"], stage=stage_label, li_site=json.dumps(comp.get("li_site", {}), sort_keys=True),
                        p_site=json.dumps(comp.get("p_site", {}), sort_keys=True), n_li=n_li_of(comp), n_li_per_uc=n_li_of(comp) / N_UC_IN_SUPERCELL,
                        config_id=r.config_id, config_mode=r.config_mode, seed=r.seed, converged=r.converged, e_per_atom=r.e_per_atom,
                        e_above_hull=r.e_above_hull, de_hull_vs_host=r.de_hull_vs_host, boltzmann_weight=wmap.get(r.config_id, np.nan),
                        a=r.a, b=r.b, c=r.c, volume=r.volume)
            any_T = False
            for T in temps:
                ap = analysis_path(cfg, model, md_key(comp["id"], r.config_id, T))
                if not os.path.exists(ap):
                    continue
                an = json.load(open(ap))
                if "error" in an:
                    continue
                any_T = True
                rows.append(dict(base, T=float(T), D_star=an["D_star"], D_star_ci_low=an["D_star_ci_low"], D_star_ci_high=an["D_star_ci_high"],
                                 sigma_Scm=an["sigma_Scm"], sigma_ci_low=an["sigma_ci_low"], sigma_ci_high=an["sigma_ci_high"],
                                 ln_sigma_mean=an["ln_sigma_mean"], ln_sigma_sd=an["ln_sigma_sd"],
                                 flag_nondiffusive=an["flag_nondiffusive"], msd_loglog_slope=an["msd_loglog_slope"],
                                 n_interstitial_16e=r.n_interstitial_16e, n_interstitial_other=r.n_interstitial_other,
                                 n_vacancy_near=r.n_vacancy_near, n_vacancy_far=r.n_vacancy_far))
            if not any_T:
                rows.append(dict(base, T=np.nan, D_star=np.nan, D_star_ci_low=np.nan, D_star_ci_high=np.nan, sigma_Scm=np.nan,
                                 sigma_ci_low=np.nan, sigma_ci_high=np.nan, ln_sigma_mean=np.nan, ln_sigma_sd=np.nan,
                                 flag_nondiffusive=np.nan, msd_loglog_slope=np.nan,
                                 n_interstitial_16e=r.n_interstitial_16e, n_interstitial_other=r.n_interstitial_other,
                                 n_vacancy_near=r.n_vacancy_near, n_vacancy_far=r.n_vacancy_far))
    return pd.DataFrame(rows)


def per_T_table(cfg: dict, rs: pd.DataFrame, host_id: str = "A") -> pd.DataFrame:
    """Composition x T combination of configs (equal weight + Boltzmann sensitivity) and ratio vs host."""
    ci = cfg["ci_level"]
    rows = []
    d = rs[np.isfinite(rs["T"].astype(float)) & np.isfinite(rs.ln_sigma_mean.astype(float))]
    for (model, cid, T), g in d.groupby(["model", "composition_id", "T"]):
        recs = g.to_dict("records")
        comb = combine_configs_ln_sigma(recs)
        w = g.boltzmann_weight.values.astype(float); w = w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else np.full(len(g), 1 / len(g))
        mu_b = float((w * g.ln_sigma_mean.values).sum())
        s = lognormal_summary(comb["mu"], comb["var"], ci)
        rows.append(dict(model=model, composition_id=cid, T=float(T), n_configs=comb["n"], ln_sigma_mu=comb["mu"], ln_sigma_var=comb["var"],
                         sigma_Scm=s["median"], sigma_ci_low=s["ci_low"], sigma_ci_high=s["ci_high"], sigma_boltzmann=math.exp(mu_b),
                         n_flag_nondiffusive=int(g.flag_nondiffusive.astype(bool).sum())))
    pt = pd.DataFrame(rows)
    if not len(pt):
        return pt
    for col in ("ratio_vs_A", "ratio_vs_A_ci_low", "ratio_vs_A_ci_high"):
        pt[col] = np.nan
    for i, r in pt.iterrows():
        h = pt[(pt.model == r.model) & (pt.composition_id == host_id) & (pt["T"] == r["T"])]
        if len(h):
            rr = ratio_summary(dict(mu=r.ln_sigma_mu, var=r.ln_sigma_var), dict(mu=float(h.ln_sigma_mu.iloc[0]), var=float(h.ln_sigma_var.iloc[0])), ci)
            pt.loc[i, ["ratio_vs_A", "ratio_vs_A_ci_low", "ratio_vs_A_ci_high"]] = [rr["ratio"], rr["ratio_ci_low"], rr["ratio_ci_high"]]
    return pt


def summary_table(cfg: dict, pt: pd.DataFrame, comps: list[dict], hull_df_by_model: dict[str, pd.DataFrame],
                  T_ref: float = 600.0, host_id: str = "A") -> pd.DataFrame:
    """results_single_summary.csv: sigma(T_ref), Ea, sigma(300 K) with CIs, ratios vs host (spec 7.7)."""
    ci = cfg["ci_level"]; t_ex = float(cfg["t_extrapolate"])
    comp_by_id = {c["id"]: c for c in comps}
    rows = []
    for (model, cid), g in pt.groupby(["model", "composition_id"]):
        per_T = {float(r["T"]): dict(mu=float(r["ln_sigma_mu"]), var=float(r["ln_sigma_var"])) for r in g.to_dict("records")}
        arr = arrhenius_mc(per_T, t_ex, int(cfg["n_mc_samples"]), derived_seed(cfg["seed"], "arr", model, cid))
        ref = g[g["T"] == T_ref]
        row = dict(model=model, composition_id=cid, n_li=n_li_of(comp_by_id[cid]), n_li_per_uc=n_li_of(comp_by_id[cid]) / N_UC_IN_SUPERCELL,
                   li_group=n_li_of(comp_by_id[cid]), n_temperatures=len(per_T))
        if len(ref):
            row.update(sigma_600K=float(ref.sigma_Scm.iloc[0]), sigma_600K_ci_low=float(ref.sigma_ci_low.iloc[0]), sigma_600K_ci_high=float(ref.sigma_ci_high.iloc[0]),
                       sigma_600K_boltzmann=float(ref.sigma_boltzmann.iloc[0]), ratio_vs_A_600K=float(ref.ratio_vs_A.iloc[0]),
                       ratio_vs_A_600K_ci_low=float(ref.ratio_vs_A_ci_low.iloc[0]), ratio_vs_A_600K_ci_high=float(ref.ratio_vs_A_ci_high.iloc[0]))
        else:
            row.update(sigma_600K=np.nan, sigma_600K_ci_low=np.nan, sigma_600K_ci_high=np.nan, sigma_600K_boltzmann=np.nan,
                       ratio_vs_A_600K=np.nan, ratio_vs_A_600K_ci_low=np.nan, ratio_vs_A_600K_ci_high=np.nan)
        s300 = lognormal_summary(arr["ln_sigma_extrap_mu"], arr["ln_sigma_extrap_var"], ci)
        row.update(Ea=arr["Ea"], Ea_ci_low=arr["Ea_ci_low"], Ea_ci_high=arr["Ea_ci_high"], sigma_300K=s300["median"], sigma_300K_ci_low=s300["ci_low"],
                   sigma_300K_ci_high=s300["ci_high"], ln_sigma_300K_mu=arr["ln_sigma_extrap_mu"], ln_sigma_300K_var=arr["ln_sigma_extrap_var"])
        hd = hull_df_by_model.get(model)
        if hd is not None and len(hd):
            sub = hd[(hd.composition_id == cid) & hd.converged]
            row["de_hull_vs_host_mean"] = float(sub.de_hull_vs_host.mean()) if len(sub) else np.nan
            row["e_above_hull_min"] = float(sub.e_above_hull.min()) if len(sub) else np.nan
        else:
            row["de_hull_vs_host_mean"] = np.nan; row["e_above_hull_min"] = np.nan
        row["notes"] = "; ".join(f"{int(r['T'])}K:{int(r['n_flag_nondiffusive'])}nondiff" for r in g.to_dict("records") if r["n_flag_nondiffusive"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    for col in ("ratio_vs_A_300K", "ratio_vs_A_300K_ci_low", "ratio_vs_A_300K_ci_high"):
        df[col] = np.nan
    for i, r in df.iterrows():
        h = df[(df.model == r.model) & (df.composition_id == host_id)]
        if len(h) and np.isfinite(r.ln_sigma_300K_mu):
            rr = ratio_summary(dict(mu=r.ln_sigma_300K_mu, var=r.ln_sigma_300K_var), dict(mu=float(h.ln_sigma_300K_mu.iloc[0]), var=float(h.ln_sigma_300K_var.iloc[0])), ci)
            df.loc[i, ["ratio_vs_A_300K", "ratio_vs_A_300K_ci_low", "ratio_vs_A_300K_ci_high"]] = [rr["ratio"], rr["ratio_ci_low"], rr["ratio_ci_high"]]
    return df


def _ratio_from_pt(pt: pd.DataFrame, model: str, num: str, den: str, T: float, ci: float) -> dict:
    a = pt[(pt.model == model) & (pt.composition_id == num) & (pt["T"] == T)]
    b = pt[(pt.model == model) & (pt.composition_id == den) & (pt["T"] == T)]
    if not len(a) or not len(b):
        return dict(ratio=np.nan, ratio_ci_low=np.nan, ratio_ci_high=np.nan)
    return ratio_summary(dict(mu=float(a.ln_sigma_mu.iloc[0]), var=float(a.ln_sigma_var.iloc[0])),
                         dict(mu=float(b.ln_sigma_mu.iloc[0]), var=float(b.ln_sigma_var.iloc[0])), ci)


def model_selection_table(cfg: dict, summary: pd.DataFrame, pt: pd.DataFrame, relax_dfs: dict[str, pd.DataFrame],
                          throughput: dict[str, float], model_tags: dict[str, str], exp_ref: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Stage I-M (spec 6.4 / 6.5).  Returns the table and {'main': ..., 'contrast': ..., 'rule': ...}."""
    ci = cfg["ci_level"]
    rows = []
    for model in cfg["models"]:
        sA = summary[(summary.model == model) & (summary.composition_id == "A")]
        sF = summary[(summary.model == model) & (summary.composition_id == "F")]
        r600 = _ratio_from_pt(pt, model, "F", "A", 600.0, ci)
        rel = relax_dfs.get(model)
        a_rel = float(rel[(rel.composition_id == "A") & rel.converged].a.mean()) / 2.0 if rel is not None and len(rel) else np.nan
        row = dict(model=model, model_tag=model_tags.get(model, "?"), a_relaxed=a_rel, throughput_ns_per_day=throughput.get(model, np.nan))
        for nm, s in (("A", sA), ("F", sF)):
            for col in ("sigma_600K", "sigma_600K_ci_low", "sigma_600K_ci_high", "sigma_300K", "sigma_300K_ci_low", "sigma_300K_ci_high", "Ea", "Ea_ci_low", "Ea_ci_high"):
                row[f"{col.replace('sigma', 'sigma_' + nm).replace('Ea', 'Ea_' + nm)}"] = float(s[col].iloc[0]) if len(s) else np.nan
        row.update(ratio_F_A_600K=r600["ratio"], ratio_F_A_600K_ci_low=r600["ratio_ci_low"], ratio_F_A_600K_ci_high=r600["ratio_ci_high"])
        if len(sF):
            row.update(ratio_F_A_300K=float(sF.ratio_vs_A_300K.iloc[0]), ratio_F_A_300K_ci_low=float(sF.ratio_vs_A_300K_ci_low.iloc[0]),
                       ratio_F_A_300K_ci_high=float(sF.ratio_vs_A_300K_ci_high.iloc[0]))
        else:
            row.update(ratio_F_A_300K=np.nan, ratio_F_A_300K_ci_low=np.nan, ratio_F_A_300K_ci_high=np.nan)
        row["passes_ge_trend"] = bool(np.isfinite(r600["ratio_ci_low"]) and r600["ratio_ci_low"] > 1.0)
        # optional experimental comparison (spec 6.3 / 6.5-4)
        row["exp_sigma_in_range"] = row["exp_Ea_in_range"] = row["exp_a_in_range"] = ""
        if exp_ref is not None and len(exp_ref):
            def _in(q, val):
                e = exp_ref[exp_ref.quantity == q]
                if not len(e) or pd.isna(e.value_low.iloc[0]) or pd.isna(e.value_high.iloc[0]) or not np.isfinite(val):
                    return ""
                return str(float(e.value_low.iloc[0]) <= val <= float(e.value_high.iloc[0]))
            row["exp_sigma_in_range"] = _in("LPSCl_sigma_RT_mScm", row["sigma_A_300K"] * 1000.0)
            row["exp_Ea_in_range"] = _in("LPSCl_Ea_eV", row["Ea_A"])
            row["exp_a_in_range"] = _in("LPSCl_lattice_a_A", a_rel)
        rows.append(row)
    df = pd.DataFrame(rows)
    passing = [r.model for r in df.itertuples() if r.passes_ge_trend]
    if len(cfg["models"]) == 1:
        main = passing[0] if passing else None
        rule = "single model: passes G-a -> main" if main else "single model FAILS G-a (sigma_F/sigma_A CI lower bound > 1) -> STOP and report"
    elif len(passing) == len(cfg["models"]):
        main = max(passing, key=lambda m: throughput.get(m, 0.0)); rule = "both pass G-a -> faster model is main"
    elif len(passing) == 1:
        main = passing[0]; rule = "only one model passes G-a -> it is main; disagreement recorded"
    else:
        main = None; rule = "NO model passes G-a (sigma_F/sigma_A CI lower bound > 1) -> STOP and report"
    contrast = next((m for m in cfg["models"] if m != main), None) if main else None     # None in a single-model run
    df["role"] = df.model.map(lambda m: "main" if m == main else ("contrast" if m == contrast else "undecided"))
    df["selection_rule"] = rule
    return df, dict(main=main, contrast=contrast, rule=rule)


def evaluate_gates(cfg: dict, summary: pd.DataFrame, pt: pd.DataFrame, main: str, contrast: str | None) -> dict:
    """Spec section 9 gates (frozen in PILOT_CRITERIA.md)."""
    ci = cfg["ci_level"]; z = _z(ci)
    g = {}
    r = _ratio_from_pt(pt, main, "F", "A", 600.0, ci)
    g["G-a"] = dict(model=main, ratio_F_A_600K=r["ratio"], ci_low=r["ratio_ci_low"], ci_high=r["ratio_ci_high"],
                    passed=bool(np.isfinite(r["ratio_ci_low"]) and r["ratio_ci_low"] > 1), on_fail="STOP")
    gb = {}
    for X in ("N", "M", "L"):
        rr = _ratio_from_pt(pt, main, f"{X}2", f"{X}0", 600.0, ci)
        gb[X] = dict(ratio=rr["ratio"], ci_low=rr["ratio_ci_low"], ci_high=rr["ratio_ci_high"], passed=bool(np.isfinite(rr["ratio_ci_low"]) and rr["ratio_ci_low"] > 1))
    g["G-b"] = dict(model=main, per_valence=gb, on_fail="record")
    rd = _ratio_from_pt(pt, main, "D", "A", 600.0, ci)
    g["G-c"] = dict(model=main, ratio_D_A_600K=rd["ratio"], ci_low=rd["ratio_ci_low"], ci_high=rd["ratio_ci_high"],
                    effect_detected=bool(np.isfinite(rd["ratio_ci_low"]) and (rd["ratio_ci_low"] > 1 or rd["ratio_ci_high"] < 1)), on_fail="record")
    lin_ids = ["A", "F", "N0", "N2", "M0", "M2", "L0", "L2"]
    s = pt[(pt.model == main) & (pt["T"] == 600.0) & pt.composition_id.isin(lin_ids)]
    nli = {c["id"]: n_li_of(c) for c in pilot_compositions()}
    if len(s) >= 4:
        x = np.array([nli[c] for c in s.composition_id], float); y = s.ln_sigma_mu.values.astype(float); w = 1.0 / np.maximum(s.ln_sigma_var.values.astype(float), 1e-6)
        lin = weighted_linear_fit(x, y, w); quad = weighted_linear_fit(x, y, w, quadratic=True)
        b2, se2 = quad["beta"][2], quad["se"][2]
        curv_ok = bool((b2 - z * se2) <= 0 <= (b2 + z * se2))
        g["G-lin"] = dict(model=main, n=int(len(s)), r2=lin["r2"], beta_linear=lin["beta"], quad_coeff=b2, quad_coeff_ci=[b2 - z * se2, b2 + z * se2],
                          passed=bool(lin["r2"] >= 0.8 and curv_ok), on_fail="S_syn denominator -> interpolation")
    else:
        g["G-lin"] = dict(model=main, n=int(len(s)), passed=False, reason="fewer than 4 compositions with 600 K data")
    if contrast:
        a = pt[(pt.model == main) & (pt["T"] == 600.0)].set_index("composition_id").ratio_vs_A
        b = pt[(pt.model == contrast) & (pt["T"] == 600.0)].set_index("composition_id").ratio_vs_A
        common = [c for c in a.index if c in b.index and c != "A"]
        rho = spearman(a.loc[common].values.astype(float), b.loc[common].values.astype(float)) if len(common) >= 3 else np.nan
        g["G-model"] = dict(main=main, contrast=contrast, n=len(common), spearman_rho=rho, passed=bool(np.isfinite(rho) and rho >= 0.6), on_fail="report")
    g["G-stab"] = dict(threshold_eV=cfg.get("stab_threshold_eV"), status="set" if cfg.get("stab_threshold_eV") is not None else "NOT SET (user decision after Stage I)")
    return g


# ---------------------------------------------------------------------------- Stage II
def stage2_stability_table(cfg: dict, model: str, comps: list[dict], hull_df: pd.DataFrame, host_eh: float) -> pd.DataFrame:
    """Composition-level stability with the ideal-mixing entropy upper bound (spec 8.4)."""
    thr = cfg.get("stab_threshold_eV")
    rows = []
    for comp in comps:
        d = hull_df[(hull_df.composition_id == comp["id"]) & hull_df.converged]
        row = comp_summary_row(comp)
        if len(d):
            best = d.sort_values("energy").iloc[0]
            eh = float(best.e_above_hull); n_atoms = int(best.n_atoms)
        else:
            eh = np.nan; n_atoms = N_ATOMS_HOST
        dS = configurational_entropy_eV_per_K(comp)
        eh_corr = eh - cfg["t_syn"] * dS / n_atoms
        row.update(model=model, n_configs_converged=int(len(d)), e_above_hull=eh, e_above_hull_mean=float(d.e_above_hull.mean()) if len(d) else np.nan,
                   de_hull_vs_host=eh - host_eh, dS_conf_meV_per_K=dS * 1000.0, e_hull_corr=eh_corr, de_hull_corr_vs_host=eh_corr - host_eh,
                   entropy_assumption="ideal mixing on 192 regular Li sites (upper bound); interstitial Li excluded",
                   pass_stability=bool(np.isfinite(eh_corr) and thr is not None and (eh_corr - host_eh) <= thr))
        rows.append(row)
    return pd.DataFrame(rows)


def s_syn_table(cfg: dict, pt: pd.DataFrame, model: str, mixtures: list[dict], baselines: list[dict], T: float,
                use_interpolation: bool, label: str) -> pd.DataFrame:
    """S_syn = sigma_mix / geometric mean of the single-dopant sigmas at the same Ge level (spec 8.5).

    If gate G-lin failed (`use_interpolation`), the denominator is corrected by the quadratic fit f(N_Li)
    of ln sigma over all baselines at the same Ge level:
        ln denom = sum_i w_i ln sigma_i + [f(N_Li_mix) - sum_i w_i f(N_Li_i)]
    """
    ci = cfg["ci_level"]
    base_by_id = {b["id"]: b for b in baselines}
    d = pt[(pt.model == model) & (pt["T"] == T)].set_index("composition_id")
    fits = {}
    if use_interpolation:
        for g in (0, 8, 16):
            ids = [b["id"] for b in baselines if b["p_site"].get("Ge", 0) == g and b["id"] in d.index]
            if len(ids) >= 4:
                x = np.array([n_li_of(base_by_id[i]) for i in ids], float); y = d.loc[ids].ln_sigma_mu.values.astype(float)
                w = 1.0 / np.maximum(d.loc[ids].ln_sigma_var.values.astype(float), 1e-6)
                fits[g] = np.array(weighted_linear_fit(x, y, w, quadratic=True)["beta"])
    rows = []
    for comp in mixtures:
        row = dict(composition_id=comp["id"], model=model, T=T)
        terms = s_syn_denominator_terms(comp)
        if comp["id"] not in d.index or any(t[0] not in d.index for t in terms):
            row.update({f"S_syn_{label}": np.nan, f"S_syn_{label}_ci_low": np.nan, f"S_syn_{label}_ci_high": np.nan,
                        f"sigma_600K_{label}": np.nan, f"sigma_600K_{label}_ci_low": np.nan, f"sigma_600K_{label}_ci_high": np.nan})
            rows.append(row); continue
        mix = d.loc[comp["id"]]
        mu_den = sum(w * float(d.loc[b].ln_sigma_mu) for b, w in terms); var_den = sum(w ** 2 * float(d.loc[b].ln_sigma_var) for b, w in terms)
        g = comp["p_site"].get("Ge", 0)
        if use_interpolation and g in fits:
            f = lambda n, beta=fits[g]: beta[0] + beta[1] * n + beta[2] * n * n
            n_mix = n_li_of(comp)
            mu_den += f(n_mix) - sum(w * f(n_li_of(base_by_id[b])) for b, w in terms)
        s = ratio_summary(dict(mu=float(mix.ln_sigma_mu), var=float(mix.ln_sigma_var)), dict(mu=mu_den, var=var_den), ci)
        sig = lognormal_summary(float(mix.ln_sigma_mu), float(mix.ln_sigma_var), ci)
        row.update({f"S_syn_{label}": s["ratio"], f"S_syn_{label}_ci_low": s["ratio_ci_low"], f"S_syn_{label}_ci_high": s["ratio_ci_high"],
                    f"ln_S_syn_{label}_mu": s["ln_ratio_mu"], f"ln_S_syn_{label}_var": s["ln_ratio_var"],
                    f"sigma_600K_{label}": sig["median"], f"sigma_600K_{label}_ci_low": sig["ci_low"], f"sigma_600K_{label}_ci_high": sig["ci_high"],
                    "denominator": "interpolated" if (use_interpolation and g in fits) else "geometric_mean"})
        rows.append(row)
    return pd.DataFrame(rows)


def select_top_by_group(df: pd.DataFrame, score_col: str, n: int = 5) -> pd.DataFrame:
    """Top-n per li_group by the score (no pooled ranking, spec 8.6)."""
    d = df[np.isfinite(df[score_col].astype(float))].copy()
    d["rank_in_group"] = d.groupby("li_group")[score_col].rank(ascending=False, method="first")
    d["selected_for_long"] = d.rank_in_group <= n
    return d


def estimate_gpu_hours(n_md: int, nvt_ps: float, npt_ps: float, ns_per_day_per_gpu: float, n_gpus: int) -> dict:
    """Wall-clock estimate for a set of MD runs (aggregate per-GPU throughput)."""
    total_ns = n_md * (nvt_ps + npt_ps) / 1000.0
    gpu_days = total_ns / max(ns_per_day_per_gpu, 1e-9)
    return dict(n_md=n_md, total_ns=total_ns, gpu_hours=gpu_days * 24.0, wall_hours=gpu_days * 24.0 / max(n_gpus, 1))


# ============================================================================ worker entry point
def _shard(items: list, worker_id: int, n_workers: int) -> list:
    return [x for i, x in enumerate(items) if i % n_workers == worker_id]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["relax", "md", "analyze"])
    ap.add_argument("--model", default="orb_v3")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--config", default="outputs/config.json")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    jobs = json.load(open(args.jobs))
    mine = _shard(jobs, args.worker_id, args.n_workers)
    print(f"[worker {args.worker_id}/{args.n_workers}] task={args.task} model={args.model} jobs={len(mine)}", flush=True)
    if args.task == "analyze":
        run_analyze_jobs(mine, cfg)
        return
    handle = load_model("mock" if args.model.startswith("mock") else args.model, cfg)
    handle.name = args.model                      # any "mock*" label maps to the Lennard-Jones test model
    print(f"[worker {args.worker_id}] model {handle.name}: {handle.tag} backend={handle.backend} device={handle.device}", flush=True)
    if args.task == "relax":
        run_relax_jobs(handle, mine, cfg)
    else:
        run_md_jobs(handle, mine, cfg)
    print(f"[worker {args.worker_id}] done", flush=True)


if __name__ == "__main__":
    if sys.platform == "darwin":                  # local CPU tests only: anaconda ships two libomp copies
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
