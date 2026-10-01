"""heo_worker.py - HEO Na layered-cathode MLIP screening worker (RunPod multi-GPU).

Engine  : v1, proven end-to-end on 7,279 compositions.
          multiprocessing SQS pre-generation, torch-sim batch relax with a per-structure
          fallback, CSV checkpoints with resume, one shard per GPU.
Physics : v2 spec (CLAUDE.md).
          exact integer charge neutrality, rule-based O3 -> P3 glide with a stacking-letter
          check, PAIRED vacancy patterns, composition-addressed RNG, meV/f.u. units.

Usage (one process per GPU, spawned by the launcher notebook):
    CUDA_VISIBLE_DEVICES=<k> python heo_worker.py --worker-id <k> --n-workers <N> --pass-no <1|2|3>

v4 (2026-09-10, CHANGE_SPEC_v4): PASS 3 adds the interior Na points x = 22/27 and 18/27 (tags
{ph}_x081_{k}, {ph}_x067_{k}); the checkpoint carries the relaxed cell matrix; a previous run can be
mounted read-only through HEO_BASE_WORKDIR so its energies, SQS cells and CIFs are reused, never rewritten.

The notebook also imports this module directly (anchors, Na reference, hull) so that every
structure in the run is built and relaxed by exactly one code path.
"""
import argparse, csv, glob, hashlib, itertools, json, math, os, time
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice, Composition

# ============================================================ CONFIG
MODE        = os.environ.get("HEO_MODE", "orb")            # orb | mace | mock
WORKDIR     = os.environ.get("HEO_WORKDIR", "/workspace/heo_v2")
SEED        = 42
N_NA, N_TM, N_O = 27, 27, 54
N_ATOMS     = N_NA + N_TM + N_O                            # 108
N_FU        = N_TM                                         # formula units NaTMO2 per cell
N_NA_REMOVE = int(os.environ.get("HEO_NNAREMOVE", 13))     # x = 14/27 = 0.5185
# Anchor-gate desodiation point (2026-09-03, user-approved): x = 20/27 = 0.741.  Zhao 2020 fully
# charges the HEO to x ~ 0.52 (110 mAh/g of ~229 mAh/g per Na) with >60% of capacity in the O3
# region, i.e. its O3->P3 transition sits near x ~ 0.71 -- so at x=0.52 BOTH anchors are P3 in
# experiment and their dE ordering there is unconstrained.  At x ~ 0.74 experiment separates them
# (HOST already gliding, Komaba 2012; HEO still O3), which is where a directional gate is justified.
N_NA_REMOVE_MILD = int(os.environ.get("HEO_NNAREMOVE_MILD", 7))
# v4 Na grid (CHANGE_SPEC_v4 B-1).  Tag stem -> number of Na removed from the 27-site cell.  pris/desod
# are the original screening points and keep their tags and RNG salts, so every existing checkpoint row
# stays bit-identical; x081/x067 are the two interior points computed by PASS 3 for the x* bracket.
X_STEMS   = {"pris": 0, "x081": 5, "x067": 9, "desod": N_NA_REMOVE}
X_OF      = {st: (N_NA - n) / N_NA for st, n in X_STEMS.items()}      # 1.0 / 0.8148 / 0.6667 / 0.5185
X4_STEMS  = ["x081", "x067"]
X_GRID_DESC = ["pris", "x081", "x067", "desod"]                       # descending x, the x* input order
X_LABEL   = {"pris": "100", "x081": "081", "x067": "067", "desod": "052"}
VAC_SALT  = {"desod": 100, "mild": 200, "x081": 300, "x067": 400}     # comp_rng salt base per stem (+k)
NA_ALLOWED = {N_NA - n for n in X_STEMS.values()}                      # invariant 8: {27, 22, 18, 14}
NA_MILD    = N_NA - N_NA_REMOVE_MILD                                   # 20: anchor/calibration diagnostic only
assert NA_ALLOWED == {27, 22, 18, 14}, f"invariant 8 grid drifted: {sorted(NA_ALLOWED)}"
SUPERCELL   = (3, 3, 1)
A_HEX, C_HEX, Z_O = 2.96, 15.95, 0.235                     # CLAUDE.md section 3
TM_CHARGE_TARGET  = -(N_NA*(+1) + N_O*(-2))                # +81
assert TM_CHARGE_TARGET == 81

N_VAC_SAMPLES = int(os.environ.get("HEO_NVAC", 5))
FMAX          = float(os.environ.get("HEO_FMAX", 0.05))
MAX_STEPS     = int(os.environ.get("HEO_MAXSTEPS", 300))
SQS_MC_STEPS  = int(os.environ.get("HEO_SQSSTEPS", 3000))
SQS_CUTOFFS   = [float(x) for x in os.environ.get("HEO_SQSCUTOFFS", "6.0,4.0").split(",")]  # v1-verified
CHUNK         = int(os.environ.get("HEO_CHUNK", 100))      # compositions per SQS+relax cycle
EHULL_EFF_CUT = float(os.environ.get("HEO_EHULLCUT", 0.10))  # on Ehull_eff (entropy-corrected)
SAVE_RELAXED  = os.environ.get("HEO_SAVE_RELAXED", "1") == "1"

ORB_MODEL  = os.environ.get("HEO_ORB_MODEL", "orb_v3_conservative_inf_mpa")   # mpa: same currency as MP-setting DFT
ORB_CKPT   = os.environ.get("HEO_ORB_CKPT", "")     # fine-tuned ORB weights (.pt/.ckpt); "" = stock
MACE_MODEL = os.environ.get("HEO_MACE_MODEL", "medium-mpa-0")
MODEL_TAG  = {"mock": "mock-v2", "orb": ORB_MODEL, "mace": "mace-" + MACE_MODEL}[MODE]
if MODE == "orb" and ORB_CKPT:
    # A distinct tag per checkpoint file: invariant 6 then refuses to mix stock and fine-tuned
    # energies in one WORKDIR, and a resumed run cannot silently reuse the other model's cache.
    MODEL_TAG = f"{ORB_MODEL}+ft-{os.path.splitext(os.path.basename(ORB_CKPT))[0]}"

K_B      = 8.617333262e-5      # eV/K
T_SYNTH  = 1173.0
DELTA_FLOOR_MEV_FU = 5.0

DOPANTS = ["B","Mg","Al","Si","Ti","Mn","Cu","Zn","Ga","Zr","Sn","Sb","Li","Co","Ca","V"]
OXI = {"Ni":(2,3), "Mn":(3,4), "Li":(1,), "Ca":(2,), "Mg":(2,), "Zn":(2,), "Cu":(2,),
       "B":(3,), "Al":(3,), "Ga":(3,), "Co":(2,3), "V":(3,4,5),
       "Ti":(4,), "Si":(4,), "Zr":(4,), "Sn":(4,), "Sb":(5,),
       "Fe":(3,)}   # anchor-only (Zhao 2020 HEO); single state -> COMP_ORDER asserts unaffected
DEFAULT_STATE      = {"Ni":2, "Mn":4, "Co":3, "V":4}
COMP_ORDER_DEFICIT = [("Ni",3), ("V",5)]      # total < 81 -> raise these
COMP_ORDER_EXCESS  = [("Mn",3), ("Co",2), ("V",3)]   # total > 81 -> lower these

# HOST: NaNi0.5Mn0.5O2 -> glides to P3 on desodiation (Komaba 2012). 13.5/13.5 cannot sit on
# 27 sites, so two variants bracket it and their average cancels the first-order charge error.
# HEO: NaNi.12Cu.12Mg.12Fe.15Co.15Mn.1Ti.1Sn.1Sb.04O2 (Zhao et al., Angew. Chem. 59, 264 (2020),
# doi:10.1002/anie.201912171) -> the O3->P3 transition is DELAYED (>60% of capacity in the O3
# region) vs HOST where it comes early.  x27 = Ni3.24 Cu3.24 Mg3.24 Fe4.05 Co4.05 Mn2.7 Ti2.7
# Sn2.7 Sb1.08 -> rounded {3,3,3,4,4,3,3,3,1} (27 sites, max error 0.3 site).  No half-integer
# problem here, so the two variants share the counts and differ only in comp_id: comp_rng() then
# gives them independent SQS + vacancy patterns, averaging 9-element configurational noise.
# The old Ti03 anchor (removed 2026-09-03): its premise "Ti0.3 retains O3 at x=0.52" contradicts
# Wang et al., Adv. Mater. 29, 1700210 (2017), which shows a reversible O3-P3 transition for the
# whole NaNi0.5Mn0.5-xTixO2 series -- so it must not gate the screening.
_HEO_COUNTS = {"Ni":3, "Cu":3, "Mg":3, "Fe":4, "Co":4, "Mn":3, "Ti":3, "Sn":3, "Sb":1}
assert sum(_HEO_COUNTS.values()) == N_TM
ANCHORS = {"HOST_v1": {"Ni":14, "Mn":13}, "HOST_v2": {"Ni":13, "Mn":14},
           "HEO_v1": dict(_HEO_COUNTS), "HEO_v2": dict(_HEO_COUNTS)}
ANCHOR_GROUPS = {"HOST": ["HOST_v1","HOST_v2"], "HEO": ["HEO_v1","HEO_v2"]}

for sub in ("", "/structures", "/structures/sqs", "/structures/relaxed",
            "/checkpoints", "/results", "/hull_cache"):
    os.makedirs(WORKDIR + sub, exist_ok=True)
SQS_DIR     = f"{WORKDIR}/structures/sqs"
RELAXED_DIR = f"{WORKDIR}/structures/relaxed"

# v4: a previous run's WORKDIR mounted READ-ONLY.  Its checkpoints, SQS cells and relaxed CIFs are
# consulted before anything is computed, and nothing is ever written under it (CHANGE_SPEC_v4 section 0).
BASE_WORKDIR = os.environ.get("HEO_BASE_WORKDIR", "").rstrip("/")
if BASE_WORKDIR:
    assert os.path.abspath(BASE_WORKDIR) != os.path.abspath(WORKDIR), \
        "HEO_BASE_WORKDIR must differ from HEO_WORKDIR: the base cache is read-only"
    assert os.path.isdir(BASE_WORKDIR), f"HEO_BASE_WORKDIR not found: {BASE_WORKDIR}"
BASE_SQS_DIR     = f"{BASE_WORKDIR}/structures/sqs" if BASE_WORKDIR else None
BASE_RELAXED_DIR = f"{BASE_WORKDIR}/structures/relaxed" if BASE_WORKDIR else None

# v4 adds "cell": the relaxed 3x3 lattice matrix (rows a, b, c, Angstrom, JSON) so volume decompositions
# never have to be recovered from CIFs.  Rows from older checkpoints simply carry NaN there.
CKPT_FIELDS = ["comp_id", "tag", "energy", "natoms", "volume", "P_order", "conv", "model", "cell"]

def _cell_json(M):
    return json.dumps([[round(float(x), 6) for x in row] for row in np.asarray(M, dtype=float)])

# ============================================================ composition-addressed RNG
def comp_rng(comp_id, salt=0):
    """Seeded by content, not by call order.  v1 threaded one RNG through every composition
    in shard order, so a vacancy pattern depended on how many compositions the worker had
    already handled -> changing the worker count or resuming changed the physics."""
    h = int(hashlib.md5(f"{comp_id}:{salt}:{SEED}".encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(h)

# ============================================================ STAGE 1: enumeration
def _default_state(e):
    return OXI[e][0] if len(OXI[e]) == 1 else DEFAULT_STATE[e]

# feasible_exact() is an INTERVAL test; assign_charge() is a one-directional greedy.  They are
# equivalent only while the compensation orders cover exactly the +-1 moves OXI allows away from
# DEFAULT_STATE and every element's states are contiguous integers.  Verified equivalent over all
# 38,108 enumerated count dicts; this guard fires if OXI / DEFAULT_STATE / COMP_ORDER_* drift apart.
_can_up   = {e for e in OXI if max(OXI[e]) > _default_state(e)}
_can_down = {e for e in OXI if min(OXI[e]) < _default_state(e)}
assert _can_up   == {e for e, _ in COMP_ORDER_DEFICIT}, f"COMP_ORDER_DEFICIT != {_can_up}"
assert _can_down == {e for e, _ in COMP_ORDER_EXCESS},  f"COMP_ORDER_EXCESS != {_can_down}"
assert all(v2 - v1 == 1 for e in OXI for v1, v2 in zip(OXI[e], OXI[e][1:])), "OXI states must be contiguous"

def sconf_kB(cnt):
    t = sum(cnt.values())
    return -sum((n/t)*math.log(n/t) for n in cnt.values() if n > 0)

def charge_bounds(cnt):
    return (sum(n*min(OXI[e]) for e, n in cnt.items()),
            sum(n*max(OXI[e]) for e, n in cnt.items()))

def feasible_exact(cnt):
    lo, hi = charge_bounds(cnt)
    return lo <= TM_CHARGE_TARGET <= hi

def feasible_legacy(cnt, tol=2.7):
    lo, hi = charge_bounds(cnt)
    return lo <= TM_CHARGE_TARGET + tol and hi >= TM_CHARGE_TARGET - tol

def assign_charge(cnt, target=TM_CHARGE_TARGET):
    """Minimum-disproportionation integer assignment from DEFAULT_STATE, or None."""
    state = {e: _default_state(e) for e in cnt}
    need  = target - sum(cnt[e]*state[e] for e in cnt)
    shifts = {"n_Ni3":0, "n_Mn3":0, "n_Co2":0, "n_V3":0, "n_V5":0}
    for el, new_state in (COMP_ORDER_DEFICIT if need > 0 else COMP_ORDER_EXCESS):
        if need == 0: break
        if el not in cnt or new_state not in OXI[el]: continue
        per_site = new_state - state[el]
        if per_site == 0 or (per_site > 0) != (need > 0): continue
        k = min(cnt[el], abs(need)); need -= k*per_site
        shifts[f"n_{el}{new_state}"] = shifts.get(f"n_{el}{new_state}", 0) + k
    return None if need != 0 else shifts

def counts_A(dset):
    c = {"Ni":6, "Mn":6}
    for d in dset: c[d] = c.get(d, 0) + 3
    return c

def counts_B_variants(dset):
    for short_idx in range(5):
        c = {"Ni":4, "Mn":4}
        for i, d in enumerate(dset): c[d] = c.get(d, 0) + (3 if i == short_idx else 4)
        yield dset[short_idx], c

def counts_C_variants(dset):
    """RESCUE lattice 4-3-3-3-2, used only where neither A nor B admits an integer-neutral
    solution.  593 of the 595 sets A+B reject are solvable at some other ratio and this is the
    closest lattice to A, so it recovers them without leaving the entropy maximum."""
    for i_short, i_long in itertools.permutations(range(5), 2):
        c = {"Ni":6, "Mn":6}
        for i, d in enumerate(dset):
            c[d] = c.get(d, 0) + (2 if i == i_short else 4 if i == i_long else 3)
        yield (dset[i_short], dset[i_long]), c

def _row(comp_id, scheme, tag, c, s, b_short="", c_short="", c_long=""):
    sh = assign_charge(c); assert sh is not None, (comp_id, c)   # exact feasibility <=> greedy success
    r = dict(comp_id=comp_id, scheme=scheme, dopants=tag, B_short_dopant=b_short,
             C_short_dopant=c_short, C_long_dopant=c_long,
             site_counts=json.dumps(c), n_species=len(c), sconf_kB=round(s, 4),
             sconf_per_atom_kB=round(s*N_TM/N_ATOMS, 4))
    r.update(sh); return r

def enumerate_compositions(use_exact=True):
    """Returns (DataFrame, n_legacy).  Deterministic; md5 0fc86c95929b0c775d7ed0505cafc76a."""
    feas = feasible_exact if use_exact else feasible_legacy
    rows, n_legacy = [], 0
    for dset in itertools.combinations(DOPANTS, 5):
        tag = "-".join(dset); hit = False
        for scheme, cands in (("A", [("", counts_A(dset))]), ("B", list(counts_B_variants(dset)))):
            best = None
            for short_d, c in cands:
                if feas(c):
                    s = sconf_kB(c)
                    # tie-break among equal-entropy variants: first in DOPANTS order (CLAUDE.md S1-1)
                    if best is None or s > best[0]: best = (s, c, short_d)
            n_legacy += any(feasible_legacy(c) for _, c in cands)
            if best is None: continue
            hit = True; s, c, short_d = best
            rows.append(_row(f"{scheme}_{tag}", scheme, tag, c, s, b_short=short_d))
        if hit: continue
        # scheme C keeps EVERY feasible variant: here they are all tied at max sconf, so there is
        # no arbitrary choice and no chemistry is silently discarded.
        for (c_short, c_long), c in counts_C_variants(dset):
            if not feas(c): continue
            rows.append(_row(f"C_{tag}_{c_short}v{c_long}", "C", tag, c, sconf_kB(c),
                             c_short=c_short, c_long=c_long))
    comps = pd.DataFrame(rows).sort_values("comp_id").reset_index(drop=True)
    comps["formula_cell"] = comps.site_counts.apply(
        lambda s: "Na27" + "".join(f"{e}{n}" for e, n in json.loads(s).items()) + "O54")
    assert comps.comp_id.is_unique, "duplicate comp_id"
    assert not comps.comp_id.str.contains("__").any(), "comp_id must not contain '__' (CIF separator)"
    assert (comps.site_counts.apply(lambda s: sum(json.loads(s).values())) == N_TM).all(), "site counts != 27"
    return comps, n_legacy

# ============================================================ STAGE 2a: templates and the glide
_prim = Structure.from_spacegroup("R-3m", Lattice.hexagonal(A_HEX, C_HEX),
                                  ["Na", "Ni", "O"], [[0,0,0],[0,0,0.5],[0,0,Z_O]])
O3_TEMPLATE = _prim.copy(); O3_TEMPLATE.make_supercell(list(SUPERCELL))
assert len(O3_TEMPLATE) == N_ATOMS, len(O3_TEMPLATE)
TM_IDX = [i for i, s in enumerate(O3_TEMPLATE) if s.specie.symbol == "Ni"]
NA_IDX = [i for i, s in enumerate(O3_TEMPLATE) if s.specie.symbol == "Na"]
assert len(TM_IDX) == N_TM and len(NA_IDX) == N_NA
_SC = SUPERCELL[0]
_LETTERS = {"A": (0.0, 0.0), "B": (1/3, 2/3), "C": (2/3, 1/3)}

def letter(frac_xy):
    """supercell fractional (x,y) -> primitive (3x,3y) mod 1 -> nearest of A/B/C."""
    x, y = (frac_xy[0]*_SC) % 1.0, (frac_xy[1]*_SC) % 1.0
    best, bd = None, 9
    for k, (lx, ly) in _LETTERS.items():
        dx, dy = abs(x-lx), abs(y-ly); dx, dy = min(dx, 1-dx), min(dy, 1-dy)
        if dx+dy < bd: best, bd = k, dx+dy
    return best

def layers_by_z(struct, symbol, tol=0.02):
    zs = sorted({round(s.frac_coords[2] % 1.0, 3) for s in struct if s.specie.symbol == symbol})
    groups = []
    for z in zs:
        if groups and abs(z - groups[-1][-1]) < tol: groups[-1].append(z)
        else: groups.append([z])
    out = []
    for g in groups:
        idx = [i for i, s in enumerate(struct) if s.specie.symbol == symbol
               and min(abs(s.frac_coords[2] % 1.0 - z) for z in g) < tol]
        out.append((float(np.mean(g)), idx))
    return out

def stacking_report(struct):
    """Na-gap types ('O' or 'P') in z order.  Different letters across a gap -> octahedral,
    same letters -> prismatic.  Raises if an O layer is mixed-letter or a TM gap is eclipsed."""
    O_layers = layers_by_z(struct, "O")
    na_z = [z for z, _ in layers_by_z(struct, "Na")]
    letters = []
    for z, idx in O_layers:
        ls = {letter(struct[i].frac_coords[:2]) for i in idx}
        assert len(ls) == 1, f"mixed letters in O layer z={z:.3f}: {ls}"
        letters.append(ls.pop())
    n = len(O_layers); gaps = []
    for k in range(n):
        z1, z2 = O_layers[k][0], O_layers[(k+1) % n][0] + (1.0 if k == n-1 else 0.0)
        mid = (z1+z2)/2 % 1.0
        has_na = any(abs((mid - z + 0.5) % 1.0 - 0.5) < 0.06 for z in na_z)
        same = letters[k] == letters[(k+1) % n]
        if has_na: gaps.append("P" if same else "O")
        else: assert not same, f"TM gap eclipsed at z~{mid:.3f} (TM must be octahedral)"
    return gaps, letters

def na_collision_free(struct):
    """In a P gap Na must not sit on the O letter, or it would lie on the O-O line."""
    O_layers = layers_by_z(struct, "O")
    o_map = {round(z,3): {letter(struct[i].frac_coords[:2]) for i in idx} for z, idx in O_layers}
    for z_na, idx in layers_by_z(struct, "Na"):
        below = max([z for z in o_map if z < z_na] or [max(o_map)])
        for i in idx:
            if letter(struct[i].frac_coords[:2]) in o_map[below]: return False
    return True

def _unit_index(z, tm_zs):
    """unit k = slab k (TM layer + flanking O) + the Na layer above it."""
    return int(((z - (min(tm_zs) - 0.12)) % 1.0) // (1.0/3))

def o3_to_p3(o3, glide=None):
    """O3 = CABCAB (every Na gap octahedral) -> translate unit k by k*v -> P3 = CAABBC.
    The stacking-letter check, not a distance heuristic, decides that the result is P3."""
    tm_zs = sorted({round(s.frac_coords[2] % 1.0, 3) for s in o3 if s.specie.symbol not in ("Na", "O")})
    tm_zs = [tm_zs[0], tm_zs[len(tm_zs)//3], tm_zs[2*len(tm_zs)//3]] if len(tm_zs) > 3 else tm_zs
    for v in ([glide] if glide else [(2/9, 1/9), (1/9, 2/9)]):
        p3 = o3.copy()
        for i, s in enumerate(p3):
            k = _unit_index(s.frac_coords[2] % 1.0, tm_zs)
            f = s.frac_coords.copy(); f[0] = (f[0] + k*v[0]) % 1.0; f[1] = (f[1] + k*v[1]) % 1.0
            p3.replace(i, s.specie, coords=f, coords_are_cartesian=False)
        gaps, _ = stacking_report(p3)
        if all(g == "P" for g in gaps) and na_collision_free(p3):
            p3.glide_vector = v
            return p3
    raise RuntimeError("no glide vector produced a valid P3 stacking")

def prismatic_order(struct, r_cut=3.2, n_tri=3):
    """Relaxation-robust O/P read-out.  Uses only the in-plane azimuths of the O triangles above
    and below each Na, so it survives buckling, cell strain and the periodic seam that break a
    z-slicing letter classifier:
        u = <exp(3i*theta)> per triangle;  P = Re(u_top * conj(u_bot)) / |u_top||u_bot|
        P -> +1 eclipsed = trigonal PRISM (P);  P -> -1 staggered 60 deg = OCTAHEDRON (O)."""
    c_hat = struct.lattice.matrix[2] / np.linalg.norm(struct.lattice.matrix[2])
    out = []
    for site in struct:
        if site.specie.symbol != "Na": continue
        up, dn = [], []
        for n in struct.get_neighbors(site, r_cut):
            if n.specie.symbol != "O": continue
            d = n.coords - site.coords
            dz = float(np.dot(d, c_hat))
            (up if dz > 0 else dn).append((np.linalg.norm(d), d - dz*c_hat))
        if len(up) < n_tri or len(dn) < n_tri: continue      # under-coordinated Na next to vacancies
        def _u(group):
            g = sorted(group, key=lambda t: t[0])[:n_tri]
            return np.mean([np.exp(3j*np.arctan2(p[1], p[0])) for _, p in g])
        ut, ub = _u(up), _u(dn)
        if abs(ut) < 1e-6 or abs(ub) < 1e-6: continue
        out.append(float(np.real(ut*np.conj(ub))/(abs(ut)*abs(ub))))
    return (float(np.mean(out)) if out else float("nan")), out

def _validate_templates(verbose=True):
    gaps_o3, letters_o3 = stacking_report(O3_TEMPLATE)
    p3 = o3_to_p3(O3_TEMPLATE)
    gaps_p3, letters_p3 = stacking_report(p3)
    assert all(g == "O" for g in gaps_o3) and all(g == "P" for g in gaps_p3)
    if verbose:
        print("O3 O-layer letters:", "".join(letters_o3), "| Na gaps:", gaps_o3)
        print("P3 O-layer letters:", "".join(letters_p3), "| Na gaps:", gaps_p3,
              "| glide:", p3.glide_vector)
    for name, st, want in (("O3", O3_TEMPLATE, -1.0), ("P3", p3, +1.0)):
        dmin = min(st.get_distance(i, j) for i in NA_IDX[:3]
                   for j in range(len(st)) if st[j].specie.symbol == "O")
        assert dmin > 2.0, f"{name}: min Na-O {dmin:.3f} A"
        po, per = prismatic_order(st)
        assert abs(po - want) < 0.05, f"{name} order parameter {po:+.3f} != {want:+.0f}"
        if verbose: print(f"{name}: min Na-O = {dmin:.3f} A | prismatic order P = {po:+.4f} "
                          f"over {len(per)} Na (expect {want:+.0f})")
    return p3

P3_TEMPLATE = _validate_templates(verbose=False)
GLIDE = P3_TEMPLATE.glide_vector

# ============================================================ STAGE 2b: TM placement (SQS)
def _sqs_worker(args):
    """(comp_id, counts) -> writes structures/sqs/<comp_id>.json.  multiprocessing-safe.
    The icet result is re-mapped onto the template site order, so NA_IDX/TM_IDX stay valid
    whatever order icet returns (CLAUDE.md section 6 item 3 needs no assumption this way)."""
    comp_id, counts = args
    path = f"{SQS_DIR}/{comp_id}.json"
    if os.path.exists(sqs_path(comp_id)): return comp_id, "cached"
    if MODE == "mock":                      # no icet in the smoke path; keeps mock runs deterministic and fast
        st = place_tm_random(counts, comp_rng(comp_id, salt=1))
        st.to(fmt="json", filename=path + ".tmp"); os.replace(path + ".tmp", path)
        return comp_id, "random(mock)"
    how = "sqs"
    try:
        from ase import Atoms  # noqa: F401  (icet needs ase present)
        from pymatgen.io.ase import AseAtomsAdaptor
        from icet import ClusterSpace
        from icet.tools.structure_generation import generate_sqs_from_supercells
        prim_ase = AseAtomsAdaptor.get_atoms(_prim)
        elems = sorted(counts)
        # one list per SITE of the conventional cell (giving 3 lists raises ValueError)
        chem = [[s.specie.symbol] if s.specie.symbol in ("Na", "O") else elems for s in _prim]
        cs = ClusterSpace(prim_ase, cutoffs=SQS_CUTOFFS, chemical_symbols=chem)
        sc = AseAtomsAdaptor.get_atoms(O3_TEMPLATE)
        conc = {e: counts[e]/N_TM for e in elems}
        sqs = generate_sqs_from_supercells(cs, supercells=[sc], target_concentrations=conc,
                                           n_steps=SQS_MC_STEPS,
                                           random_seed=int(comp_rng(comp_id, salt=1).integers(1e9)))
        raw = AseAtomsAdaptor.get_structure(sqs)
        st = _remap_to_template(raw, counts)
    except Exception as e:
        how = f"random_fallback({type(e).__name__}: {str(e)[:80]})"
        st = place_tm_random(counts, comp_rng(comp_id, salt=1))
    st.to(fmt="json", filename=path + ".tmp")     # atomic: a torn cache file must never be read back
    os.replace(path + ".tmp", path)
    return comp_id, how

def _remap_to_template(raw, counts):
    """Read the species icet put on each site and rewrite them onto O3_TEMPLATE's site order."""
    assert len(raw) == N_ATOMS, f"icet returned {len(raw)} atoms"
    st = O3_TEMPLATE.copy()
    used = set()
    for i in TM_IDX:
        f = st[i].frac_coords
        j = min((k for k in range(len(raw)) if k not in used),
                key=lambda k: raw.lattice.get_distance_and_image(f, raw[k].frac_coords)[0])
        used.add(j); st.replace(i, raw[j].specie)
    got = {}
    for i in TM_IDX: got[st[i].specie.symbol] = got.get(st[i].specie.symbol, 0) + 1
    assert got == counts, f"remap lost stoichiometry: {got} != {counts}"
    return st

def place_tm_random(counts, rng):
    species = [e for e, n in sorted(counts.items()) for _ in range(n)]
    assert len(species) == len(TM_IDX), counts
    rng.shuffle(species)
    st = O3_TEMPLATE.copy()
    for i, e in zip(TM_IDX, species): st.replace(i, e)
    return st

def sqs_path(comp_id):
    """Own SQS cell first, then the read-only base; the own path is returned when neither exists
    (that is where _sqs_worker will write)."""
    own = f"{SQS_DIR}/{comp_id}.json"
    if os.path.exists(own) or BASE_SQS_DIR is None: return own
    base = f"{BASE_SQS_DIR}/{comp_id}.json"
    return base if os.path.exists(base) else own

def pregen_sqs(rows, n_proc=None):
    """Parallel SQS pre-generation for a chunk of composition rows.  CPU-bound (~12 s each);
    running it serially inside the relax loop is what starves the GPUs."""
    import multiprocessing as mp
    todo = [(r.comp_id, json.loads(r.site_counts)) for r in rows
            if not os.path.exists(sqs_path(r.comp_id))]
    if not todo: return 0
    n_fallback = 0
    with mp.Pool(n_proc or max(1, mp.cpu_count() - 1)) as p:
        for cid, how in p.imap_unordered(_sqs_worker, todo):
            if "fallback" in how:
                n_fallback += 1
                if n_fallback <= 5: print("  [SQS-FALLBACK]", cid, how, flush=True)
    if n_fallback: print(f"  [SQS-FALLBACK] {n_fallback}/{len(todo)} in this chunk", flush=True)
    return n_fallback

def pristine_O3(comp_id, counts):
    path = sqs_path(comp_id)
    if not os.path.exists(path): _sqs_worker((comp_id, counts)); path = sqs_path(comp_id)
    st = Structure.from_file(path)
    assert all(st[i].specie.symbol == "Na" for i in NA_IDX), f"{comp_id}: site order broken"
    return st

# ============================================================ STAGE 2c: desodiation
def na_layers_idx(struct):
    """Periodic-safe Na layer grouping (merges z~0 and z~1)."""
    na = [(i, s.frac_coords[2] % 1.0) for i, s in enumerate(struct) if s.specie.symbol == "Na"]
    zs = sorted({round(z, 3) for _, z in na}); groups = []
    for z in zs:
        if not groups or abs(z - groups[-1][-1]) > 0.05: groups.append([z])
        else: groups[-1].append(z)
    if len(groups) > 1 and abs(groups[0][0] + 1 - groups[-1][-1]) < 0.05:
        groups[0] = groups.pop() + groups[0]
    return [[i for i, z in na if any(min(abs(z-gz), 1-abs(z-gz)) < 0.03 for gz in g)] for g in groups]

def vacancy_pattern(struct, rng, n_remove=N_NA_REMOVE):
    """Site indices of the Na to remove: split n_remove as evenly as possible over the layers
    (13 -> [5,4,4]), then greedy max-min in-plane separation inside each layer.  Returns indices,
    NOT a structure, so the SAME pattern can be applied to O3 and P3 (paired comparison)."""
    layers = na_layers_idx(struct); nl = len(layers)
    kper = [n_remove // nl + (1 if i < n_remove % nl else 0) for i in range(nl)]
    rng.shuffle(kper)
    remove = []
    for lay, k in zip(layers, kper):
        chosen = [lay[int(rng.integers(len(lay)))]]
        while len(chosen) < k:
            best, bd = None, -1.0
            for cand in lay:
                if cand in chosen: continue
                d = min(struct.get_distance(cand, c) for c in chosen)
                if d > bd: best, bd = cand, d
            chosen.append(best)
        remove += chosen
    assert len(remove) == n_remove and len(set(remove)) == n_remove
    return sorted(remove)

def parse_tag(tag):
    """'O3_x081_3' -> ('O3', 'x081', 3, 5)   'P3_pris' -> ('P3', 'pris', None, 0)   'O3_mild_1' -> (.., 7).
    None for tags off the composition grid (HULL entries, the Na reference).  The stem names the Na
    count, the last field the vacancy sample: together with comp_id and the model column this is the
    full cache key (invariant 10)."""
    parts = tag.split("_")
    if parts[0] not in ("O3", "P3"): return None
    if len(parts) == 2 and parts[1] == "pris": return parts[0], "pris", None, 0
    if len(parts) == 3 and parts[2].isdigit():
        if parts[1] in X_STEMS: return parts[0], parts[1], int(parts[2]), X_STEMS[parts[1]]
        if parts[1] == "mild":  return parts[0], "mild", int(parts[2]), N_NA_REMOVE_MILD
    return None

def _check_na(comp_id, tag, st):
    """Invariant 8 (v4): every structure on the grid carries Na in {27, 22, 18, 14}.  The x = 0.74
    'mild' point (Na 20) is a diagnostic used by the anchors and the calibration only."""
    n_na = sum(1 for s in st if s.specie.symbol == "Na")
    pt = parse_tag(tag)
    assert pt is not None, f"invariant 8: unparseable tag {tag}"
    assert n_na in NA_ALLOWED or (pt[1] == "mild" and n_na == NA_MILD), \
        f"invariant 8: {comp_id}/{tag} has {n_na} Na; allowed {sorted(NA_ALLOWED)} (+{NA_MILD} for mild only)"
    assert n_na == N_NA - pt[3], f"invariant 8: {comp_id}/{tag} has {n_na} Na but the tag says {N_NA - pt[3]}"
    return comp_id, tag, st

def build_items(comp_id, counts, which):
    """(comp_id, tag, structure) for O3/P3 x pristine/desodiated.
    o3_to_p3 preserves site indices, so one vacancy pattern serves both phases.
    which: 'pris' | 'desod' | 'all' (= pris + desod) | 'mild' | 'x4' (= x081 + x067) | 'x081' | 'x067'."""
    o3 = pristine_O3(comp_id, counts)
    p3 = o3_to_p3(o3, glide=GLIDE)
    if which in ("pris", "all"):
        yield _check_na(comp_id, "O3_pris", o3)
        yield _check_na(comp_id, "P3_pris", p3)
    if which in ("desod", "all"):
        for k in range(N_VAC_SAMPLES):
            pat = vacancy_pattern(o3, comp_rng(comp_id, salt=VAC_SALT["desod"]+k))   # same pattern for O3 and P3
            for ph, st in (("O3", o3), ("P3", p3)):
                d = st.copy(); d.remove_sites(pat)
                yield _check_na(comp_id, f"{ph}_desod_{k}", d)
    if which == "mild":
        # Anchor-gate point x = 20/27 (see N_NA_REMOVE_MILD).  Tag must NOT contain "desod":
        # metrics_from() collects desod tags by substring, and mixing n=7 with n=13 there would
        # corrupt dE_desod/sigma_vac.  salt=200+k keeps the patterns independent of the 100+k set.
        for k in range(N_VAC_SAMPLES):
            pat = vacancy_pattern(o3, comp_rng(comp_id, salt=VAC_SALT["mild"]+k), n_remove=N_NA_REMOVE_MILD)
            for ph, st in (("O3", o3), ("P3", p3)):
                d = st.copy(); d.remove_sites(pat)
                yield _check_na(comp_id, f"{ph}_mild_{k}", d)
    if which == "x4" or which in X4_STEMS:
        # v4 interior points.  Each x is an independent vacancy sample set (salt base per stem, +k),
        # O3 and P3 sharing every pattern exactly as at x = 0.52; nothing is nested across x.
        for stem in (X4_STEMS if which == "x4" else [which]):
            for k in range(N_VAC_SAMPLES):
                pat = vacancy_pattern(o3, comp_rng(comp_id, salt=VAC_SALT[stem]+k), n_remove=X_STEMS[stem])
                for ph, st in (("O3", o3), ("P3", p3)):
                    d = st.copy(); d.remove_sites(pat)
                    yield _check_na(comp_id, f"{ph}_{stem}_{k}", d)

def tags_for(which):
    t = ["O3_pris", "P3_pris"] if which in ("pris", "all") else []
    if which in ("desod", "all"):
        t += [f"{ph}_desod_{k}" for k in range(N_VAC_SAMPLES) for ph in ("O3", "P3")]
    if which == "mild":
        t += [f"{ph}_mild_{k}" for k in range(N_VAC_SAMPLES) for ph in ("O3", "P3")]
    if which == "x4" or which in X4_STEMS:
        for stem in (X4_STEMS if which == "x4" else [which]):
            t += [f"{ph}_{stem}_{k}" for k in range(N_VAC_SAMPLES) for ph in ("O3", "P3")]
    return t

# ============================================================ STAGE 3: relax engine
DTYPE = None; _TS = None; _MODEL = None; _CALC = None
if MODE in ("orb", "mace"):
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32 if MODE == "orb" else torch.float64
else:
    DEVICE = "cpu"

def get_model():
    global _TS, _MODEL, _CALC
    if _MODEL is None:
        import torch_sim as ts
        if MODE == "orb":
            from orb_models.forcefield import pretrained
            from torch_sim.models.orb import OrbModel
            loaded = getattr(pretrained, ORB_MODEL)(device=DEVICE)
            raw, adapter = loaded if isinstance(loaded, tuple) else (loaded, None)
            if ORB_CKPT:                       # fine-tuned weights over the stock architecture
                assert os.path.exists(ORB_CKPT), f"HEO_ORB_CKPT not found: {ORB_CKPT}"
                sd = torch.load(ORB_CKPT, map_location="cpu")
                for key in ("state_dict", "model_state_dict", "model"):   # lightning/trainer wrappers
                    if isinstance(sd, dict) and key in sd and hasattr(sd[key], "items"):
                        sd = sd[key]; break
                try:
                    raw.load_state_dict(sd)                               # strict: surface any mismatch
                except RuntimeError:
                    raw.load_state_dict({k.split("model.", 1)[-1]: v for k, v in sd.items()})
                print(f"[model] fine-tuned weights loaded: {ORB_CKPT} -> tag {MODEL_TAG}", flush=True)
            _MODEL = (OrbModel(model=raw, atoms_adapter=adapter, device=DEVICE, dtype=DTYPE)
                      if adapter is not None else OrbModel(model=raw, device=DEVICE, dtype=DTYPE))
        else:
            from mace.calculators import mace_mp
            from torch_sim.models.mace import MaceModel
            _CALC = mace_mp(model=MACE_MODEL, device=DEVICE, default_dtype="float64")
            _MODEL = MaceModel(model=_CALC.models[0], device=DEVICE, dtype=DTYPE)
        _TS = ts
    return _TS, _MODEL

_E_NA_MOCK = -1.31
def _mock_params(comp_id, counts):
    rng = comp_rng(comp_id, salt=999)
    stab = sum(counts.get(e, 0) for e in ("Ti", "B", "Ca", "Li"))    # O3-stabilising proxies
    return dict(E0=-5.9*N_ATOMS + rng.normal(0, 0.3),
                dE_pris=(25 + rng.normal(0, 8)) * N_FU / 1000,
                dE_desod=(-30 + 12*stab + rng.normal(0, 10)) * N_FU / 1000,
                V_avg=2.8 + rng.normal(0, 0.2),
                vol0=14.1*N_ATOMS + rng.normal(0, 8),
                Ehull=max(0.0, 0.09 + rng.normal(0, 0.03)))

def _mock_cell(struct, tag, V, vol0):
    """Mock 'relaxation' of the lattice only: rescale the input cell to the mock volume V, anisotropically
    for desodiated cells (in-plane shrink, c expand; the prismatic P3 gap expands more) so that dV_report
    sees a non-trivial dA / dh_perp / von Mises split.  Energies are untouched -> the section-5 mock
    expectation values are unchanged."""
    st = struct.copy(); V0 = st.lattice.volume
    iso = (vol0 / V0) ** (1.0/3.0)                       # template -> mock pristine cell, isotropic
    if tag.endswith("_pris"):
        sa = sc = iso * (V / vol0) ** (1.0/3.0)
    else:                                                # relative to the pristine mock cell
        sc_rel = 1.02 if tag.startswith("O3") else 1.05
        sa_rel = math.sqrt(V / vol0 / sc_rel)
        sa, sc = iso * sa_rel, iso * sc_rel
    M = np.array(st.lattice.matrix, dtype=float); M[0] *= sa; M[1] *= sa; M[2] *= sc
    return Structure(Lattice(M), st.species, st.frac_coords)

def relax_mock(comp_id, tag, struct, counts):
    p = _mock_params(comp_id, counts)
    rng = comp_rng(comp_id, salt=int(hashlib.md5(tag.encode()).hexdigest()[:6], 16))
    pt = parse_tag(tag)
    n = pt[3] if pt is not None else N_NA_REMOVE          # 13 branch below is bit-identical to v3
    if tag == "O3_pris":   E, V = p["E0"], p["vol0"]
    elif tag == "P3_pris": E, V = p["E0"] + p["dE_pris"], p["vol0"]*1.003
    else:
        # dE(P3-O3) interpolates linearly in removed Na; the exact branch at n == N_NA_REMOVE keeps
        # the section-5 mock expectation values bit-identical (a+(b-a) can differ from b by one ulp).
        dE_ph = p["dE_desod"] if n == N_NA_REMOVE else \
                p["dE_pris"] + (p["dE_desod"] - p["dE_pris"]) * n / N_NA_REMOVE
        E_desod_O3 = p["E0"] - n*_E_NA_MOCK + n*p["V_avg"]
        extra = abs(rng.normal(0, 0.12))                     # vacancy-arrangement penalty
        if tag.startswith("O3"): E, V = E_desod_O3 + extra, p["vol0"]*(1 - 0.004 + rng.normal(0, 0.002))
        else:                    E, V = E_desod_O3 + dE_ph + extra, p["vol0"]*(1 + 0.03 + rng.normal(0, 0.005))
    return float(E), float(V), True, _mock_cell(struct, tag, float(V), p["vol0"])

def _p_order(st):
    try:
        v, _ = prismatic_order(st); return float(v)
    except Exception:
        return float("nan")

def relax_batch(items, ckpt, done, counts_of=None, save_cif=None):
    """items: list of (comp_id, tag, Structure).  Appends one CSV row per structure.
    A failed batch falls back to per-structure relaxation and drops only the structures that
    themselves fail, so one bad cell can never kill a 20-hour shard."""
    todo = [(cid, tag, st) for cid, tag, st in items if (cid, tag) not in done]
    if not todo: return
    save_cif = SAVE_RELAXED if save_cif is None else save_cif
    rows = []

    if MODE == "mock":
        for cid, tag, st in todo:
            E, V, conv, rst = relax_mock(cid, tag, st, (counts_of or {}).get(cid, {}))
            rows.append(dict(comp_id=cid, tag=tag, energy=E, natoms=len(rst), volume=V,
                             P_order=_p_order(rst), conv=conv, model=MODEL_TAG,
                             cell=_cell_json(rst.lattice.matrix)))
    else:
        import gc
        from pymatgen.io.ase import AseAtomsAdaptor
        from torch_sim.optimizers.cell_filters import CellFilter
        ts, model = get_model()
        atoms = [AseAtomsAdaptor.get_atoms(st) for *_, st in todo]

        def _opt(atom_list, use_autobatch):
            state = ts.io.atoms_to_state(atom_list, device=DEVICE, dtype=DTYPE)
            return ts.optimize(system=state, model=model, optimizer=ts.optimizers.Optimizer.fire,
                               convergence_fn=ts.generate_force_convergence_fn(force_tol=FMAX),
                               max_steps=MAX_STEPS, init_kwargs={"cell_filter": CellFilter.frechet},
                               autobatcher=use_autobatch)
        def _free_gpu():
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

        # 2026-09-04 fix (first ORB run: 6 chunks = 1,200 cells silently skipped).  The per-structure
        # fallback used to run INSIDE the `except` block, where `err.__traceback__` still pinned the
        # failed batch's GPU tensors -> every retry died with "Failed to allocate 20 bytes" and the
        # whole 200-cell chunk was dropped.  The fallback now runs after the exception object is
        # released, with an explicit gc + empty_cache, and again after each per-structure failure.
        # The batch itself failed with "Failed to allocate 20 bytes" 6 times in 71 chunks: the
        # autobatcher packs states up to its memory estimate, and blocks still reserved from the
        # previous chunk left no headroom.  Start every chunk from a clean allocator.
        _free_gpu()
        batch_err = None
        try:
            final = _opt(atoms, True)
            E = [float(x) for x in final.energy.detach().cpu().numpy()]
            rel = ts.io.state_to_atoms(final)
            # the batch engine may regroup structures internally; atom counts are a cheap
            # fingerprint that catches a reordering before it silently mislabels energies.
            assert [len(a) for a in rel] == [len(a) for a in atoms], \
                "torch-sim returned structures in a different order -> energies would be mislabelled"
        except AssertionError:
            raise
        except (ValueError, RuntimeError) as err:
            batch_err = f"{type(err).__name__}: {str(err)[:150]}"
        if batch_err is not None:
            final = None; _free_gpu()          # the exception (and its traceback) is out of scope here
            print(f"[batch fallback] {batch_err} -> per-structure relax", flush=True)
            E, rel, ok = [], [], []
            for j, at in enumerate(atoms):
                try:
                    f1 = _opt([at], False)
                    E.append(float(f1.energy.detach().cpu().numpy().ravel()[0]))
                    rel.extend(ts.io.state_to_atoms(f1)); ok.append(j)
                    del f1
                except Exception as err2:
                    cid_j, tag_j, _ = todo[j]
                    print(f"[skip structure] {cid_j}/{tag_j}: {type(err2).__name__}: {str(err2)[:100]}", flush=True)
                    err2 = None; _free_gpu()
            todo = [todo[j] for j in ok]
        for (cid, tag, _), en, at in zip(todo, E, rel):
            st = AseAtomsAdaptor.get_structure(at)
            rows.append(dict(comp_id=cid, tag=tag, energy=float(en), natoms=len(at),
                             volume=float(at.get_volume()), P_order=_p_order(st),
                             conv=True, model=MODEL_TAG, cell=_cell_json(at.cell[:])))
            if save_cif and cid not in ("HULL", "REF"):
                st.to(filename=f"{RELAXED_DIR}/{cid}__{tag}.cif")
    if MODE == "mock" and save_cif:
        for (cid, tag, st) in todo:
            if cid not in ("HULL", "REF"): st.to(filename=f"{RELAXED_DIR}/{cid}__{tag}.cif")

    hdr = not os.path.exists(ckpt)
    if not hdr:
        with open(ckpt) as f: have = f.readline().strip().split(",")
        # Appending v4 rows (one extra column) to a v3 checkpoint would produce a ragged CSV that
        # pandas cannot read back.  Old caches are consumed through HEO_BASE_WORKDIR instead.
        assert have == CKPT_FIELDS, (f"{ckpt}: header {have} != {CKPT_FIELDS} -> this WORKDIR holds a "
                                     "pre-v4 checkpoint; use a fresh HEO_WORKDIR and mount the old one as HEO_BASE_WORKDIR")
    with open(ckpt, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CKPT_FIELDS)
        if hdr: w.writeheader()
        w.writerows(rows)
    done.update((r["comp_id"], r["tag"]) for r in rows)

def ckpt_files():
    """Base (read-only) checkpoints first, then our own: on a duplicate key the base row wins, so a
    value computed in the previous run is reused rather than recomputed."""
    fps = sorted(glob.glob(f"{BASE_WORKDIR}/checkpoints/energies*.csv")) if BASE_WORKDIR else []
    return fps + sorted(glob.glob(f"{WORKDIR}/checkpoints/energies*.csv"))

def load_done_all():
    done = set()
    for fp in ckpt_files():
        try:
            df = pd.read_csv(fp); done.update(zip(df.comp_id, df.tag))
        except Exception:
            pass
    return done

def _check_tag_natoms(df):
    """Invariant 10 (v4): the cache key is (comp_id, phase, Na count, sample, model).  The Na count is
    encoded in the tag stem, so every grid row must satisfy natoms == 108 - n_removed(tag); a row that
    does not is a mislabelled energy and must stop the run before it is aggregated."""
    grid = df[~df.comp_id.isin(["HULL", "REF"])]
    if not len(grid): return
    parsed = grid.tag.map(parse_tag)
    unk = grid[parsed.isna()]
    assert unk.empty, f"invariant 10: tags outside the grid in the checkpoints: {sorted(unk.tag.unique())[:5]}"
    expect = np.array([N_ATOMS - p[3] for p in parsed], dtype=int)
    bad = grid[grid.natoms.astype(int).values != expect]
    assert bad.empty, ("invariant 10: tag/natoms mismatch (cache key corruption), e.g. "
                       f"{bad[['comp_id', 'tag', 'natoms']].head(3).to_dict('records')}")

def merged_energies(check_model=True):
    fps = ckpt_files()
    if not fps: return pd.DataFrame(columns=CKPT_FIELDS)
    df = pd.concat([pd.read_csv(fp) for fp in fps], ignore_index=True).drop_duplicates(["comp_id","tag"])
    if "cell" not in df.columns: df["cell"] = np.nan
    if check_model and "model" in df.columns and len(df):
        models = set(df.model.dropna().unique())
        # invariant 6, enforced HERE rather than at the end of the run: mixing currencies is a
        # reason to stop before spending GPU-hours, not after.
        assert models == {MODEL_TAG}, f"invariant 6: mixed currencies in checkpoints {models} != {MODEL_TAG}"
        _check_tag_natoms(df)
    return df

# ============================================================ STAGE 4: metrics
def metrics_from(recs):
    """recs: dict tag -> {'energy','volume',...} in eV/cell and A^3.  Returns meV/f.u. quantities.
    dE_desod is the PAIRED minimum min_k(E_P3,k - E_O3,k): O3 and P3 share a vacancy pattern for
    each k, so the vacancy-arrangement energy cancels to first order.  sigma_vac is the spread of
    that same paired difference, for the same reason."""
    m = {}
    Eo, Ep = recs["O3_pris"]["energy"], recs["P3_pris"]["energy"]
    m["E_pris_O3"], m["E_pris_P3"] = Eo, Ep
    m["Vol_pris_O3"] = recs["O3_pris"]["volume"]
    m["dE_pris"] = (Ep - Eo) * 1000 / N_FU
    ks = sorted({int(t.rsplit("_", 1)[1]) for t in recs if "desod" in t})
    ks = [k for k in ks if f"O3_desod_{k}" in recs and f"P3_desod_{k}" in recs]
    if ks:
        o3s = [recs[f"O3_desod_{k}"] for k in ks]
        p3s = [recs[f"P3_desod_{k}"] for k in ks]
        d_paired = [(p["energy"] - o["energy"]) * 1000 / N_FU for o, p in zip(o3s, p3s)]
        kbest = int(np.argmin(d_paired))
        m["dE_desod"]   = float(d_paired[kbest])
        m["k_best"]     = ks[kbest]
        m["sigma_vac"]  = float(np.std(d_paired))
        m["ddE"]        = m["dE_desod"] - m["dE_pris"]
        bo, bp = o3s[kbest], p3s[kbest]
        m["E_desod_O3"], m["E_desod_P3"] = bo["energy"], bp["energy"]
        # V_avg is a property of the O3 -> O3 discharge path, so it uses the lowest O3 cell.
        eo_min = min(r["energy"] for r in o3s)
        m["V_avg"] = (eo_min + N_NA_REMOVE*E_NA_GLOBAL[0] - Eo) / N_NA_REMOVE
        v0 = m["Vol_pris_O3"]
        m["dV_O3"]    = 100*abs(bo["volume"] - v0)/v0
        m["dV_trans"] = 100*abs(bp["volume"] - v0)/v0
        m["n_unconverged"] = sum(not bool(r.get("conv", True)) for r in recs.values())
        # a P3 cell that glided back to O3 gives dE_desod ~ 0, read as "O3 retained" -- the one
        # failure mode that biases toward the answer we hope for, so count it explicitly.
        m["n_phase_flip"] = sum(1 for t, r in recs.items()
                                if np.isfinite(r.get("P_order", np.nan))
                                and (r["P_order"] > 0) != t.startswith("P3"))
    return m

def judge_phase(dE, delta):
    if dE >  delta: return "O3"
    if dE < -delta: return "P3"
    return "AMB"

# ---------------------------------------------------------------- v4: per-x metrics and the x* bracket
def metrics_x(recs, stem):
    """Paired O3/P3 statistics at one grid point (stem in X_STEMS), the metrics_from conventions
    applied to that stem: dE = min_k(E_P3,k - E_O3,k) in meV/f.u., sigma = std of the paired
    differences.  {} when the stem has no complete O3/P3 pair.  metrics_from itself is untouched,
    so every x = 0.52 number stays bit-identical to v3."""
    ks = sorted(k for k in range(N_VAC_SAMPLES) if f"O3_{stem}_{k}" in recs and f"P3_{stem}_{k}" in recs)
    if not ks: return {}
    o3s = [recs[f"O3_{stem}_{k}"] for k in ks]; p3s = [recs[f"P3_{stem}_{k}"] for k in ks]
    d = [(p["energy"] - o["energy"]) * 1000 / N_FU for o, p in zip(o3s, p3s)]
    kb = int(np.argmin(d))
    return {f"dE_{stem}": float(d[kb]), f"sigma_vac_{stem}": float(np.std(d)), f"k_best_{stem}": ks[kb],
            f"n_pairs_{stem}": len(ks),
            f"E_O3_{stem}": o3s[kb]["energy"], f"E_P3_{stem}": p3s[kb]["energy"],
            f"Vol_O3_{stem}": o3s[kb]["volume"], f"Vol_P3_{stem}": p3s[kb]["volume"]}

def x_star_bracket(verdicts, xs=None):
    """CHANGE_SPEC_v4 B-4.  verdicts: O3/AMB/P3 in DESCENDING x order (x = 1 first).
    Returns x_star_class, x_star_lo, x_star_hi, x_star_mid.
      suppressed_full : O3 at every point (x* < 0.52, below the window)
      transition      : first departure is P3 and no O3 returns  -> x* in (x[i], x[i-1])
      amb_boundary    : first departure is AMB and no O3 returns -> same bracket, grey verdict
      nonmonotonic    : O3 reappears after a departure -> review flag, not an error
      incomplete      : a point is missing;  no_O3_start: not O3 even at x = 1 (cannot bracket)"""
    xs = list(xs) if xs is not None else [X_OF[s] for s in X_GRID_DESC]
    nan = float("nan")
    if any(v not in ("O3", "AMB", "P3") for v in verdicts):
        return dict(x_star_class="incomplete", x_star_lo=nan, x_star_hi=nan, x_star_mid=nan)
    i = next((j for j, v in enumerate(verdicts) if v != "O3"), None)
    if i is None:
        return dict(x_star_class="suppressed_full", x_star_lo=nan, x_star_hi=xs[-1], x_star_mid=nan)
    if i == 0:
        return dict(x_star_class="no_O3_start", x_star_lo=nan, x_star_hi=nan, x_star_mid=nan)
    tail = verdicts[i:]
    if verdicts[i] == "P3":
        cls = "transition" if all(v in ("P3", "AMB") for v in tail) else "nonmonotonic"
    else:
        cls = "amb_boundary" if all(v != "O3" for v in tail) else "nonmonotonic"
    if cls == "nonmonotonic":
        return dict(x_star_class=cls, x_star_lo=nan, x_star_hi=nan, x_star_mid=nan)
    lo, hi = xs[i], xs[i-1]
    return dict(x_star_class=cls, x_star_lo=lo, x_star_hi=hi, x_star_mid=(lo + hi) / 2.0)

def e_nat_of(verdict, E_O3, E_P3):
    """Natural-phase energy at one x: the verdict's phase, or the lower of the two when AMB."""
    if verdict == "O3": return E_O3, "O3"
    if verdict == "P3": return E_P3, "P3"
    return (E_O3, "O3(min)") if E_O3 <= E_P3 else (E_P3, "P3(min)")

V_SEGMENTS = [("pris", "x081"), ("x081", "x067"), ("x067", "desod")]

def v_segments(E_nat, E_Na):
    """Step-wise voltages V_seg(x_i -> x_{i+1}) = [E_nat(x_{i+1}) + dn*E_Na - E_nat(x_i)] / dn  [V],
    dn = 5, 4, 4.  Discharge sign convention as V_avg (CLAUDE.md section 3): always positive
    (invariant 9, asserted by the caller so the evidence gets written first)."""
    out = {}
    for a, b in V_SEGMENTS:
        dn = X_STEMS[b] - X_STEMS[a]
        key = f"V_seg_{X_LABEL[a]}_{X_LABEL[b]}"
        out[key] = ((E_nat[b] + dn * E_Na - E_nat[a]) / dn) if (a in E_nat and b in E_nat) else float("nan")
    return out

def formation_x(E_nat):
    """x-direction formation energy against the (x=1, x=0.52) tie line, meV/f.u.:
    Ef(x) = E_nat(x) - [E_nat(1)*(x - x052) + E_nat(x052)*(1 - x)] / (1 - x052).
    Ef < 0: a stable intermediate (sloped, solid-solution-like region); Ef >= 0: two-phase tendency.
    convexity_flag from the two signs: convex (both < 0) / concave (both >= 0) / mixed / n/a."""
    x1, x0 = X_OF["pris"], X_OF["desod"]
    out = {}
    for s in X4_STEMS:
        if all(k in E_nat for k in ("pris", "desod", s)):
            x = X_OF[s]
            tie = (E_nat["pris"] * (x - x0) + E_nat["desod"] * (1 - x)) / (1 - x0)
            out[f"Ef_{s}"] = (E_nat[s] - tie) * 1000 / N_FU
        else:
            out[f"Ef_{s}"] = float("nan")
    e = [out[f"Ef_{s}"] for s in X4_STEMS]
    if any(np.isnan(v) for v in e): out["convexity_flag"] = "n/a"
    elif all(v < 0 for v in e):     out["convexity_flag"] = "convex"
    elif all(v >= 0 for v in e):    out["convexity_flag"] = "concave"
    else:                           out["convexity_flag"] = "mixed"
    return out

E_NA_GLOBAL = [float("nan")]        # set by na_reference(); metrics_from reads it
def set_E_Na(v): E_NA_GLOBAL[0] = float(v)

def na_reference(ckpt, done):
    """Na bcc in the SAME currency.  3x3x3 (54 atoms) rather than a 2-atom cell."""
    if MODE == "mock":
        set_E_Na(_E_NA_MOCK); return _E_NA_MOCK
    na = Structure(Lattice.cubic(4.29), ["Na","Na"], [[0,0,0],[.5,.5,.5]]) * (3,3,3)
    relax_batch([("REF", "Na_bcc", na)], ckpt, done, save_cif=False)
    df = merged_energies()
    row = df[(df.comp_id == "REF") & (df.tag == "Na_bcc")].iloc[0]
    e = float(row.energy) / int(row.natoms)
    assert -1.6 < e < -1.0, f"Na reference {e:.3f} eV/atom far from PBE (~-1.31) -> wrong currency?"
    set_E_Na(e); return e

def recs_by_comp(df, comp_ids=None):
    """energy table -> {comp_id: {tag: record}}"""
    if comp_ids is not None:
        df = df[df.comp_id.isin(set(comp_ids))]
    out = {}
    for r in df.itertuples():
        c = getattr(r, "cell", None)
        out.setdefault(r.comp_id, {})[r.tag] = dict(
            energy=float(r.energy), volume=float(r.volume), natoms=int(r.natoms),
            P_order=float(getattr(r, "P_order", float("nan"))), conv=bool(getattr(r, "conv", True)),
            cell=c if isinstance(c, str) and c.startswith("[") else None)
    return out

# ============================================================ STAGE 4b (v4): cell geometry
# Why not 2*(da/a) + dc/c: (i) that assumes the angles stay hexagonal, but the Frechet cell filter
# relaxes all six lattice degrees of freedom, and a 1-2 degree drift is already a %-scale error;
# (ii) it is a first-order expansion, which wobbles at the 13/27-desodiation strains; (iii) a scalar
# dV hides the real mechanics -- in-plane contraction and interlayer expansion cancel (cell-11 result
# 2026-09-04, section 7-5).  Exact quantities from the lattice matrix instead, all rotation-invariant.
def cell_metrics(cell):
    """cell: 3x3 array-like, rows = a, b, c (Angstrom).
    Returns exact volume, in-plane area, tilt-immune perpendicular height."""
    cell = np.asarray(cell, dtype=float)
    a, b, _c = cell
    V = abs(np.linalg.det(cell))
    A_par = np.linalg.norm(np.cross(a, b))
    h_perp = V / A_par
    return V, A_par, h_perp

def cell_angles(cell):
    """(alpha, beta, gamma) in degrees from the row vectors."""
    a, b, c = np.asarray(cell, dtype=float)
    ang = lambda u, v: math.degrees(math.acos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    return ang(b, c), ang(a, c), ang(a, b)

def dV_report(cell_pris, cell_desod, n_slabs=3):
    """Volume change decomposition between paired cells.
    Requires lattice-vector correspondence (guaranteed: P3 built from O3 by glide,
    desod built from pris by Na removal in the same cell).
    F = deformation gradient (columns convention), E = Green-Lagrange strain; tr(E) ~ volumetric part,
    von Mises of the deviator = shape change.  F^T F only, so any rigid rotation between the two
    stored cells (CIF round trip) drops out."""
    Vp, Ap, hp = cell_metrics(cell_pris)
    Vd, Ad, hd = cell_metrics(cell_desod)
    Lp = np.asarray(cell_pris, dtype=float).T
    Ld = np.asarray(cell_desod, dtype=float).T
    F = Ld @ np.linalg.inv(Lp)
    E = 0.5 * (F.T @ F - np.eye(3))
    vol_part = np.trace(E)
    dev = E - np.eye(3) * vol_part / 3.0
    vm = float(np.sqrt(2.0 / 3.0 * np.sum(dev * dev)))
    return {
        "dV_pct":       100.0 * (Vd - Vp) / Vp,
        "dA_pct":       100.0 * (Ad - Ap) / Ap,
        "dh_perp_pct":  100.0 * (hd - hp) / hp,
        "d_inter_pris": hp / n_slabs,
        "vm_strain":    vm,
    }

def relaxed_cif_path(comp_id, tag):
    """Own relaxed CIF first, then the read-only base; None if neither exists."""
    for d in (RELAXED_DIR, BASE_RELAXED_DIR):
        if d is None: continue
        fp = f"{d}/{comp_id}__{tag}.cif"
        if os.path.exists(fp): return fp
    return None

def cell_of(comp_id, tag, rec=None):
    """Relaxed cell matrix: the checkpoint 'cell' column (v4 rows) or, for pre-v4 rows, the CIF that
    HEO_SAVE_RELAXED=1 wrote (own dir, then base).  None when neither is available."""
    c = (rec or {}).get("cell")
    if isinstance(c, str) and c.startswith("["):
        return np.asarray(json.loads(c), dtype=float)
    fp = relaxed_cif_path(comp_id, tag)
    if fp is None: return None
    return np.asarray(Structure.from_file(fp).lattice.matrix, dtype=float)

def _cell_job(args):
    cid, tag = args
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # pymatgen CIF parser chatter x 10k files
            c = cell_of(cid, tag)
        return cid, tag, (None if c is None else c.tolist())
    except Exception:
        return cid, tag, None

def cells_bulk(keys, recs_of=None, n_proc=None):
    """{(comp_id, tag): 3x3 array or None} for many keys; checkpoint cells are taken directly,
    CIF fallbacks are parsed in a process pool (tens of thousands of CIFs otherwise take an hour)."""
    out, todo = {}, []
    for cid, tag in dict.fromkeys(keys):                 # de-duplicated, order kept
        rec = (recs_of or {}).get(cid, {}).get(tag, {})
        c = rec.get("cell") if rec else None
        if isinstance(c, str) and c.startswith("["): out[(cid, tag)] = np.asarray(json.loads(c), dtype=float)
        else: todo.append((cid, tag))
    if todo:
        import multiprocessing as mp
        n_proc = n_proc or max(1, min(mp.cpu_count() - 1, 32))
        if len(todo) < 50 or n_proc == 1:
            res = [_cell_job(t) for t in todo]
        else:
            with mp.Pool(n_proc) as p: res = list(p.imap_unordered(_cell_job, todo, chunksize=64))
        for cid, tag, c in res:
            out[(cid, tag)] = None if c is None else np.asarray(c, dtype=float)
    return out

# ============================================================ worker main
def _pass2_survivors():
    """Ehull_eff < cut AND pristine phase == O3, read from the notebook's phase-1 table."""
    fp = f"{WORKDIR}/results/results_phase1.csv"
    assert os.path.exists(fp), "results_phase1.csv missing -> run the hull cell before PASS 2"
    r = pd.read_csv(fp)
    need = {"comp_id", "Ehull_eff", "dE_pris"}
    assert need <= set(r.columns), f"results_phase1.csv missing {need - set(r.columns)}"
    keep = r[(r.Ehull_eff < EHULL_EFF_CUT) & (r.dE_pris > DELTA_FLOOR_MEV_FU)]
    return set(keep.comp_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--pass-no", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--limit", type=int, default=0, help="cap #compositions (smoke tests); 0 = all")
    args = ap.parse_args()

    comps = pd.read_csv(f"{WORKDIR}/results/compositions_master.csv")
    if args.limit:
        comps = comps.sample(args.limit, random_state=SEED).sort_values("comp_id").reset_index(drop=True)
    done = load_done_all()
    merged_energies()                     # invariant 6 up front, before any GPU time is spent
    ckpt = f"{WORKDIR}/checkpoints/energies_w{args.worker_id}.csv"

    if args.pass_no == 1:
        which = "pris"; pool = comps
    elif args.pass_no == 2:
        which = "desod"; surv = _pass2_survivors()
        pool = comps[comps.comp_id.isin(surv)]
        print(f"[w{args.worker_id}] PASS-1 filter: {len(pool)} / {len(comps)} survive", flush=True)
    else:
        # v4 PASS 3: the interior x points for the compositions the notebook selected (HEO_X4_SCOPE).
        which = "x4"; fp = f"{WORKDIR}/results/x4_targets.csv"
        assert os.path.exists(fp), "results/x4_targets.csv missing -> run the PASS 3 target cell first"
        tg = set(pd.read_csv(fp).comp_id)
        pool = comps[comps.comp_id.isin(tg)]
        print(f"[w{args.worker_id}] PASS-3 targets: {len(pool)} / {len(comps)} "
              f"(HEO_X4_SCOPE={os.environ.get('HEO_X4_SCOPE', '?')})", flush=True)
    tags = tags_for(which)
    todo = [row for _, row in pool.iterrows()
            if any((row.comp_id, t) not in done for t in tags)]
    shard = todo[args.worker_id::args.n_workers]
    print(f"[w{args.worker_id}] PASS{args.pass_no} shard: {len(shard)} comps "
          f"(model {MODEL_TAG}, engine {'mock' if MODE=='mock' else 'torch-sim'})", flush=True)

    t0 = time.time()
    for i in range(0, len(shard), CHUNK):
        chunk = shard[i:i+CHUNK]
        pregen_sqs(chunk)                                     # all cores, then the GPU
        counts_of = {r.comp_id: json.loads(r.site_counts) for r in chunk}
        items = [it for r in chunk for it in build_items(r.comp_id, counts_of[r.comp_id], which)]
        relax_batch(items, ckpt, done, counts_of=counts_of)
        n = min(i + CHUNK, len(shard)); rate = n / max(time.time() - t0, 1)
        print(f"[w{args.worker_id}] {n}/{len(shard)} | {rate*3600:.0f} comps/h | "
              f"ETA {(len(shard)-n)/max(rate,1e-9)/3600:.1f} h", flush=True)
    print(f"[w{args.worker_id}] DONE in {(time.time()-t0)/3600:.2f} h", flush=True)

if __name__ == "__main__":
    main()
