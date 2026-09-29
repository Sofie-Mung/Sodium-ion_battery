"""heo_worker.py - HEO Na layered-cathode MLIP screening worker (RunPod multi-GPU).

Engine  : v1, proven end-to-end on 7,279 compositions.
          multiprocessing SQS pre-generation, torch-sim batch relax with a per-structure
          fallback, CSV checkpoints with resume, one shard per GPU.
Physics : v2 spec (CLAUDE.md).
          exact integer charge neutrality, rule-based O3 -> P3 glide with a stacking-letter
          check, PAIRED vacancy patterns, composition-addressed RNG, meV/f.u. units.

Usage (one process per GPU, spawned by the launcher notebook):
    CUDA_VISIBLE_DEVICES=<k> python heo_worker.py --worker-id <k> --n-workers <N> --pass-no <1|2>

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

CKPT_FIELDS = ["comp_id", "tag", "energy", "natoms", "volume", "P_order", "conv", "model"]

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
    if os.path.exists(path): return comp_id, "cached"
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

def pregen_sqs(rows, n_proc=None):
    """Parallel SQS pre-generation for a chunk of composition rows.  CPU-bound (~12 s each);
    running it serially inside the relax loop is what starves the GPUs."""
    import multiprocessing as mp
    todo = [(r.comp_id, json.loads(r.site_counts)) for r in rows
            if not os.path.exists(f"{SQS_DIR}/{r.comp_id}.json")]
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
    path = f"{SQS_DIR}/{comp_id}.json"
    if not os.path.exists(path): _sqs_worker((comp_id, counts))
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

def build_items(comp_id, counts, which):
    """(comp_id, tag, structure) for O3/P3 x pristine/desodiated.
    o3_to_p3 preserves site indices, so one vacancy pattern serves both phases."""
    o3 = pristine_O3(comp_id, counts)
    p3 = o3_to_p3(o3, glide=GLIDE)
    if which in ("pris", "all"):
        yield comp_id, "O3_pris", o3
        yield comp_id, "P3_pris", p3
    if which in ("desod", "all"):
        for k in range(N_VAC_SAMPLES):
            pat = vacancy_pattern(o3, comp_rng(comp_id, salt=100+k))   # same pattern for O3 and P3
            for ph, st in (("O3", o3), ("P3", p3)):
                d = st.copy(); d.remove_sites(pat)
                yield comp_id, f"{ph}_desod_{k}", d
    if which == "mild":
        # Anchor-gate point x = 20/27 (see N_NA_REMOVE_MILD).  Tag must NOT contain "desod":
        # metrics_from() collects desod tags by substring, and mixing n=7 with n=13 there would
        # corrupt dE_desod/sigma_vac.  salt=200+k keeps the patterns independent of the 100+k set.
        for k in range(N_VAC_SAMPLES):
            pat = vacancy_pattern(o3, comp_rng(comp_id, salt=200+k), n_remove=N_NA_REMOVE_MILD)
            for ph, st in (("O3", o3), ("P3", p3)):
                d = st.copy(); d.remove_sites(pat)
                yield comp_id, f"{ph}_mild_{k}", d

def tags_for(which):
    t = ["O3_pris", "P3_pris"] if which in ("pris", "all") else []
    if which in ("desod", "all"):
        t += [f"{ph}_desod_{k}" for k in range(N_VAC_SAMPLES) for ph in ("O3", "P3")]
    if which == "mild":
        t += [f"{ph}_mild_{k}" for k in range(N_VAC_SAMPLES) for ph in ("O3", "P3")]
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

def relax_mock(comp_id, tag, struct, counts):
    p = _mock_params(comp_id, counts)
    rng = comp_rng(comp_id, salt=int(hashlib.md5(tag.encode()).hexdigest()[:6], 16))
    n = N_NA_REMOVE_MILD if "_mild_" in tag else N_NA_REMOVE
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
    return float(E), float(V), True, struct

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
                             P_order=_p_order(rst), conv=conv, model=MODEL_TAG))
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
                             conv=True, model=MODEL_TAG))
            if save_cif and cid not in ("HULL", "REF"):
                st.to(filename=f"{RELAXED_DIR}/{cid}__{tag}.cif")
    if MODE == "mock" and save_cif:
        for (cid, tag, st) in todo:
            if cid not in ("HULL", "REF"): st.to(filename=f"{RELAXED_DIR}/{cid}__{tag}.cif")

    hdr = not os.path.exists(ckpt)
    with open(ckpt, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CKPT_FIELDS)
        if hdr: w.writeheader()
        w.writerows(rows)
    done.update((r["comp_id"], r["tag"]) for r in rows)

def load_done_all():
    done = set()
    for fp in glob.glob(f"{WORKDIR}/checkpoints/energies*.csv"):
        try:
            df = pd.read_csv(fp); done.update(zip(df.comp_id, df.tag))
        except Exception:
            pass
    return done

def merged_energies(check_model=True):
    fps = glob.glob(f"{WORKDIR}/checkpoints/energies*.csv")
    if not fps: return pd.DataFrame(columns=CKPT_FIELDS)
    df = pd.concat([pd.read_csv(fp) for fp in fps], ignore_index=True).drop_duplicates(["comp_id","tag"])
    if check_model and "model" in df.columns and len(df):
        models = set(df.model.dropna().unique())
        # invariant 6, enforced HERE rather than at the end of the run: mixing currencies is a
        # reason to stop before spending GPU-hours, not after.
        assert models == {MODEL_TAG}, f"invariant 6: mixed currencies in checkpoints {models} != {MODEL_TAG}"
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
        out.setdefault(r.comp_id, {})[r.tag] = dict(
            energy=float(r.energy), volume=float(r.volume), natoms=int(r.natoms),
            P_order=float(getattr(r, "P_order", float("nan"))), conv=bool(getattr(r, "conv", True)))
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
    ap.add_argument("--pass-no", type=int, default=1, choices=(1, 2))
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
    else:
        which = "desod"; surv = _pass2_survivors()
        pool = comps[comps.comp_id.isin(surv)]
        print(f"[w{args.worker_id}] PASS-1 filter: {len(pool)} / {len(comps)} survive", flush=True)
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
