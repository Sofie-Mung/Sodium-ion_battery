"""dft_common.py - shared constants and pure functions for the DFT verification pipeline.

Safe to import on the lab server and the GPU machine: stdlib + numpy + pymatgen only,
no side effects, no heo_worker import (heo_worker creates $HEO_WORKDIR dirs at import).

Functions marked "lifted" are verbatim copies from heo_worker.py (source line cited).
build_manifest.py cross-checks one of them against heo_worker at runtime on RunPod,
so a silent divergence between the two copies aborts the bundle build.
"""
import hashlib, json, math, os
import numpy as np

# ============================================================ grid constants (heo_worker.py:29-51)
N_NA, N_TM, N_O = 27, 27, 54
N_ATOMS = N_NA + N_TM + N_O            # 108
N_FU    = N_TM                         # formula units NaTMO2 per cell
X_STEMS = {"pris": 0, "x081": 5, "x067": 9, "desod": 13}
X_OF    = {st: (N_NA - n) / N_NA for st, n in X_STEMS.items()}
X_LABEL = {"pris": "100", "x081": "081", "x067": "067", "desod": "052"}
X_GRID_DESC = ["pris", "x081", "x067", "desod"]
NA_ALLOWED  = {N_NA - n for n in X_STEMS.values()}          # {27, 22, 18, 14}
assert NA_ALLOWED == {27, 22, 18, 14}

# x_tag (spec naming, dft_inputs/... path component) <-> tag stem (cache naming)
XTAG_OF_STEM = {"pris": "x100", "x081": "x081", "x067": "x067", "desod": "x052"}
STEM_OF_XTAG = {v: k for k, v in XTAG_OF_STEM.items()}
NA_OF_XTAG   = {XTAG_OF_STEM[s]: N_NA - n for s, n in X_STEMS.items()}

DELTA_FLOOR_MEV_FU = 5.0               # heo_worker.py:76 - MLIP delta floor, anchor GATE-2 threshold

_SC = 3                                # SUPERCELL = (3, 3, 1) (heo_worker.py:51,248)
_LETTERS = {"A": (0.0, 0.0), "B": (1/3, 2/3), "C": (2/3, 1/3)}

# ============================================================ manifest schema
MANIFEST_COLUMNS = [
    "row_id", "set", "comp_id", "phase", "x_tag", "na_count", "k_best",
    "struct_hash", "mlip_E", "mlip_dE", "priority", "status", "fw_id", "task_id",
    "note", "vasp_mode", "attempts", "magmom_json", "acct_json", "mlip_cell", "poscar",
]
STATUS_VALUES = {"pending", "queued", "running", "done", "fizzled", "manual"}
PRIORITY = {"pilot": 30, "E_Na": 25, "anchor": 20, "x100": 10, "x052": 10, "x081": 5, "x067": 5}

# ============================================================ INCAR overrides (spec section 5)
# MP defaults (MPRelaxSet / MPStaticSet) are the currency: LDAU, U values, ENCUT, POTCAR are
# NEVER overridden (spec section 8-2).  Only the keys below are touched.
INCAR_OVERRIDES_RELAX = {
    "ISPIN": 2, "LORBIT": 11,          # site-projected magnetization - mandatory (spec 8-3)
    "ISIF": 3,
    "EDIFF": 1e-5, "EDIFFG": -0.02, "NSW": 99, "IBRION": 2,
    "LREAL": "Auto", "NCORE": 4, "LWAVE": False, "LCHARG": False,
}
INCAR_OVERRIDES_STATIC = {
    "ISPIN": 2, "LORBIT": 11,
    "EDIFF": 1e-5,
    "LREAL": "Auto", "NCORE": 4, "LWAVE": False, "LCHARG": False,
}
_FORBIDDEN_INCAR = {"LDAU", "LDAUU", "LDAUJ", "LDAUL", "LDAUTYPE", "ENCUT", "ALGO"}
for _d in (INCAR_OVERRIDES_RELAX, INCAR_OVERRIDES_STATIC):
    assert not (_FORBIDDEN_INCAR & set(_d)), \
        f"spec 8-2: currency keys must not be overridden: {_FORBIDDEN_INCAR & set(_d)}"

# ============================================================ MAGMOM seeding (spec section 5)
# Initial moments per oxidation state (mu_B).  The accounting fixes only the COUNT of each state,
# not which site carries it, so each element gets its count-weighted AVERAGE and SCF decides;
# the audit reads the converged site moments instead.
MAGMOM_BY_STATE = {
    ("Ni", 2): 2.0, ("Ni", 3): 1.0, ("Ni", 4): 0.0,
    ("Mn", 4): 3.0, ("Mn", 3): 4.0,
    ("Co", 3): 0.0, ("Co", 2): 3.0,            # Co3+ low-spin assumed
    ("V", 3): 2.0, ("V", 4): 1.0, ("V", 5): 0.0,
    ("Fe", 3): 5.0, ("Cu", 2): 1.0,
}
# Charge accounting (mirrors heo_worker.py OXI/DEFAULT_STATE/COMP_ORDER, :79-85).  Used only for
# the two anchors, whose accounting columns are not in the results csv; top-30 rows take
# n_Ni3/n_Mn3/n_Co2/n_V3/n_V5 straight from the csv.
OXI = {"Ni": (2, 3), "Mn": (3, 4), "Li": (1,), "Ca": (2,), "Mg": (2,), "Zn": (2,), "Cu": (2,),
       "B": (3,), "Al": (3,), "Ga": (3,), "Co": (2, 3), "V": (3, 4, 5),
       "Ti": (4,), "Si": (4,), "Zr": (4,), "Sn": (4,), "Sb": (5,), "Fe": (3,)}
DEFAULT_STATE = {"Ni": 2, "Mn": 4, "Co": 3, "V": 4}
TM_CHARGE_TARGET = 81


def charge_account(counts):
    """counts: {element: n_sites on the 27 TM sites} -> {n_Ni3, n_Mn3, n_Co2, n_V3, n_V5}.
    Same greedy as heo_worker: deficit raised via Ni->3+ then V->5+, excess lowered via
    Mn->3+, Co->2+, V->3+.  Asserts exact neutrality (TM total = +81)."""
    assert sum(counts.values()) == N_TM, counts
    state_n = {}                                        # (el, state) -> count
    for el, n in counts.items():
        st = OXI[el][0] if len(OXI[el]) == 1 else DEFAULT_STATE[el]
        state_n[(el, st)] = state_n.get((el, st), 0) + n
    total = sum(st * n for (el, st), n in state_n.items())
    def _move(el, s_from, s_to, k):
        state_n[(el, s_from)] -= k
        state_n[(el, s_to)] = state_n.get((el, s_to), 0) + k
    if total < TM_CHARGE_TARGET:
        for el, s_to in [("Ni", 3), ("V", 5)]:
            s_from = DEFAULT_STATE.get(el)
            avail = state_n.get((el, s_from), 0)
            k = min(avail, (TM_CHARGE_TARGET - total) // (s_to - s_from))
            if k > 0:
                _move(el, s_from, s_to, k); total += k * (s_to - s_from)
            if total == TM_CHARGE_TARGET: break
    elif total > TM_CHARGE_TARGET:
        for el, s_to in [("Mn", 3), ("Co", 2), ("V", 3)]:
            s_from = DEFAULT_STATE.get(el)
            avail = state_n.get((el, s_from), 0)
            k = min(avail, (total - TM_CHARGE_TARGET) // (s_from - s_to))
            if k > 0:
                _move(el, s_from, s_to, k); total -= k * (s_from - s_to)
            if total == TM_CHARGE_TARGET: break
    assert total == TM_CHARGE_TARGET, f"charge_account: could not neutralize {counts} (total {total})"
    return {"n_Ni3": state_n.get(("Ni", 3), 0), "n_Mn3": state_n.get(("Mn", 3), 0),
            "n_Co2": state_n.get(("Co", 2), 0), "n_V3": state_n.get(("V", 3), 0),
            "n_V5": state_n.get(("V", 5), 0)}


def magmom_for(counts, acct):
    """Per-element MAGMOM seed (mu_B): count-weighted average over the element's states.
    counts: {el: n_sites}; acct: {n_Ni3, n_Mn3, n_Co2, n_V3, n_V5}.  Na/O and elements
    without an entry in MAGMOM_BY_STATE seed at 0."""
    out = {"Na": 0.0, "O": 0.0}
    for el, n in counts.items():
        if n == 0: continue
        if el == "Ni":
            n3 = int(acct.get("n_Ni3", 0)); parts = [(2, n - n3), (3, n3)]
        elif el == "Mn":
            n3 = int(acct.get("n_Mn3", 0)); parts = [(4, n - n3), (3, n3)]
        elif el == "Co":
            n2 = int(acct.get("n_Co2", 0)); parts = [(3, n - n2), (2, n2)]
        elif el == "V":
            n3, n5 = int(acct.get("n_V3", 0)), int(acct.get("n_V5", 0))
            parts = [(4, n - n3 - n5), (3, n3), (5, n5)]
        else:
            st = OXI[el][0] if len(OXI.get(el, ())) == 1 else DEFAULT_STATE.get(el)
            parts = [(st, n)]
        mu = sum(MAGMOM_BY_STATE.get((el, s), 0.0) * k for s, k in parts) / n
        out[el] = round(float(mu), 4)
    return out


# ============================================================ tag helpers (heo_worker.py:495)
def tag_for(phase, x_tag, k_best):
    """(phase, x_tag, k) -> cache tag.  pris carries no sample index."""
    stem = STEM_OF_XTAG[x_tag]
    if stem == "pris": return f"{phase}_pris"
    return f"{phase}_{stem}_{int(k_best)}"


def parse_tag(tag):
    """lifted: heo_worker.py:495.  'O3_x081_3' -> ('O3','x081',3,5); 'P3_pris' -> ('P3','pris',None,0)."""
    parts = tag.split("_")
    if parts[0] not in ("O3", "P3"): return None
    if len(parts) == 2 and parts[1] == "pris": return parts[0], "pris", None, 0
    if len(parts) == 3 and parts[2].isdigit() and parts[1] in X_STEMS:
        return parts[0], parts[1], int(parts[2]), X_STEMS[parts[1]]
    return None


# ============================================================ stacking classifiers
def letter(frac_xy):
    """lifted: heo_worker.py:251."""
    x, y = (frac_xy[0]*_SC) % 1.0, (frac_xy[1]*_SC) % 1.0
    best, bd = None, 9
    for k, (lx, ly) in _LETTERS.items():
        dx, dy = abs(x-lx), abs(y-ly); dx, dy = min(dx, 1-dx), min(dy, 1-dy)
        if dx+dy < bd: best, bd = k, dx+dy
    return best


def layers_by_z(struct, symbol, tol=0.02):
    """lifted: heo_worker.py:260."""
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
    """lifted: heo_worker.py:273.  Raises on relaxed/buckled cells (mixed-letter O layers);
    use prismatic_order for relaxed structures and keep this for letter-level diagnostics."""
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


def prismatic_order(struct, r_cut=3.2, n_tri=3):
    """lifted: heo_worker.py:324.  Relaxation-robust O/P read-out from O-triangle azimuths:
    P -> +1 trigonal prism (P3), P -> -1 octahedron (O3)."""
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
        if len(up) < n_tri or len(dn) < n_tri: continue
        def _u(group):
            g = sorted(group, key=lambda t: t[0])[:n_tri]
            return np.mean([np.exp(3j*np.arctan2(p[1], p[0])) for _, p in g])
        ut, ub = _u(up), _u(dn)
        if abs(ut) < 1e-6 or abs(ub) < 1e-6: continue
        out.append(float(np.real(ut*np.conj(ub))/(abs(ut)*abs(ub))))
    return (float(np.mean(out)) if out else float("nan")), out


P_ORDER_O3_MAX = -0.5      # mean prismatic order below this -> O3
P_ORDER_P3_MIN = +0.5      # above this -> P3


def phase_of_p_order(mean_p):
    if not np.isfinite(mean_p): return "n/a"
    if mean_p <= P_ORDER_O3_MAX: return "O3"
    if mean_p >= P_ORDER_P3_MIN: return "P3"
    return "AMB"


# ============================================================ cell geometry (heo_worker.py:949-987)
def cell_metrics(cell):
    """lifted: heo_worker.py:949."""
    cell = np.asarray(cell, dtype=float)
    a, b, _c = cell
    V = abs(np.linalg.det(cell))
    A_par = np.linalg.norm(np.cross(a, b))
    h_perp = V / A_par
    return V, A_par, h_perp


def cell_angles(cell):
    """lifted: heo_worker.py:959."""
    a, b, c = np.asarray(cell, dtype=float)
    ang = lambda u, v: math.degrees(math.acos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    return ang(b, c), ang(a, c), ang(a, b)


def dV_report(cell_pris, cell_desod, n_slabs=3):
    """lifted: heo_worker.py:965.  Rotation-invariant volume decomposition between paired cells."""
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


# ============================================================ voltages (heo_worker.py:881-892)
V_SEGMENTS = [("pris", "x081"), ("x081", "x067"), ("x067", "desod")]


def v_segments(E_nat, E_Na):
    """lifted: heo_worker.py:883.  V_seg(x_i -> x_{i+1}) = [E_nat(x_{i+1}) + dn*E_Na - E_nat(x_i)]/dn,
    dn = 5, 4, 4.  Discharge sign convention as V_avg (CLAUDE.md section 3): always positive."""
    out = {}
    for a, b in V_SEGMENTS:
        dn = X_STEMS[b] - X_STEMS[a]
        key = f"V_seg_{X_LABEL[a]}_{X_LABEL[b]}"
        out[key] = ((E_nat[b] + dn * E_Na - E_nat[a]) / dn) if (a in E_nat and b in E_nat) else float("nan")
    return out


def v_avg(E_pris, E_desod, E_Na):
    """CLAUDE.md section 3 (frozen): V_avg = [E_desod + 13*E_Na - E_pris] / 13, must be > 0."""
    return (E_desod + 13.0 * E_Na - E_pris) / 13.0


def de_mev_fu(E_O3, E_P3):
    """CLAUDE.md section 3 (frozen): dE = (E_P3 - E_O3) * 1000 / 27 meV/f.u.; dE > 0 = O3 wins."""
    return (E_P3 - E_O3) * 1000.0 / N_FU


# ============================================================ misc
def struct_hash(struct):
    """Deterministic structure fingerprint: sha256 over lattice (r6) + sorted (species, frac r6).
    Computed once on RunPod and carried in the manifest; identical code here so any machine
    can recompute it from the POSCAR."""
    lat = [[round(float(x), 6) for x in row] for row in struct.lattice.matrix]
    sites = sorted(
        (s.specie.symbol, round(float(s.frac_coords[0]) % 1.0, 6),
         round(float(s.frac_coords[1]) % 1.0, 6), round(float(s.frac_coords[2]) % 1.0, 6))
        for s in struct)
    return hashlib.sha256(json.dumps([lat, sites]).encode()).hexdigest()[:16]


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def bcc_na_2atom():
    """The one structure the spec allows to be built in code (section 5): bcc Na, 2-atom cell,
    relaxed with the same makers/settings; only its DFT energy enters V_seg/V_avg (spec 8-5)."""
    from pymatgen.core import Structure, Lattice
    return Structure(Lattice.cubic(4.29), ["Na", "Na"], [[0, 0, 0], [0.5, 0.5, 0.5]])
