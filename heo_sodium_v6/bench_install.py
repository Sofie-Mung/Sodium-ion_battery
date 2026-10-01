#!/usr/bin/env python3
"""bench_install.py - one-shot install + smoke test for every benchmark adapter (M01-M12).

    python bench_install.py                      # all 12 models, 3 installs in parallel
    python bench_install.py --models orb_mpa,mace_mpa --jobs 1
    python bench_install.py --force tece_oam_rra # wipe that venv and rebuild it
    python bench_install.py --smoke-only         # skip installs, re-run the smoke tests

Per model, in <bench>/venvs/<adapter>:
  1. venv       - shared-torch venv (--system-site-packages) or, for isolated=True, a uv venv
                  with its own python/torch (bench_models.ADAPTERS is the single source of specs)
  2. pip_steps  - ordered multi-stage installs (own index URLs), then `pip`
  3. runner deps- numpy/pandas/scipy/pymatgen/ase: bench_anchor_runner.py imports them, and the
                  isolated venvs do NOT inherit them from the system (the 09-17 M06/M11 failure).
                  Installed under a constraints file that freezes the torch/numpy the model
                  already pinned, so this step can never upgrade the model's stack.
  4. smoke      - in the venv: import the runner modules, build the ASE calculator, single-point
                  E/F/stress on rocksalt NaCl + NiO (Na + a 3d TM + O, like the benchmark cells).
                  THIS is the real pass criterion: an import-only check passed on 09-17 for
                  models that then died at load time.

Outputs: <bench>/results/install_report.json (+ pkg_versions_bench.json, the notebook's old name)
         <bench>/logs/install_<adapter>.log  (full pip + smoke output - send this when asking why)
A model whose spec hash is unchanged and whose smoke passed is skipped on re-runs.
Failures are recorded, never fatal (MODEL_BENCHMARK_SPEC 10-6: they stay in the results table).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BENCH = os.environ.get("HEO_BENCH_ROOT", "/workspace/heo_bench")
VENVS = f"{BENCH}/venvs"
LOGS = f"{BENCH}/logs"
REPORT = f"{BENCH}/results/install_report.json"
# Package caches on the volume, next to the venvs (2026-09-30): the default ~/.cache sits on the
# pod's ~20 GB container disk, which four torch builds (~15-20 GB of wheels) overflow; on the same
# filesystem as the venvs uv can also hardlink instead of copying.
os.environ.setdefault("UV_CACHE_DIR", f"{BENCH}/.cache/uv")
os.environ.setdefault("PIP_CACHE_DIR", f"{BENCH}/.cache/pip")

RUNNER_DEPS = ["numpy", "pandas", "scipy", "pymatgen", "ase>=3.23"]   # ase>=3.23: FrechetCellFilter
# packages the model stack pins -> frozen while the runner deps go in
FREEZE = ["torch", "numpy", "scipy", "ase", "pymatgen", "e3nn", "torch-geometric", "torch_geometric",
          "torch-scatter", "torch_scatter", "torch-sparse", "torch_sparse", "pandas"]
_SMOKE_LOCK = threading.Lock()   # one smoke at a time: parallel ones would fight over GPU memory
STEP_TIMEOUT = int(os.environ.get("HEO_INSTALL_STEP_TIMEOUT", 3600))
SMOKE_TIMEOUT = int(os.environ.get("HEO_SMOKE_TIMEOUT", 1800))   # includes checkpoint download


# ============================================================ helpers
def _uv():
    uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
    if not os.path.exists(uv):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
    return uv


def _run(cmd, log, timeout=STEP_TIMEOUT, env=None):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {' '.join(cmd)[:300]}")


def _pip(py, args, log):
    """pip inside the venv; uv-created venvs have no pip -> `uv pip install --python`."""
    has_pip = subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode == 0
    if has_pip:
        _run([py, "-m", "pip", "install", "--progress-bar", "off", *args], log)
    else:
        _run([_uv(), "pip", "install", "--python", py, *args], log)


def _torch_pin(py, log):
    """-c constraints freezing the torch trio already in an isolated venv (its CUDA build and
    driver range are deliberate): later steps may add packages but never re-resolve torch."""
    tp = _installed_versions(py, ["torch", "torchvision", "torchaudio"])
    if not tp:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(f"{n}=={v}" for n, v in tp.items()) + "\n")
    log.write(f"[pin] torch frozen: {tp}\n")
    return ["-c", tf.name]


def _spec_hash(ad):
    spec = dict(pip=ad.get("pip"), pip_steps=ad.get("pip_steps"), python=ad.get("python"),
                git=ad.get("git"), ts_pip=ad.get("ts_pip"),
                torch=ad.get("torch"), isolated=ad.get("isolated"), runner=RUNNER_DEPS)
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def _installed_versions(py, names):
    code = ("import json, importlib.metadata as m\nout = {}\n"
            f"for n in {names!r}:\n"
            "    try: out[n] = m.version(n)\n"
            "    except Exception: pass\nprint(json.dumps(out))")
    r = subprocess.run([py, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


# ============================================================ install one model
def install(key, force=False, smoke_only=False):
    import bench_models
    ad = bench_models.ADAPTERS[key]
    vdir, py = f"{VENVS}/{key}", f"{VENVS}/{key}/bin/python"
    marker = f"{vdir}/.heo_install_ok"
    h = _spec_hash(ad)
    rec = dict(adapter=key, model_id=ad["model_id"], spec_hash=h, stage="venv", ok=False,
               error="", t_s=0.0)
    t0 = time.time()
    os.makedirs(LOGS, exist_ok=True)
    with open(f"{LOGS}/install_{key}.log", "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} install {key} spec {h} =====\n")
        try:
            if force and os.path.isdir(vdir):
                shutil.rmtree(vdir)
            # A venv whose python/isolation differs from the spec is rebuilt even when it never
            # passed a smoke test (no marker): the 09-17 venvs (py3.13 M11, shared-env M01/M02)
            # would otherwise be reused under the new spec.  .heo_venv is written at creation.
            venv_sig = f"python={ad.get('python')}|isolated={bool(ad.get('isolated'))}"
            sig_file = f"{vdir}/.heo_venv"
            if not smoke_only and os.path.isdir(vdir):
                have = open(sig_file).read().strip() if os.path.exists(sig_file) else None
                if have != venv_sig:
                    log.write(f"[venv] {vdir}: {have or 'no .heo_venv (pre-09-30 venv)'} != {venv_sig} "
                              "-> rebuilding from scratch\n")
                    shutil.rmtree(vdir)
            done = os.path.exists(marker) and open(marker).read().strip() == h
            if not smoke_only and not done:
                if os.path.isdir(vdir) and os.path.exists(marker):      # spec changed -> rebuild clean
                    shutil.rmtree(vdir)
                if not os.path.exists(py):
                    if ad.get("isolated") and ad.get("python"):
                        _run([_uv(), "venv", vdir, "--python", ad["python"]], log)
                    elif ad.get("isolated"):
                        _run([_uv(), "venv", vdir, "--system-site-packages"], log)
                    else:
                        _run([sys.executable, "-m", "venv", "--system-site-packages", vdir], log)
                    open(sig_file, "w").write(venv_sig)
                src = f"{vdir}/src"
                if ad.get("git"):                 # editable installs that need a real clone (M06)
                    rec["stage"] = "git"
                    if not os.path.isdir(f"{src}/.git"):
                        _run(["git", "clone", "--depth", "1", ad["git"], src], log)
                rec["stage"] = "torch"
                if ad.get("torch"):
                    _pip(py, [ad["torch"]], log)
                # isolated venvs: the first torch-installing step fixes torch for everything after
                pin = (lambda: _torch_pin(py, log)) if ad.get("isolated") else (lambda: [])
                for i, step in enumerate(ad.get("pip_steps", [])):
                    rec["stage"] = f"pip_step{i}"
                    _pip(py, [*pin(), *[a.replace("{SRC}", src) for a in step]], log)
                rec["stage"] = "pip"
                if ad.get("pip"):
                    _pip(py, [*pin(), *ad["pip"]], log)
                rec["stage"] = "runner_deps"
                pins = _installed_versions(py, FREEZE)
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as cf:
                    cf.write("\n".join(f"{n}=={v}" for n, v in pins.items()) + "\n")
                log.write(f"[runner_deps] constraints: {pins}\n")
                missing = [d for d in RUNNER_DEPS
                           if d.split(">")[0].split("=")[0] not in pins]
                if missing:
                    _pip(py, ["-c", cf.name, *missing], log)
                os.unlink(cf.name)
                # torch-sim (batched engine): after the ASE stack, under the same freeze, and never
                # fatal - a model whose torch-sim step fails still runs on the ASE engine.
                rec["ts_install"] = "not requested"
                if ad.get("ts_pip"):
                    rec["stage"] = "ts_pip"
                    pins = _installed_versions(py, FREEZE)
                    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as cf:
                        cf.write("\n".join(f"{n}=={v}" for n, v in pins.items()) + "\n")
                    try:
                        _pip(py, ["-c", cf.name, *ad["ts_pip"]], log)
                        rec["ts_install"] = "ok"
                    except Exception as e:             # noqa: BLE001 - recorded, ASE path unaffected
                        rec["ts_install"] = f"FAIL {e}"
                        log.write(f"\n[ts_pip FAILED - model stays on the ASE engine] {e}\n")
                    os.unlink(cf.name)
            if not os.path.exists(py):
                raise RuntimeError(f"no venv at {vdir} - run without --smoke-only first")
            rec["stage"] = "smoke"
            with _SMOKE_LOCK, tempfile.TemporaryDirectory() as tmp:
                env = dict(os.environ, HEO_MODE="mock", HEO_WORKDIR=tmp, PYTHONUNBUFFERED="1")
                r = subprocess.run([py, os.path.abspath(__file__), "--_smoke", key],
                                   capture_output=True, text=True, timeout=SMOKE_TIMEOUT, env=env,
                                   cwd=HERE)
            log.write(f"\n[smoke stdout]\n{r.stdout}\n[smoke stderr]\n{r.stderr}\n")
            res = None
            for line in r.stdout.splitlines():
                if line.startswith("SMOKE_JSON "):
                    res = json.loads(line[len("SMOKE_JSON "):])
            if res is None:
                raise RuntimeError(f"smoke crashed (exit {r.returncode}): {r.stderr.strip()[-600:]}")
            rec.update(res)
            if not res["ok"]:
                raise RuntimeError(res["error"])
            rec["stage"], rec["ok"] = "done", True
            open(marker, "w").write(h)
        except Exception as e:                     # noqa: BLE001 - recorded, never fatal
            rec["error"] = rec["error"] or f"{type(e).__name__}: {e}"   # smoke already set its own
            log.write(f"\n[FAIL at {rec['stage']}] {rec['error']}\n{traceback.format_exc()}\n")
    if os.path.exists(py):
        rec["versions"] = _installed_versions(py, ["torch", "numpy", "ase", "pymatgen"])
    rec["t_s"] = round(time.time() - t0, 1)
    return rec


# ============================================================ smoke test (runs INSIDE the venv)
E_RANGE = (-15.0, -1.0)      # eV/atom: any MP/OMat-currency model on these oxides/halides lands inside;
                             # outside = broken load (wrong head, unit error, untrained weights)


def _namo2_cell():
    """Ordered rocksalt Na2NiMnO4 = Na(Ni0.5Mn0.5)O2, 8 atoms: the benchmark's own chemistry."""
    from ase.build import bulk
    at = bulk("NiO", "rocksalt", a=4.40, cubic=True)
    sym = at.get_chemical_symbols()
    tm = [i for i, x in enumerate(sym) if x == "Ni"]
    for i, el in zip(tm, ("Na", "Na", "Ni", "Mn")):
        sym[i] = el
    at.set_chemical_symbols(sym)
    return at


def _check(atoms, calc, name, chk):
    import numpy as np
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces())
    st = np.asarray(atoms.get_stress())
    if not (np.isfinite(e) and np.all(np.isfinite(f)) and np.all(np.isfinite(st))):
        raise RuntimeError(f"{name}: non-finite energy/forces/stress")
    if f.shape != (len(atoms), 3):
        raise RuntimeError(f"{name}: force array shape {f.shape}, expected ({len(atoms)}, 3)")
    epa = e / len(atoms)
    if not (E_RANGE[0] <= epa <= E_RANGE[1]):
        raise RuntimeError(f"{name}: E = {epa:.3f} eV/atom outside {E_RANGE} - the model loaded but "
                           "its output is not a PBE-level energy (wrong head/checkpoint?)")
    chk[name] = round(epa, 4)
    return e


def _fd_audit(calc, base):
    """Check 7: the model's stress and forces against central finite differences of its own energy
    on a 3%-compressed, rattled Na(Ni,Mn)O2 cell (compression makes the stress large enough that a
    unit or sign error cannot hide).  ASE convention: stress = (1/V) dE/d(strain).  The smoke test
    only checked 'finite' before, which a GPa-vs-eV/A^3 (x160) stress passes - and the cell filter,
    dV/dA/dh all hang on the stress.  OK <= 10% (+ small abs), GROSS > 50%: GROSS fails the model."""
    import numpy as np
    at = base.copy()
    at.set_cell(at.cell * 0.97, scale_atoms=True)
    at.rattle(stdev=0.03, seed=1)
    at.calc = calc
    s_model = np.asarray(at.get_stress(voigt=False))
    f_model = np.asarray(at.get_forces())
    V = at.get_volume()

    def e_at(cell=None, pos=None):
        a = at.copy()
        if cell is not None:
            a.set_cell(cell, scale_atoms=True)
        if pos is not None:
            a.set_positions(pos)
        a.calc = calc
        return float(a.get_potential_energy())

    def verdict(fd, model, scale_floor, abs_tol):
        err, scale = abs(fd - model), max(abs(fd), scale_floor)
        v = "OK" if err <= 0.10 * scale + abs_tol else ("WARN" if err <= 0.50 * scale + 5 * abs_tol else "GROSS")
        return dict(fd=round(fd, 6), model=round(float(model), 6), verdict=v)

    res, d = {}, 5e-3
    for i, j in ((0, 0), (2, 2), (0, 1)):
        eps = np.zeros((3, 3)); eps[i, j] = eps[j, i] = d
        dEde = (e_at(cell=at.cell[:] @ (np.eye(3) + eps)) - e_at(cell=at.cell[:] @ (np.eye(3) - eps))) / (2 * d)
        fd = dEde / V / (1.0 if i == j else 2.0)                 # symmetric shear counts sigma_ij twice
        res[f"stress_{i}{j}"] = verdict(fd, s_model[i, j], 0.01, 2e-4)
    h, p0 = 1e-2, at.get_positions()
    for a_i, c in ((0, 0), (len(at) - 1, 2)):
        pp, pm = p0.copy(), p0.copy()
        pp[a_i, c] += h; pm[a_i, c] -= h
        fd = -(e_at(pos=pp) - e_at(pos=pm)) / (2 * h)
        res[f"force_{a_i}{'xyz'[c]}"] = verdict(fd, f_model[a_i, c], 0.5, 0.02)
    return res


def _ts_audit(key, calc, base):
    """Checks 8-9 (torch-sim engine).  8: one batched forward over three different cells (8, 16 and
    8 atoms, different compressions) must reproduce the ASE calculator per structure - catches
    per-atom-vs-total energy, GPa-vs-eV/A^3, stress sign and atom/system mapping errors.  9: a real
    torch-sim FIRE + Frechet relaxation through bench_anchor_runner.ts_relax (the runner's code path,
    autobatcher on, capped) must lower every energy.  Also times a batch of 8 x 128-atom cells."""
    import time as _t
    import numpy as np
    out = dict(ts_ok=False, ts_check="8 build", ts_error="")
    try:
        import bench_anchor_runner as R
        import bench_models
        import torch_sim as ts
        out["ts_version"] = getattr(ts, "__version__", "?")
        t0 = _t.time()
        tsm = bench_models.build_ts(key, calc)
        out["t_load_ts_s"] = round(_t.time() - t0, 1)

        out["ts_check"] = "8 batch == ASE"
        cells = []
        for k, (rep, scale, seed) in enumerate((((1, 1, 1), 0.98, 2), ((2, 1, 1), 1.02, 3), ((1, 1, 1), 0.96, 4))):
            a = base.copy() * rep
            a.set_cell(a.cell * scale, scale_atoms=True)
            a.rattle(stdev=0.04, seed=seed)
            cells.append(a)
        ref = []
        for a in cells:
            b = a.copy(); b.calc = calc
            ref.append((b.get_potential_energy(), b.get_forces(), b.get_stress(voigt=False)))
        state = ts.io.atoms_to_state(cells, device=tsm.device, dtype=tsm.dtype)
        o = tsm(state)
        E = o["energy"].detach().cpu().numpy().ravel()
        F = o["forces"].detach().cpu().numpy().reshape(-1, 3)
        S = o["stress"].detach().cpu().numpy().reshape(-1, 3, 3)
        n = np.cumsum([0] + [len(a) for a in cells])
        dE = max(abs(E[i] - ref[i][0]) / len(a) for i, a in enumerate(cells))
        dF = max(float(np.abs(F[n[i]:n[i + 1]] - ref[i][1]).max()) for i in range(len(cells)))
        dS = max(float(np.abs(S[i] - ref[i][2]).max()) for i in range(len(cells)))
        out["ts_diff"] = dict(E_per_atom=float(dE), F=dF, S=dS)
        if not (dE <= 2e-3 and dF <= 2e-2 and dS <= 2e-3):
            raise RuntimeError(f"batched torch-sim output != ASE calculator: |dE|/atom {dE:.2e} (tol 2e-3), "
                               f"|dF| {dF:.2e} (2e-2 eV/A), |dS| {dS:.2e} (2e-3 eV/A^3)")

        out["ts_check"] = "9 torch-sim relax"
        R.TS_MAX_ATOMS = 2000                     # keep the autobatcher's memory probe short here
        t0 = _t.time()
        final, conv, _ = R.ts_relax(tsm, [c.copy() for c in cells[:2]], True)
        e1 = final.energy.detach().cpu().numpy().ravel()
        out["ts_relax"] = dict(s=round(_t.time() - t0, 1), converged=[bool(x) for x in conv],
                               dE_eV=[round(float(e1[i] - ref[i][0]), 4) for i in range(2)])
        if not (np.all(np.isfinite(e1)) and all(e1[i] < ref[i][0] + 1e-3 for i in range(2))):
            raise RuntimeError(f"torch-sim relax did not lower the energy: {out['ts_relax']}")

        out["ts_check"] = "10 batch timing"
        big = base.copy() * (2, 2, 4)
        batch = []
        for s in range(8):
            b = big.copy(); b.rattle(stdev=0.03, seed=10 + s); batch.append(b)
        st8 = ts.io.atoms_to_state(batch, device=tsm.device, dtype=tsm.dtype)
        tsm(st8)                                  # warm-up (graph build, kernels)
        t0 = _t.time()
        tsm(st8)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:                         # noqa: BLE001
            pass
        out["t_ts_128_batch8_s"] = round(_t.time() - t0, 3)   # one FIRE step for 8 cells
        out["ts_check"], out["ts_ok"] = "all", True
    except Exception as e:                        # noqa: BLE001 - recorded; model runs on ASE
        out["ts_error"] = f"[check {out['ts_check']}] {type(e).__name__}: {str(e)[:1200]}"
        traceback.print_exc()
    return out


def smoke(key):
    """Runs INSIDE the model's venv.  Pass = every check below; each failure names the check.
      1 runner imports    pandas/scipy/pymatgen/ase + heo_worker/xrd_tools/bench_anchor_runner
      2 GPU               torch sees CUDA (HEO_ALLOW_CPU=1 to waive) - a CPU run of 1,089 relaxes is useless
      3 load              bench_models.build(key) -> ASE calculator
      4 single points     NaCl, NiO, Na(Ni,Mn)O2: finite E/F/stress, E/atom in E_RANGE
      5 relax             bench_anchor_runner.ase_relax (the benchmark's exact FIRE+FrechetCellFilter
                          path) on a rattled Na(Ni,Mn)O2: energy goes down, cell stays sane
      6 128-atom cell     benchmark-sized single point: no OOM; time + peak GPU memory recorded
      7 finite diffs      stress/forces == dE/d(strain), -dE/dx of the model's own energy (units, sign)
      8-10 torch-sim      batched output == ASE, torch-sim FIRE relax lowers E, batch timing ->
                          ts_ok.  Does NOT affect ok: a model failing 8-10 runs on the ASE engine."""
    out = dict(ok=False, error="", check="", torch="", cuda=False, device="", E=None,
               t_load_s=None, t_calc_s=None)
    try:
        out["check"] = "1 runner imports"
        try:
            import torch
            out["torch"], out["cuda"] = torch.__version__, bool(torch.cuda.is_available())
        except ImportError:
            torch = None
            out["torch"] = "none"                                     # fine for mock only
        import pandas, scipy, pymatgen.core, ase                      # noqa: F401 - runner deps
        import heo_worker, xrd_tools                                  # noqa: F401 - runner modules
        import bench_anchor_runner                                    # the relax path itself
        import bench_models
        out["device"] = bench_models._device()
        if key == "mock":
            out["ok"] = True
            return
        out["check"] = "2 GPU"
        if not out["cuda"] and os.environ.get("HEO_ALLOW_CPU") != "1":
            raise RuntimeError(f"torch {out['torch']} in this venv sees no CUDA device (CPU-only wheel, "
                               "or a CUDA build newer than the pod driver) - the benchmark would run on CPU")
        out["check"] = "3 load"
        t0 = time.time()
        calc = bench_models.build(key)
        out["t_load_s"] = round(time.time() - t0, 1)
        if calc is None:
            raise RuntimeError("build() returned no calculator")

        import numpy as np
        from ase.build import bulk
        out["check"] = "4 single points"
        chk = {}
        t0 = time.time()
        _check(bulk("NaCl", "rocksalt", a=5.64), calc, "NaCl", chk)
        _check(bulk("NiO", "rocksalt", a=4.17) * (2, 1, 1), calc, "NiO", chk)
        cell = _namo2_cell()
        _check(cell.copy(), calc, "NaNiMnO", chk)
        out["t_calc_s"] = round(time.time() - t0, 2)

        out["check"] = "5 relax"
        at = cell.copy()
        at.rattle(stdev=0.05, seed=0)
        e0 = _check(at, calc, "NaNiMnO_rattled", chk)
        v0 = at.get_volume()
        t0 = time.time()
        at, conv, nsteps = bench_anchor_runner.ase_relax(at, calc, fmax=0.05, max_steps=60)
        e1 = _check(at, calc, "NaNiMnO_relaxed", chk)
        out["relax"] = dict(steps=nsteps, converged=conv, dE_eV=round(e1 - e0, 4),
                            vol_ratio=round(at.get_volume() / v0, 3),
                            s_per_step=round((time.time() - t0) / max(nsteps, 1), 3))
        if e1 > e0 + 1e-3:
            raise RuntimeError(f"relax raised the energy ({e0:.4f} -> {e1:.4f} eV): forces are not the "
                               "energy gradient, or the optimizer diverged")
        if not 0.7 < out["relax"]["vol_ratio"] < 1.3:
            raise RuntimeError(f"relax changed the volume by x{out['relax']['vol_ratio']} (stress broken?)")

        out["check"] = "6 128-atom cell"
        big = cell * (2, 2, 4)
        if torch is not None and out["cuda"]:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _check(big, calc, "NaNiMnO_128", chk)
        out["t_128_s"] = round(time.time() - t0, 2)
        if torch is not None and out["cuda"]:
            out["gpu_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        out["E"] = chk                                               # eV/atom per test cell

        out["check"] = "7 finite differences"
        out["fd"] = _fd_audit(calc, cell)
        gross = [k for k, v in out["fd"].items() if v["verdict"] == "GROSS"]
        if gross:
            raise RuntimeError(f"reported stress/forces disagree grossly with dE/d(strain), -dE/dx "
                               f"(unit, sign or non-conservative output): { {k: out['fd'][k] for k in gross} }")
        out["check"], out["ok"] = "all", True
        # 8-9: the batched torch-sim engine.  Never flips ok - a model failing here runs on ASE.
        out.update(_ts_audit(key, calc, cell))
    except Exception as e:                        # noqa: BLE001
        out["error"] = f"[check {out['check']}] {type(e).__name__}: {str(e)[:1500]}"
        traceback.print_exc()
    finally:
        print("SMOKE_JSON " + json.dumps(out), flush=True)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("HEO_BENCH_MODELS", ""))
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("HEO_INSTALL_JOBS", 3)))
    ap.add_argument("--force", default="", help="comma list of adapters to rebuild from scratch")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--_smoke", default="")
    a = ap.parse_args()
    if a._smoke:
        return smoke(a._smoke)

    import bench_models
    models = [m for m in a.models.split(",") if m] or list(bench_models.DEFAULT_MODELS)
    bad = [m for m in models if m not in bench_models.ADAPTERS]
    if bad:
        sys.exit(f"unknown adapter(s) {bad}; known: {sorted(bench_models.ADAPTERS)}")
    force = set(m for m in a.force.split(",") if m)
    os.makedirs(VENVS, exist_ok=True); os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    print(f"installing {len(models)} adapters into {VENVS} ({a.jobs} parallel) - "
          f"per-model logs: {LOGS}/install_<adapter>.log", flush=True)

    report = json.load(open(REPORT)) if os.path.exists(REPORT) else {}

    def _one(k):
        r = install(k, force=k in force, smoke_only=a.smoke_only)
        tag = "OK  " if r["ok"] else "FAIL"
        extra = (f"load {r.get('t_load_s')}s | relax {r.get('relax', {}).get('s_per_step')} s/step | "
                 f"128at {r.get('t_128_s')}s {r.get('gpu_peak_GB')} GB | "
                 + (f"torch-sim OK (8x128at step {r.get('t_ts_128_batch8_s')}s)" if r.get("ts_ok")
                    else f"torch-sim FAIL -> ASE engine: {str(r.get('ts_error', ''))[:200]}")
                 if r["ok"] else f"at {r['stage']}: {r['error'][:300]}")
        print(f"  [{tag}] {r['model_id']:>4} {k:<18} {r['t_s']:>7.0f}s  {extra}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for r in ex.map(_one, models):
            report[r["adapter"]] = r
    json.dump(report, open(REPORT, "w"), indent=1)
    # the notebook's older file name / shape ({"venv", "import", "t_s"}) keeps cell 3 working
    legacy = {k: {"venv": v["stage"], "import": "ok" if v["ok"] else f"FAIL {v['error'][:200]}",
                  "t_s": v["t_s"]} for k, v in report.items()}
    json.dump(legacy, open(f"{BENCH}/results/pkg_versions_bench.json", "w"), indent=1)

    ok = [k for k in models if report[k]["ok"]]
    ts_ok = [k for k in ok if report[k].get("ts_ok")]
    print(f"\n{len(ok)}/{len(models)} adapters pass install + smoke (load, E/F/stress, relax, 128 atoms, "
          f"finite differences).  report: {REPORT}")
    print(f"{len(ts_ok)}/{len(ok)} of them also pass the torch-sim batch checks (batched engine); "
          f"ASE engine for: {[k for k in ok if k not in ts_ok]}")
    for k in models:
        if not report[k]["ok"]:
            print(f"  FAIL {k}: stage={report[k]['stage']} -> tail {LOGS}/install_{k}.log")
    return 0 if len(ok) == len(models) else 1


if __name__ == "__main__":
    sys.exit(main())
