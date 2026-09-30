"""Builds sse_screening.ipynb from cell sources (kept as a script so the notebook is reproducible)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md(r"""# Li₆PS₅Cl dual-doping MLIP screening — `sse_screening.ipynb`

Implements `SSE_PIPELINE_SPEC.md` end to end (ORB v3 only since 2026-09-28 — SevenNet dropped for speed, see spec section 0.1; torch-sim, kinisi). All functions live in `sse_worker.py`; this notebook only sets `CONFIG`, launches one worker per GPU, waits, aggregates and decides.

```
1 CONFIG      1b watchdog      2 INSTALL      3 environment check -> run_manifest.json
4 charge-ledger test   5 structure generation + invariants   6 kinisi synthetic test
7 smoke test (A, 600 K, 2 ps + 5 ps)   8 throughput -> logs/throughput.txt (user check if > budget_hours = 8 h)
9 Stage I-M model selection   10 Stage I pilot (relax, hull, MD, kinisi, gates, figures 1-4)   ** STOP: user confirms gates + G-stab **
11 Stage II (main model: 585+33 relax, entropy-corrected hull filter, short MD, S_syn, group top-5, long MD with a new seed, contrast rescoring)
12 DFT hand-off export   13 finish (manifest, DONE, pod stop)
```

**Run All** runs cells 1–10 and stops before Stage II unless `RUN_STAGE_II=True` **and** `CONFIG["stab_threshold_eV"]` is set (user decision after Stage I). Every step is resume-safe: results already on disk are never recomputed.

Environment variables read by cell 1: `SSE_WORKDIR` (default: notebook directory), `MP_API_KEY` (hull), `SSE_MOCK=1` (Lennard-Jones wiring test, CPU), `RUNPOD_POD_ID` (auto stop).
""")

md("## 0 — secrets (a terminal `export` does not reach an already running Jupyter kernel)")
code(r'''# ============ CELL 0: SECRETS ============
# Put the key in a file once:  echo -n "<key>" > /workspace/.mp_api_key && chmod 600 /workspace/.mp_api_key
# Never write the key itself into this notebook (it is shared and committed far more often than a shell).
import os
for _p in ("/workspace/.mp_api_key", os.path.expanduser("~/.mp_api_key")):
    if not os.environ.get("MP_API_KEY") and os.path.exists(_p):
        os.environ["MP_API_KEY"] = open(_p).read().strip(); print("MP_API_KEY loaded from", _p)
print("MP_API_KEY set:", bool(os.environ.get("MP_API_KEY")), "| needed from cell 10 (hull references) onward")
''')

md("## 1 — CONFIG (single source of truth; the workers read `outputs/config.json`)")
code(r'''# ============ CELL 1: CONFIG ============
import os, sys, json, time, glob, subprocess, hashlib, shutil
from pathlib import Path
import numpy as np, pandas as pd

WORKDIR = os.environ.get("SSE_WORKDIR", os.getcwd())
os.chdir(WORKDIR); sys.path.insert(0, WORKDIR)
MOCK = os.environ.get("SSE_MOCK", "0") == "1"           # Lennard-Jones wiring test on CPU: no MLIP, no physics

CONFIG = {
    "seed": 20260927,
    "workdir": WORKDIR,
    "host_cif": "inputs/Li6PS5Cl.cif",
    "exp_reference_csv": "inputs/exp_reference.csv",
    "supercell": (2, 2, 2),

    # anion disorder: fraction of 4c sites occupied by Cl (same for ALL compositions)
    "scl_disorder_4c_cl_fraction": 0.375,      # 12 of 32 swaps in 2x2x2
    # Li 48h pair detection
    "li48h_pair_cutoff": 2.0,                  # Angstrom; verified with a distance histogram (spec 5.2)
    # the provided CIF may have Li on 24g (doublet midpoints, MP-style ordered cell) -> spec 5.2 pair
    # sampling is impossible.  The loader refuses such a file unless the user sets this to True.
    "allow_24g_host": True,                    # USER DECISION 2026-09-27: inputs/Li6PS5Cl.cif is the 24g ordered cell
    # P-site cap
    "max_ge_per_supercell": 16,                # 50% of P
    # structure generation
    "n_configs_pilot": 8,                      # 4 random + 2 clustered + 2 dispersed
    "n_configs_stage2": 5,                     # 3 random + 1 clustered + 1 dispersed
    "min_interatomic_dist": 1.6,               # Angstrom, hard assert (INV-7)
    "interstitial_min_cation_dist": 2.0,
    "interstitial_min_anion_dist": 1.9,
    "vacancy_near_radius": 4.0,
    # relax
    "relax_fmax": 0.02,                        # eV/A
    "relax_max_steps": 1000,
    "relax_cell": True,                        # FIRE + Frechet cell filter
    "relax_chunk": 16,                         # structures per torch-sim optimize call (autobatched)
    # config selection for MD
    "t_syn": 800.0,                            # K, Boltzmann weights (sensitivity) and entropy correction
    "n_configs_md_pilot": 2,                   # USER DECISION 2026-09-28 (was 3): 4-8 h on 4 GPUs
    "n_configs_md_stage2_short": 2,
    # MD
    "md_timestep_fs": 2.0,                     # USER DECISION 2026-09-28 (was 1.0)
    "md_temperatures_pilot": [600, 750, 900],  # USER DECISION 2026-09-28 (was 500/600/700); 600 K = gate temperature
    "md_npt_ps": 10.0,                         # USER DECISION 2026-09-28 (was 20)
    "md_nvt_ps_pilot": 100.0,                  # USER DECISION 2026-09-28 (was 2000): resolves ~1.5x differences in sigma ratio
    "md_nvt_ps_stage2_short": 500.0,
    "md_stage2_short_T": 600,
    "md_save_every_fs": 100.0,
    "md_thermostat": "nose_hoover",            # fallback: "langevin" with friction <= 0.002 /fs (recorded)
    "md_langevin_friction_per_fs": 0.002,
    "md_batch_size": 8,                        # systems per ts.integrate call on one GPU (OOM -> halves automatically)
    "md_keep_h5": False,                       # nvt.npz (unwrapped, float32) is the archive; keep torch-sim h5 too?
    "md_nondiffusive_msd_A2": 20.0,
    # analysis
    "kinisi_start_dt_ps": 2.0,                 # skip the ballistic regime
    "kinisi_n_dt": 120,
    "kinisi_n_samples": 1000, "kinisi_n_walkers": 32, "kinisi_n_burn": 500, "kinisi_n_thin": 10,
    "ci_level": 0.95,
    "t_extrapolate": 300.0,
    "n_mc_samples": 4000,
    # hull
    "hull_mp_ehull_max": 0.05,                 # eV/atom: MP phases kept as references
    "hull_min_ref_atoms": 24,
    # models
    "models": ["orb_v3"],                      # USER DECISION 2026-09-28: ORB only (SevenNet: 0.23 ns/day, OOM at batch 4)
    "orb_checkpoint": "orb_v3_conservative_inf_mpa",
    "orb_precision": "float32-high",
    "sevenn_checkpoint": "7net-omni", "sevenn_modal": "mpa", "sevenn_fallback_checkpoint": "7net-mf-ompa",
    # Stage II stability threshold (G-stab): None until the user confirms it after Stage I
    "stab_threshold_eV": None,
    # run control
    "budget_hours": 8.0,                       # USER DECISION 2026-09-28 (was 72): 4-8 h on 4 GPUs
    "confirm_over_budget": False,              # set True only after the user has confirmed
    "continue_on_ga_fail": True,               # USER DECISION 2026-09-29: G-a failure is a warning, Stage I still runs
    "auto_terminate": True,
    "max_hours": 96.0, "idle_min": 180.0,
}
RUN_STAGE_IM, RUN_STAGE_I, RUN_STAGE_II, RUN_EXPORT = True, True, False, True
if os.environ.get("SSE_RUN_STAGE_II") == "1": RUN_STAGE_II = True            # after the user has confirmed Stage I
if os.environ.get("SSE_STAB_THRESHOLD"): CONFIG["stab_threshold_eV"] = float(os.environ["SSE_STAB_THRESHOLD"])
N_CPU_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))

if MOCK:   # tiny, CPU-only wiring run
    CONFIG.update(models=["mock", "mock2"], relax_max_steps=20, md_npt_ps=0.1, md_nvt_ps_pilot=0.4, md_nvt_ps_stage2_short=0.3,
                  md_save_every_fs=10.0, kinisi_n_samples=150, kinisi_n_burn=60, n_mc_samples=500, n_configs_md_pilot=2,
                  md_temperatures_pilot=[500, 600, 700], auto_terminate=False, budget_hours=1e9)
    N_CPU_WORKERS = 2

for sub in ("outputs", "outputs/jobs", "outputs/structures", "outputs/trajectories", "logs"):
    os.makedirs(os.path.join(WORKDIR, sub), exist_ok=True)
Path(WORKDIR, "outputs", "DONE").unlink(missing_ok=True)
json.dump({**CONFIG, "supercell": list(CONFIG["supercell"])}, open(os.path.join(WORKDIR, "outputs", "config.json"), "w"), indent=1)

# GPU count
N_GPUS = 0
try:
    import torch; N_GPUS = torch.cuda.device_count()
except Exception:
    pass
N_GPUS = int(os.environ.get("SSE_NGPU", 0)) or N_GPUS or 1

# every notebook print also goes to logs/notebook_stdout.log (browser Run All loses output on disconnect)
class _Tee:
    _sse_tee = True
    def __init__(self, stream, path): self._s = stream; self._f = open(path, "a", buffering=1)
    def write(self, x): self._s.write(x); self._f.write(x); return len(x)
    def flush(self): self._s.flush(); self._f.flush()
    def __getattr__(self, a): return getattr(self._s, a)
if not getattr(sys.stdout, "_sse_tee", False):
    sys.stdout = _Tee(sys.stdout, os.path.join(WORKDIR, "logs", "notebook_stdout.log")); sys.stderr = _Tee(sys.stderr, os.path.join(WORKDIR, "logs", "notebook_stdout.log"))
print(f"\n===== run start {time.strftime('%Y-%m-%d %H:%M:%S')} | workdir {WORKDIR} | GPUs {N_GPUS} | mock={MOCK} | models {CONFIG['models']}")
print("MP_API_KEY set:", bool(os.environ.get("MP_API_KEY")), "| pod:", os.environ.get("RUNPOD_POD_ID"))
''')

md("## 1b — pod cost guard (detached watchdog: hard deadline, idle, DONE)")
code(r'''# ============ CELL 1b: WATCHDOG ============
# A cell stops at the first exception, so the terminate cell may never run and the pod bills forever.
# This independent process stops the pod on (1) hard deadline, (2) no file activity + idle GPUs, (3) DONE + 5 min.
import textwrap
WATCHDOG_SRC = textwrap.dedent(r"""
    import os, sys, time, subprocess, urllib.request
    from pathlib import Path
    WORKDIR = Path(sys.argv[1]); MAX_H = float(sys.argv[2]); IDLE_MIN = float(sys.argv[3]); MODE = sys.argv[4]
    LOG = open(WORKDIR / "logs" / "watchdog.log", "a", buffering=1)
    def say(m): LOG.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")
    def pod_stop():
        pod = os.environ.get("RUNPOD_POD_ID")
        if not pod: return False, "RUNPOD_POD_ID not set"
        for c in (["runpodctl", "stop", "pod", pod], ["runpodctl", "pod", "stop", pod]):
            try:
                if subprocess.run(c, capture_output=True).returncode == 0: return True, " ".join(c)
            except Exception as e: say(f"  {c[0]} unusable: {e!r}")
        key = os.environ.get("RUNPOD_API_KEY")
        if not key: return False, "runpodctl failed and RUNPOD_API_KEY not set"
        req = urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}/stop", method="POST", headers={"Authorization": f"Bearer {key}"})
        try: urllib.request.urlopen(req, timeout=20); return True, "REST"
        except Exception as e: return False, f"REST failed: {e!r}"
    def gpus_busy():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
            return any(int(x) > 5 for x in out.split() if x.strip().isdigit())
        except Exception: return False
    def snapshot():
        s = []
        for p in (WORKDIR / "outputs").rglob("*"):
            if p.is_file():
                try: s.append((str(p), p.stat().st_size, int(p.stat().st_mtime)))
                except OSError: pass
        return hash(tuple(sorted(s)))
    def fire(why):
        if MODE != "stop": say(f"LOG-ONLY - would stop now: {why}"); return
        say(f"STOPPING - {why}"); ok, how = pod_stop(); say(f"  -> {'OK ' + how if ok else 'FAILED: ' + how}")
        if ok: sys.exit(0)
    say(f"watchdog up | max {MAX_H} h | idle {IDLE_MIN} min | mode {MODE} | pod {os.environ.get('RUNPOD_POD_ID')}")
    t0 = time.time(); last_sig = snapshot(); last_change = time.time(); done_at = None
    while True:
        time.sleep(60)
        if (WORKDIR / "outputs" / "DONE").exists():
            done_at = done_at or time.time()
            if time.time() - done_at > 300: fire("DONE written, pod still up after 5 min"); done_at += 21600
            continue
        if time.time() - t0 > MAX_H * 3600: fire(f"hard deadline {MAX_H} h"); t0 += 21600
        sig = snapshot()
        if sig != last_sig: last_sig, last_change = sig, time.time(); continue
        idle_s = time.time() - last_change
        if idle_s > IDLE_MIN * 60:
            if gpus_busy(): say(f"  files idle {idle_s/60:.0f} min but GPUs busy - holding"); last_change = time.time()
            else: fire(f"no file activity for {idle_s/60:.0f} min and GPUs idle"); last_change = time.time()
""")
_mode = "stop" if CONFIG["auto_terminate"] else "none"
if os.environ.get("RUNPOD_POD_ID") and os.environ.get("SSE_WATCHDOG", "1") == "1":
    subprocess.run(["pkill", "-f", "sse_watchdog.py"], capture_output=True)
    _wd = Path(WORKDIR, "logs", "sse_watchdog.py"); _wd.write_text(WATCHDOG_SRC)
    _p = subprocess.Popen([sys.executable, str(_wd), WORKDIR, str(CONFIG["max_hours"]), str(CONFIG["idle_min"]), _mode],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"watchdog pid {_p.pid} | mode {_mode} | max {CONFIG['max_hours']} h | idle {CONFIG['idle_min']} min")
else:
    print("watchdog not started (no RUNPOD_POD_ID or SSE_WATCHDOG=0)")
''')

md("## 2 — packages (import first; pip only for what is missing; versions recorded)")
code(r'''# ============ CELL 2: INSTALL ============
import importlib.util
from importlib.metadata import version as _ver, PackageNotFoundError
PKGS = {"pymatgen": "pymatgen", "mp_api": "mp-api", "ase": "ase", "torch_sim": "torch-sim-atomistic", "kinisi": "kinisi",
        "pptx": "python-pptx", "scipy": "scipy", "pandas": "pandas", "orb_models": "orb-models", "sevenn": "sevenn", "tables": "tables"}
need = list(PKGS) if not MOCK else [m for m in PKGS if m not in ("orb_models", "sevenn", "mp_api")]
missing = [PKGS[m] for m in need if importlib.util.find_spec(m) is None]
if missing:
    # never let pip replace the pod's torch: pin the installed version through a constraints file
    extra = []
    try:
        import torch as _t
        _c = os.path.join(WORKDIR, "logs", "pip_constraints.txt")
        open(_c, "w").write("torch==" + _t.__version__.split("+")[0] + "\n"); extra = ["-c", _c]
    except Exception:
        pass
    print("installing:", missing, "| torch pinned:", bool(extra))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *extra, *missing])
PKG_VERSIONS = {}
for mod, pip in PKGS.items():
    if importlib.util.find_spec(mod) is None:
        continue
    try: PKG_VERSIONS[pip] = _ver(pip)
    except PackageNotFoundError: PKG_VERSIONS[pip] = "?"
try:
    import torch; PKG_VERSIONS["torch"] = torch.__version__; PKG_VERSIONS["cuda"] = torch.version.cuda
except Exception: pass
print("versions:", PKG_VERSIONS)
''')

md("## 3 — worker module + environment check (spec 12-1) → `run_manifest.json`")
code(r'''# ============ CELL 3: ENVIRONMENT CHECK ============
import importlib, sse_worker as W; importlib.reload(W)
cfg = W.load_config(os.path.join(WORKDIR, "outputs", "config.json"))
print("sse_worker.py md5:", W.file_md5(W.__file__))
assert os.path.exists(os.path.join(WORKDIR, "PILOT_CRITERIA.md")), "PILOT_CRITERIA.md must exist before any Stage I run"
CRITERIA_MD5 = W.file_md5(os.path.join(WORKDIR, "PILOT_CRITERIA.md"))

HOST = W.load_host(cfg, log_dir=os.path.join(WORKDIR, "logs"))
print(f"host: a={HOST.a:.4f} A | Li site mode: {HOST.li_site_mode} | pairs {HOST.li_pairs.shape} | {HOST.pair_stats}")
if HOST.li_site_mode == "24g":
    print("  !! NOTE: Li on 24g (ordered MP-style cell). Spec 5.2 48h-pair sampling is NOT performed (user-approved deviation).")

# one host configuration for the single-point test
_atoms, _meta = W.build_config(HOST, W.pilot_compositions()[0], "random", W.derived_seed(cfg["seed"], "envcheck"), cfg)
MANIFEST = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "workdir": WORKDIR, "seed": cfg["seed"], "packages": PKG_VERSIONS,
            "host_cif_md5": HOST.source_md5, "host_li_site_mode": HOST.li_site_mode, "host_pair_stats": HOST.pair_stats,
            "pilot_criteria_md5": CRITERIA_MD5, "n_gpus": N_GPUS, "config": {**cfg, "supercell": list(cfg["supercell"])}, "models": {}, "mock": MOCK}
for m in cfg["models"]:
    t0 = time.time()
    h = W.load_model("mock" if m.startswith("mock") else m, cfg)
    e, f = W.energy_forces(h, _atoms)
    MANIFEST["models"][m] = dict(tag=h.tag, backend=h.backend, device=h.device, notes=h.notes, e_per_atom_host_test=e / len(_atoms),
                                fmax_host_test=float(np.linalg.norm(f, axis=1).max()), load_s=time.time() - t0)
    print(f"{m}: {h.tag} | backend {h.backend} | device {h.device} | E/atom {e/len(_atoms):.4f} eV | Fmax {np.linalg.norm(f, axis=1).max():.3f} eV/A | {time.time()-t0:.0f}s")
    if m == "sevenn":
        print("   torch-sim support for SevenNet:", "YES (sevenn.torchsim.SevenNetModel)" if h.backend == "torchsim" else "NO -> ASE calculator + ASE MD fallback (recorded)")
    del h
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
print("run_manifest.json written")
''')

md("## 4 — charge-ledger unit test (spec 12-2): the 9 pilot compositions must match the spec 4.2 table")
code(r'''# ============ CELL 4: CHARGE LEDGER ============
ledger = W.check_pilot_ledger()
print(ledger.to_string(index=False))
b2, m2 = W.stage2_compositions()
print(f"Stage II enumeration: {len(b2)} single-dopant baselines + {len(m2)} mixtures")
print("charge ledger OK")
''')

md("## 5 — structure generation test (spec 12-3): 9 compositions × 8 configs, invariants INV-1…9")
code(r'''# ============ CELL 5: STRUCTURE GENERATION ============
PILOT = W.pilot_compositions()
t0 = time.time()
META = W.ensure_generated(HOST, PILOT, "I", cfg)          # writes outputs/structures/generated/<comp>__<cid>.extxyz
gen = pd.DataFrame(list(META.values()))
print(f"{len(gen)} configurations on disk ({time.time()-t0:.0f}s), all invariants passed")
print(gen.groupby("composition_id")[["n_atoms", "n_vacancy", "n_vacancy_near", "n_interstitial_16e", "n_interstitial_other", "min_dist"]].agg(["min", "max"]).to_string())
print("48h pair histogram:", os.path.join(WORKDIR, "logs", "li48h_pair_hist.txt") if HOST.li_site_mode == "48h_pairs" else "(24g host: no pair histogram)")
''')

md("## 6 — kinisi validation (spec 12-4): synthetic Brownian trajectory with known D")
code(r'''# ============ CELL 6: KINISI CHECK ============
chk = W.synthetic_brownian_check(cfg)
print(chk)
assert chk["recovered"], "kinisi did not recover the known diffusion coefficient inside its 95% CI"
MANIFEST["kinisi_synthetic_check"] = chk
W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
''')

md("## Launcher helpers — one worker process per GPU (torch-sim autobatching inside each), CPU pool for kinisi")
code(r'''# ============ LAUNCHER HELPERS ============
def _write_jobs(name, jobs):
    p = os.path.join(WORKDIR, "outputs", "jobs", f"{name}.json")
    W.atomic_write_json(jobs, p); return p

def launch_gpu_task(task, model, jobs, name, n_workers=None):
    """Blocking: runs `jobs` of `task` for `model` on all GPUs, returns when every worker has exited."""
    if not jobs:
        print(f"[{task}:{model}] nothing to do"); return
    n_workers = n_workers or (1 if MOCK else N_GPUS)
    jp = _write_jobs(name, jobs)
    procs = []
    for k in range(n_workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        log = open(os.path.join(WORKDIR, "logs", f"{name}_w{k}.log"), "a")
        cmd = [sys.executable, os.path.join(WORKDIR, "sse_worker.py"), "--task", task, "--model", model, "--jobs", jp,
               "--config", os.path.join(WORKDIR, "outputs", "config.json"), "--worker-id", str(k), "--n-workers", str(n_workers)]
        procs.append((k, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    print(f"[{task}:{model}] {len(jobs)} jobs on {n_workers} worker(s) ({name}) ...", flush=True)
    t0 = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(20 if not MOCK else 2)
        if int(time.time() - t0) % 600 < 20:
            done = sum(1 for j in jobs if _job_done(task, model, j))
            print(f"   {done}/{len(jobs)} done | {(time.time()-t0)/60:.0f} min", flush=True)
    bad = [(k, p.returncode) for k, p in procs if p.returncode != 0]
    done = sum(1 for j in jobs if _job_done(task, model, j))
    print(f"[{task}:{model}] finished in {(time.time()-t0)/60:.1f} min | {done}/{len(jobs)} results | worker exit codes {[p.returncode for _, p in procs]}")
    assert not bad, f"worker(s) failed: {bad} -> see logs/{name}_w*.log"
    assert done == len(jobs), f"{len(jobs)-done} job results missing -> see logs/{name}_w*.log"

def _job_done(task, model, j):
    if task == "relax":
        return os.path.exists(W.relax_job_paths(cfg, model, j["key"])[1])
    if task == "md":
        return os.path.exists(os.path.join(W.md_dir(cfg, model, j["key"]), "result.json"))
    return os.path.exists(j["out"])

def launch_analysis(model, md_jobs, name):
    jobs = W.analyze_jobs_for(cfg, model, md_jobs)
    launch_gpu_task("analyze", model, jobs, name, n_workers=N_CPU_WORKERS)
    errs = [j["out"] for j in jobs if "error" in json.load(open(j["out"]))]
    if errs:
        print(f"   !! {len(errs)} analyses failed (kept as rows without sigma):", errs[:3])

def md_errors(model, jobs):
    out = []
    for j in jobs:
        r = json.load(open(os.path.join(W.md_dir(cfg, model, j["key"]), "result.json")))
        if "error" in r: out.append((j["key"], r["error"]))
    return out
print("launcher ready |", N_GPUS, "GPU worker(s) |", N_CPU_WORKERS, "CPU analysis worker(s)")
''')

md("## 7 — smoke test (spec 12-5): composition A, every model in CONFIG, 600 K, NPT 2 ps + NVT 5 ps; unwrapped trajectory saved")
code(r'''# ============ CELL 7: SMOKE TEST ============
SMOKE_NPT, SMOKE_NVT = (2.0, 5.0) if not MOCK else (0.1, 0.3)
for m in cfg["models"]:
    launch_gpu_task("relax", m, W.relax_jobs_for(cfg, META, ["A__c0"]), f"smoke_relax_{m}", n_workers=1)
    jobs = [dict(key="SMOKE__A__c0__600K", model=m, composition_id="A", config_id="c0", T=600.0,
                 input=W.relax_job_paths(cfg, m, "A__c0")[0], npt_ps=SMOKE_NPT, nvt_ps=SMOKE_NVT)]
    launch_gpu_task("md", m, jobs, f"smoke_md_{m}", n_workers=1)
    r = json.load(open(os.path.join(W.md_dir(cfg, m, jobs[0]["key"]), "result.json")))
    assert "error" not in r, f"smoke MD failed for {m}: {r['error']}"
    z = np.load(os.path.join(W.md_dir(cfg, m, jobs[0]["key"]), "nvt.npz"))
    assert z["positions"].shape[0] == r["n_frames"] and r["max_frame_jump_A"] < 0.5 * min(np.linalg.norm(z["cell"], axis=1)), "trajectory not unwrapped"
    print(f"{m}: {r['n_frames']} frames | V={r['volume_A3']:.0f} A^3 | max frame jump {r['max_frame_jump_A']:.2f} A | "
          f"Li MSD(last) {r['msd_li_last_A2']:.1f} A^2 | NVT {r['ns_per_day_nvt_per_system']:.2f} ns/day (1 system)")
    MANIFEST["models"][m]["smoke"] = {k: r[k] for k in ("thermostat", "barostat", "backend", "n_frames", "ns_per_day_nvt_per_system", "max_frame_jump_A")}
W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
''')

md("## 8 — throughput (spec 7.5-5): A at 600 K for 20 ps with the production batch size → `logs/throughput.txt`; user confirmation if the pilot exceeds the budget")
code(r'''# ============ CELL 8: THROUGHPUT ============
TP_NPT, TP_NVT = (1.0, 5.0) if not MOCK else (0.1, 0.3)
THROUGHPUT = {}
cids = [f"c{k}" for k in range(min(cfg["md_batch_size"], cfg["n_configs_pilot"]))]
for m in cfg["models"]:
    launch_gpu_task("relax", m, W.relax_jobs_for(cfg, META, [f"A__{c}" for c in cids]), f"tp_relax_{m}", n_workers=1)
    jobs = [dict(key=f"TP__A__{c}__600K__dt{cfg['md_timestep_fs']:g}fs__b{len(cids)}", model=m, composition_id="A", config_id=c, T=600.0, input=W.relax_job_paths(cfg, m, f"A__{c}")[0],
                 npt_ps=TP_NPT, nvt_ps=TP_NVT) for c in cids]
    launch_gpu_task("md", m, jobs, f"tp_md_{m}", n_workers=1)
    rs = [json.load(open(os.path.join(W.md_dir(cfg, m, j["key"]), "result.json"))) for j in jobs]
    assert all("error" not in r for r in rs), f"throughput MD failed for {m}: {[r['error'] for r in rs if 'error' in r][:2]}"
    THROUGHPUT[m] = dict(ns_per_day_per_gpu=float(rs[0]["ns_per_day_nvt_batch"]), ns_per_day_per_system=float(rs[0]["ns_per_day_nvt_per_system"]),
                         batch_size=len(jobs), n_atoms=416, timestep_fs=cfg["md_timestep_fs"], backend=rs[0]["backend"])
n_md_pilot = len(PILOT) * cfg["n_configs_md_pilot"] * len(cfg["md_temperatures_pilot"]) * len(cfg["models"])
lines = [f"# throughput {time.strftime('%Y-%m-%d %H:%M')} | 416 atoms | {cfg['md_timestep_fs']} fs | GPUs {N_GPUS}"]
worst = 0.0
for m, t in THROUGHPUT.items():
    est = W.estimate_gpu_hours(n_md_pilot // len(cfg["models"]), cfg["md_nvt_ps_pilot"], cfg["md_npt_ps"], t["ns_per_day_per_gpu"], N_GPUS)
    lines.append(f"{m}: {t['ns_per_day_per_gpu']:.2f} ns/day per GPU (batch {t['batch_size']}, {t['ns_per_day_per_system']:.2f} ns/day per system, {t['backend']})")
    lines.append(f"   pilot share ({est['n_md']} MD x {cfg['md_nvt_ps_pilot']+cfg['md_npt_ps']:.0f} ps): {est['gpu_hours']:.0f} GPU-h -> {est['wall_hours']:.0f} h wall on {N_GPUS} GPUs")
    worst += est["wall_hours"]
lines.append(f"TOTAL pilot MD (Stage I-M + Stage I, {len(cfg['models'])} model(s)): {worst:.1f} h wall on {N_GPUS} GPUs | budget {cfg['budget_hours']} h")
_inv = sum(1.0 / t["ns_per_day_per_gpu"] for t in THROUGHPUT.values()); _n_per_model = n_md_pilot // len(cfg["models"])
_nvt_fit = cfg["budget_hours"] * N_GPUS / 24.0 * 1000.0 / (_n_per_model * _inv) - cfg["md_npt_ps"]
lines.append(f"longest md_nvt_ps_pilot that fits the budget: {_nvt_fit:.0f} ps (current {cfg['md_nvt_ps_pilot']:.0f} ps; info only, not applied; relax + kinisi add ~0.5-1 h)")
open(os.path.join(WORKDIR, "logs", "throughput.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
MANIFEST["throughput"] = THROUGHPUT; MANIFEST["pilot_wall_hours_estimate"] = worst
W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
if worst > cfg["budget_hours"] and not cfg["confirm_over_budget"]:
    raise RuntimeError(f"Estimated pilot wall time {worst:.0f} h exceeds the {cfg['budget_hours']} h budget. "
                       "Spec 7.5-5: do NOT shorten md_nvt_ps_pilot silently. Report to the user; then either set "
                       "CONFIG['confirm_over_budget']=True or change md_nvt_ps_pilot with the user's approval and re-run.")
''')

md("## 9 — Stage I-M: model check (A, F; relax → NPT → NVT 600/750/900 K → kinisi; single model: G-a must pass)")
code(r'''# ============ CELL 9: STAGE I-M ============
TEMPS = [float(t) for t in cfg["md_temperatures_pilot"]]
IM_IDS = ["A", "F"]
RELAX_DF, MD_JOBS = {}, {}
if RUN_STAGE_IM:
    for m in cfg["models"]:
        keys = [k for k in META if META[k]["composition_id"] in IM_IDS]
        launch_gpu_task("relax", m, W.relax_jobs_for(cfg, META, keys), f"IM_relax_{m}")
        rel = W.relax_table(cfg, m, META, keys)
        print(f"{m}: {int(rel.converged.sum())}/{len(rel)} converged | e/atom range {rel.e_per_atom.min():.4f}..{rel.e_per_atom.max():.4f}")
        jobs = []
        for cid in IM_IDS:
            sel = W.select_md_configs(rel, cid, cfg["n_configs_md_pilot"])
            assert len(sel) >= 1, f"no converged configuration for {cid} ({m})"
            jobs += W.md_jobs_for(cfg, m, cid, sel, TEMPS, cfg["md_npt_ps"], cfg["md_nvt_ps_pilot"])
        launch_gpu_task("md", m, jobs, f"IM_md_{m}")
        errs = md_errors(m, jobs); print("   MD errors:", errs if errs else "none")
        launch_analysis(m, jobs, f"IM_an_{m}")
        RELAX_DF[m], MD_JOBS[m] = rel, jobs
    # tables (hull columns are filled in Stage I; here NaN)
    rs_IM = []
    for m in cfg["models"]:
        hd = RELAX_DF[m].copy(); hd["e_above_hull"] = np.nan; hd["e_above_hull_host"] = np.nan; hd["de_hull_vs_host"] = np.nan
        rs_IM.append(W.results_single_table(cfg, m, [c for c in PILOT if c["id"] in IM_IDS], hd, TEMPS, "I-M"))
    rs_IM = pd.concat(rs_IM); pt_IM = W.per_T_table(cfg, rs_IM); summ_IM = W.summary_table(cfg, pt_IM, PILOT, {})
    exp_ref = pd.read_csv(os.path.join(WORKDIR, cfg["exp_reference_csv"]))
    if exp_ref[["value_low", "value_high"]].isna().all().all():
        print("   exp_reference.csv is empty -> absolute comparison with experiment skipped (warning only)")
    MSEL, CHOICE = W.model_selection_table(cfg, summ_IM, pt_IM, RELAX_DF, {m: THROUGHPUT[m]["ns_per_day_per_gpu"] for m in cfg["models"]},
                                           {m: MANIFEST["models"][m]["tag"] for m in cfg["models"]}, exp_ref)
    MSEL.to_csv(os.path.join(WORKDIR, "outputs", "results_model_selection.csv"), index=False)
    print(MSEL[["model", "a_relaxed", "sigma_A_600K", "sigma_A_600K_ci_low", "sigma_A_600K_ci_high", "Ea_A", "ratio_F_A_600K", "ratio_F_A_600K_ci_low",
                "throughput_ns_per_day", "passes_ge_trend", "role"]].round(4).to_string())
    print("selection rule:", CHOICE["rule"])
    if CHOICE["main"] is None and cfg.get("continue_on_ga_fail"):
        CHOICE = dict(main=cfg["models"][0], contrast=None, rule="G-a FAILED; continued by user decision (continue_on_ga_fail)")
        MSEL["role"] = "main"; MSEL["selection_rule"] = CHOICE["rule"]
        MSEL.to_csv(os.path.join(WORKDIR, "outputs", "results_model_selection.csv"), index=False)
        print("!! WARNING: G-a failed -> continuing with", CHOICE["main"], "(user override)")
    MANIFEST["model_selection"] = CHOICE
    W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
    assert CHOICE["main"] is not None, "STOP (spec 6.5-3): no model reproduces sigma_F/sigma_A > 1 (CI lower bound). Report to the user."
    MAIN, CONTRAST = CHOICE["main"], CHOICE["contrast"]
else:
    MSEL = pd.read_csv(os.path.join(WORKDIR, "outputs", "results_model_selection.csv"))
    MAIN = MSEL[MSEL.role == "main"].model.iloc[0]; CONTRAST = next(iter(MSEL[MSEL.role == "contrast"].model), None)
print("main model:", MAIN, "| contrast model:", CONTRAST)
''')

md("## 10 — Stage I pilot: relax → hull references (MP, relaxed per model) → MD (9 comps × 2 configs × 3 T per model) → kinisi → tables → gates → figures 1–4 → **STOP for user confirmation**")
code(r'''# ============ CELL 10: STAGE I ============
if RUN_STAGE_I:
    assert W.file_md5(os.path.join(WORKDIR, "PILOT_CRITERIA.md")) == CRITERIA_MD5, "PILOT_CRITERIA.md changed during the run -> forbidden (spec rule 5)"
    ALL_KEYS = list(META)
    HULL_DF, MD_JOBS_I = {}, {}
    # ---- hull references: MP near-hull pool inside the union of the pilot chemical systems, relaxed with EACH model
    POOL_ELEMENTS = sorted(set(sum([W.chemsys_elements(c) for c in PILOT], [])))
    MOCK_HULL = MOCK and not os.environ.get("MP_API_KEY")
    if MOCK_HULL:
        print("MOCK without MP_API_KEY -> synthetic e_above_hull (uniform 0..0.04 eV/atom), MP not queried"); REF_IDS = []; POOL = None
    else:
        POOL = W.fetch_mp_near_hull_pool(cfg, POOL_ELEMENTS)
        REF_IDS = sorted(set(sum([W.assert_required_references(POOL, W.chemsys_elements(c)) for c in PILOT], [])))
        print(f"MP near-hull pool: {len(POOL['entries'])} entries in {POOL_ELEMENTS} | {len(REF_IDS)} reference phases for the pilot")
    for m in cfg["models"]:
        launch_gpu_task("relax", m, W.relax_jobs_for(cfg, META, ALL_KEYS), f"I_relax_{m}")
        rel = W.relax_table(cfg, m, META, ALL_KEYS)
        print(f"{m}: {int(rel.converged.sum())}/{len(rel)} configurations converged")
        if REF_IDS:
            launch_gpu_task("relax", m, W.prepare_reference_inputs(cfg, REF_IDS), f"I_refs_{m}")
            HULL_DF[m] = W.hull_table(cfg, m, rel, PILOT, POOL)
        else:
            _eh = np.random.default_rng(1).uniform(0, 0.04, len(rel)) if MOCK_HULL else np.full(len(rel), np.nan)
            HULL_DF[m] = rel.assign(e_above_hull=_eh, e_above_hull_host=np.nanmin(_eh) if MOCK_HULL else np.nan, de_hull_vs_host=_eh - (np.nanmin(_eh) if MOCK_HULL else np.nan))
        jobs = []
        for c in PILOT:
            sel = W.select_md_configs(rel, c["id"], cfg["n_configs_md_pilot"])
            assert sel, f"no converged configuration for {c['id']} ({m})"
            jobs += W.md_jobs_for(cfg, m, c["id"], sel, TEMPS, cfg["md_npt_ps"], cfg["md_nvt_ps_pilot"])
        launch_gpu_task("md", m, jobs, f"I_md_{m}")          # A/F runs from Stage I-M are reused (same keys)
        errs = md_errors(m, jobs); print("   MD errors:", errs if errs else "none")
        launch_analysis(m, jobs, f"I_an_{m}")
        MD_JOBS_I[m] = jobs
    RS = pd.concat([W.results_single_table(cfg, m, PILOT, HULL_DF[m], TEMPS, "I") for m in cfg["models"]])
    RS.to_csv(os.path.join(WORKDIR, "outputs", "results_single.csv"), index=False)
    PT = W.per_T_table(cfg, RS); PT.to_csv(os.path.join(WORKDIR, "outputs", "results_single_perT.csv"), index=False)
    SUMM = W.summary_table(cfg, PT, PILOT, HULL_DF); SUMM.to_csv(os.path.join(WORKDIR, "outputs", "results_single_summary.csv"), index=False)
    GATES = W.evaluate_gates(cfg, SUMM, PT, MAIN, CONTRAST)
    W.atomic_write_json(GATES, os.path.join(WORKDIR, "outputs", "gates.json"))
    W.make_figures(cfg, SUMM, MSEL, None, None, PT, GATES, os.path.join(WORKDIR, "outputs", "figures.pptx"))
    # ---- report
    print("\n=== Stage I summary (main model =", MAIN, ") ===")
    cols = ["composition_id", "n_li_per_uc", "sigma_600K", "sigma_600K_ci_low", "sigma_600K_ci_high", "ratio_vs_A_600K", "ratio_vs_A_600K_ci_low",
            "ratio_vs_A_600K_ci_high", "Ea", "Ea_ci_low", "Ea_ci_high", "de_hull_vs_host_mean", "notes"]   # 300 K extrapolation not reported (user decision 2026-09-28)
    for m in cfg["models"]:
        print(f"\n[{m}]"); print(SUMM[SUMM.model == m][cols].round(4).to_string(index=False))
    print("\n=== gates (PILOT_CRITERIA.md) ===")
    for k, v in GATES.items():
        print(f"{k}: {'PASS' if v.get('passed', v.get('effect_detected', None)) else 'FAIL/RECORD'} | {json.dumps({a: b for a, b in v.items() if a not in ('beta_linear',)}, default=str)[:300]}")
    if not GATES["G-a"]["passed"] and cfg.get("continue_on_ga_fail"):
        print("!! WARNING: G-a FAILED (sigma_F/sigma_A CI lower bound <= 1) - continuing by user decision")
    else:
        assert GATES["G-a"]["passed"], "GATE G-a FAILED: the main model does not reproduce sigma_F/sigma_A > 1 -> STOP and report to the user (spec 9)"
    # G-stab proposal (spec 8.4): the user confirms the threshold before Stage II
    mod = SUMM[(SUMM.model == MAIN) & SUMM.composition_id.isin(["D", "N2"])]
    prop = float(np.nanmax(mod.de_hull_vs_host_mean.values)) if len(mod) and np.isfinite(mod.de_hull_vs_host_mean.values).any() else float("nan")
    print(f"\nG-stab proposal: de_hull_vs_host of the moderate compositions D/N2 (main model) = {mod.de_hull_vs_host_mean.round(4).tolist()} eV/atom "
          f"-> proposed threshold max(0.03, {prop:.3f}) = {max(0.03, prop) if np.isfinite(prop) else 0.03:.3f} eV/atom (USER MUST CONFIRM, then set CONFIG['stab_threshold_eV'])")
    MANIFEST["stage1_done"] = time.strftime("%Y-%m-%d %H:%M:%S"); MANIFEST["gates"] = GATES
    W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
else:
    RS = pd.read_csv(os.path.join(WORKDIR, "outputs", "results_single.csv")); PT = pd.read_csv(os.path.join(WORKDIR, "outputs", "results_single_perT.csv"))
    SUMM = pd.read_csv(os.path.join(WORKDIR, "outputs", "results_single_summary.csv")); GATES = json.load(open(os.path.join(WORKDIR, "outputs", "gates.json")))
    HULL_DF = {}
print("\nSTAGE I COMPLETE. Report the summary table, gates.json and figures.pptx slides 1-4 to the user; "
      "Stage II starts only with RUN_STAGE_II=True and a user-confirmed CONFIG['stab_threshold_eV'].")
''')

md("## 11 — Stage II (main model): multi-dopant synergy screening → group top-5 → long MD (new seed) → contrast rescoring")
code(r'''# ============ CELL 11: STAGE II ============
if RUN_STAGE_II:
    assert cfg.get("stab_threshold_eV") is not None, "Stage II refused: CONFIG['stab_threshold_eV'] (G-stab) is not set - user decision required after Stage I"
    assert W.file_md5(os.path.join(WORKDIR, "PILOT_CRITERIA.md")) == CRITERIA_MD5, "PILOT_CRITERIA.md changed -> forbidden"
    BASE, MIX = W.stage2_compositions()
    MOCK_HULL = MOCK and not os.environ.get("MP_API_KEY")
    if MOCK:
        MIX = MIX[::49][:12]; print("MOCK: Stage II restricted to", len(MIX), "mixtures")
    T2 = float(cfg["md_stage2_short_T"]); SHORT = cfg["md_nvt_ps_stage2_short"]
    # ---- structures + relax (main model)
    META2 = W.ensure_generated(HOST, BASE + MIX, "II", cfg)
    KEYS2 = list(META2)
    launch_gpu_task("relax", MAIN, W.relax_jobs_for(cfg, META2, KEYS2), f"II_relax_{MAIN}")
    REL2 = W.relax_table(cfg, MAIN, META2, KEYS2)
    print(f"Stage II relax ({MAIN}): {int(REL2.converged.sum())}/{len(REL2)} converged")
    # ---- hull with the same model: references for the union pool of all Stage II chemical systems
    POOL2_EL = sorted(set(sum([W.chemsys_elements(c) for c in BASE + MIX], [])))
    relA = W.relax_table(cfg, MAIN, META, [k for k in META if META[k]["composition_id"] == "A"])
    def _hull_stage2(model, rel_df, comps):
        """Own PhaseDiagram per model (references relaxed with that model); synthetic values in MOCK without MP."""
        if MOCK_HULL:
            eh = np.random.default_rng(abs(hash(model)) % 2**31).uniform(0, 0.04, len(rel_df))
            return rel_df.assign(e_above_hull=eh, e_above_hull_host=0.01, de_hull_vs_host=eh - 0.01)
        launch_gpu_task("relax", model, W.prepare_reference_inputs(cfg, REF2), f"II_refs_{model}")
        return W.hull_table(cfg, model, rel_df, comps, POOL2)
    if not MOCK_HULL:
        POOL2 = W.fetch_mp_near_hull_pool(cfg, POOL2_EL)
        REF2 = sorted(set(sum([W.assert_required_references(POOL2, W.chemsys_elements(c)) for c in BASE + MIX], [])))
        print(f"MP near-hull pool for Stage II: {len(POOL2['entries'])} entries, {len(REF2)} reference phases")
    HULL2 = _hull_stage2(MAIN, pd.concat([REL2, relA]), BASE + MIX + [PILOT[0]])
    host_eh = float(HULL2.e_above_hull_host.iloc[0])
    STAB = W.stage2_stability_table(cfg, MAIN, MIX, HULL2, host_eh)
    print(f"stability filter (threshold {cfg['stab_threshold_eV']} eV/atom): {int(STAB.pass_stability.sum())}/{len(STAB)} mixtures pass")
    # ---- short MD: baselines + passing mixtures (2 lowest-energy configs, 600 K, 500 ps)
    short_ids = [b["id"] for b in BASE] + list(STAB[STAB.pass_stability].composition_id)
    jobs_s = []
    for cid in short_ids:
        sel = W.select_md_configs(REL2, cid, cfg["n_configs_md_stage2_short"])
        if sel: jobs_s += W.md_jobs_for(cfg, MAIN, cid, sel, [T2], cfg["md_npt_ps"], SHORT)
    launch_gpu_task("md", MAIN, jobs_s, f"II_short_md_{MAIN}"); launch_analysis(MAIN, jobs_s, f"II_short_an_{MAIN}")
    RS2 = W.results_single_table(cfg, MAIN, BASE + MIX, HULL2, [T2], "II-short")
    PT2 = W.per_T_table(cfg, RS2)
    use_interp = not GATES["G-lin"]["passed"]
    SSYN = W.s_syn_table(cfg, PT2, MAIN, MIX, BASE, T2, use_interp, "short")
    MULTI = STAB.merge(SSYN.drop(columns=["model", "T"]), on="composition_id", how="left")
    MULTI = W.select_top_by_group(MULTI, "S_syn_short", n=5) if MULTI.S_syn_short.notna().any() else MULTI.assign(selected_for_long=False)
    MULTI = STAB[["composition_id"]].merge(MULTI, on="composition_id", how="left")
    MULTI["selected_for_long"] = MULTI.selected_for_long.fillna(False).astype(bool)
    # single-dopant baselines go to results_single.csv too (spec 7.7)
    pd.concat([pd.read_csv(os.path.join(WORKDIR, "outputs", "results_single.csv")), RS2[RS2.composition_id.isin([b["id"] for b in BASE])]]).drop_duplicates(
        subset=["model", "composition_id", "config_id", "T"]).to_csv(os.path.join(WORKDIR, "outputs", "results_single.csv"), index=False)
    # ---- long MD with a NEW seed for the selected mixtures and their same-Ge baselines (winner's curse, spec 8.7)
    sel_ids = list(MULTI[MULTI.selected_for_long].composition_id)
    assert sel_ids, "no mixture passed the stability filter / short MD -> revisit CONFIG['stab_threshold_eV'] with the user"
    long_base_ids = sorted({b for cid in sel_ids for b, _ in W.s_syn_denominator_terms(next(c for c in MIX if c["id"] == cid))})
    LONG_COMPS = [c for c in MIX if c["id"] in sel_ids] + [b for b in BASE if b["id"] in long_base_ids]
    print(f"long MD: {len(sel_ids)} selected mixtures + {len(long_base_ids)} baselines (new seed, {cfg['n_configs_md_pilot']} configs x {TEMPS} K x {cfg['md_nvt_ps_pilot']} ps)")
    METAL = W.ensure_generated(HOST, LONG_COMPS, "I", cfg, salt="long", cid_prefix="l")
    KEYSL = list(METAL)
    def _long_stage(model):
        launch_gpu_task("relax", model, W.relax_jobs_for(cfg, METAL, KEYSL, salt="long"), f"II_long_relax_{model}")
        relA_m = relA if model == MAIN else W.relax_table(cfg, model, META, [k for k in META if META[k]["composition_id"] == "A"])
        rel = W.relax_table(cfg, model, METAL, KEYSL)
        hull = _hull_stage2(model, pd.concat([rel, relA_m]), LONG_COMPS + [PILOT[0]])
        jobs = []
        for c in LONG_COMPS:
            sel = W.select_md_configs(rel, c["id"], cfg["n_configs_md_pilot"])
            if sel: jobs += W.md_jobs_for(cfg, model, c["id"], sel, TEMPS, cfg["md_npt_ps"], cfg["md_nvt_ps_pilot"])
        launch_gpu_task("md", model, jobs, f"II_long_md_{model}"); launch_analysis(model, jobs, f"II_long_an_{model}")
        rs = W.results_single_table(cfg, model, LONG_COMPS, hull, TEMPS, "II-long")
        pt = W.per_T_table(cfg, rs); summ = W.summary_table(cfg, pt, LONG_COMPS + PILOT, {model: hull})
        return rs, pt, summ
    RSL, PTL, SUMML = _long_stage(MAIN)
    SSYN_L = W.s_syn_table(cfg, PTL, MAIN, [c for c in MIX if c["id"] in sel_ids], BASE, 600.0, use_interp, "long")
    MULTI = MULTI.merge(SSYN_L.drop(columns=["model", "T", "denominator"], errors="ignore"), on="composition_id", how="left")
    MULTI = MULTI.merge(SUMML[SUMML.model == MAIN][["composition_id", "Ea", "Ea_ci_low", "Ea_ci_high", "sigma_300K", "sigma_300K_ci_low", "sigma_300K_ci_high"]],
                        on="composition_id", how="left")
    # ---- contrast model rescoring (relax from scratch, own hull, same MD protocol; spec 8.8)
    RSC, PTC, SUMMC = _long_stage(CONTRAST)
    SSYN_C = W.s_syn_table(cfg, PTC, CONTRAST, [c for c in MIX if c["id"] in sel_ids], BASE, 600.0, use_interp, "contrast")
    MULTI = MULTI.merge(SSYN_C.drop(columns=["model", "T", "denominator"], errors="ignore"), on="composition_id", how="left")
    MULTI["sign_agree"] = (MULTI.S_syn_long - 1) * (MULTI.S_syn_contrast - 1) > 0
    MULTI["spearman_in_group"] = np.nan
    for grp, g in MULTI[MULTI.selected_for_long].groupby("li_group"):
        MULTI.loc[g.index, "spearman_in_group"] = W.spearman(g.S_syn_long.values.astype(float), g.S_syn_contrast.values.astype(float))
    MULTI["main_model"] = MAIN; MULTI["contrast_model"] = CONTRAST; MULTI["S_syn_denominator"] = "interpolated" if use_interp else "geometric_mean"
    MULTI.to_csv(os.path.join(WORKDIR, "outputs", "results_multi.csv"), index=False)
    pd.concat([RS2, RSL, RSC]).to_csv(os.path.join(WORKDIR, "outputs", "results_multi_configs.csv"), index=False)
    TOP = MULTI[MULTI.selected_for_long].copy()
    TOP["confirmed"] = (TOP.S_syn_long_ci_low > 1) & TOP.sign_agree.astype(bool)
    TOP = TOP.sort_values(["li_group", "rank_in_group"])
    TOP.to_csv(os.path.join(WORKDIR, "outputs", "top_candidates.csv"), index=False)
    W.make_figures(cfg, SUMM, MSEL, MULTI, TOP, PT, GATES, os.path.join(WORKDIR, "outputs", "figures.pptx"))
    print(TOP[["composition_id", "li_group", "valence_combo", "flag_redox_risk", "de_hull_corr_vs_host", "S_syn_short", "S_syn_long", "S_syn_long_ci_low",
               "S_syn_long_ci_high", "S_syn_contrast", "sign_agree", "confirmed"]].round(3).to_string(index=False))
    MANIFEST["stage2_done"] = time.strftime("%Y-%m-%d %H:%M:%S")
    W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
else:
    print("RUN_STAGE_II=False -> Stage II skipped (user confirmation of Stage I gates and G-stab threshold required)")
''')

md("## 12 — DFT hand-off export (structures only; spec 11)")
code(r'''# ============ CELL 12: EXPORT ============
if RUN_EXPORT:
    relaxed_keys = {c["id"]: [k for k in META if META[k]["composition_id"] == c["id"]] for c in PILOT}
    md_keys = {}
    for c in PILOT:
        for T in TEMPS:
            for k in relaxed_keys[c["id"]]:
                key = W.md_key(c["id"], META[k]["config_id"], T)
                if os.path.exists(os.path.join(W.md_dir(cfg, MAIN, key), "nvt.npz")):
                    md_keys.setdefault(c["id"], []).append((key, T)); break
    top_ids, ref_by_model = [], {}
    if os.path.exists(os.path.join(WORKDIR, "outputs", "top_candidates.csv")):
        top_ids = list(pd.read_csv(os.path.join(WORKDIR, "outputs", "top_candidates.csv")).composition_id)
        for m in cfg["models"]:
            for c in top_ids:
                ks = [k for k in (globals().get("METAL") or {}) if METAL[k]["composition_id"] == c]
                relaxed_keys[c] = ks
                for k in ks:
                    key = W.md_key(c, METAL[k]["config_id"], 600.0)
                    if os.path.exists(os.path.join(W.md_dir(cfg, MAIN, key), "nvt.npz")):
                        md_keys.setdefault(c, []).append((key, 600.0)); break
    for m in cfg["models"]:
        ref_by_model[m] = sorted({os.path.basename(p)[5:-5] for p in glob.glob(os.path.join(WORKDIR, "outputs", "structures", "relaxed", m, "REF__*.json"))})
    for must in ("D", "M2", "L2"):
        assert must in relaxed_keys, f"pilot composition {must} must be exported (spec 11)"
    out = W.export_dft_handoff(cfg, MAIN, [c["id"] for c in PILOT], top_ids, relaxed_keys, md_keys, ref_by_model, MANIFEST)
    print("DFT hand-off written to", out)
''')

md("## 13 — finish: manifest, DONE sentinel, pod stop (`CONFIG['auto_terminate']`)")
code(r'''# ============ CELL 13: FINISH ============
assert W.file_md5(os.path.join(WORKDIR, "PILOT_CRITERIA.md")) == CRITERIA_MD5, "PILOT_CRITERIA.md changed during the run"
MANIFEST["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
MANIFEST["outputs"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WORKDIR, "outputs", "*.csv")) + glob.glob(os.path.join(WORKDIR, "outputs", "*.pptx")))
W.atomic_write_json(MANIFEST, os.path.join(WORKDIR, "outputs", "run_manifest.json"))
for f in ("results_model_selection.csv", "results_single.csv", "results_single_summary.csv", "figures.pptx", "run_manifest.json"):
    print(f"{f:32s} {'OK' if os.path.exists(os.path.join(WORKDIR, 'outputs', f)) else 'MISSING'}")
open(os.path.join(WORKDIR, "outputs", "DONE"), "w").write(json.dumps({"t": time.time(), "stage2": RUN_STAGE_II}))
pod = os.environ.get("RUNPOD_POD_ID")
if CONFIG["auto_terminate"] and pod:
    time.sleep(5)
    r = subprocess.run(["runpodctl", "stop", "pod", pod], capture_output=True, text=True)
    print("runpodctl stop pod:", r.returncode, (r.stdout or r.stderr).strip()[:200])
    if r.returncode != 0:
        print("  !! stop failed; the watchdog retries in ~5 min (needs runpodctl or RUNPOD_API_KEY)")
else:
    print("auto_terminate off or not on RunPod -> pod left running")
''')

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}})
import os
nbf.write(nb, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sse_screening.ipynb"))
print("notebook written:", len(cells), "cells")
