"""bench_models.py - uMLIP adapters for the 12-model benchmark (MODEL_BENCHMARK_SPEC).

One entry per participant M01-M12 plus a mock for wiring tests.  Each adapter knows how to
install itself (pip specs, isolation needs) and how to build an ASE calculator.  Every model
runs through the SAME relax path (ASE FIRE + FrechetCellFilter, bench_anchor_runner.py), so
the calculator is the only model-specific code.

Uncertain call signatures (M06, M09-M11) try several known entry points and raise AdapterError
with everything they tried: the runner records that as status='adapter_error' instead of dying,
and the registry marks these models unverified.  Do NOT silently drop a model that fails here -
MODEL_BENCHMARK_SPEC section 10-6 requires the failure to appear in the results table.
"""
import os

BENCH_ROOT = os.environ.get("HEO_BENCH_ROOT", "/workspace/heo_bench")
CKPT_DIR   = os.environ.get("HEO_BENCH_CKPTS", f"{BENCH_ROOT}/model_ckpts")


class AdapterError(RuntimeError):
    """Model could not be loaded; the message says what was tried and what to check."""


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _hf_download(repo_id, filename, revision=None):
    os.makedirs(CKPT_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                           cache_dir=CKPT_DIR, token=os.environ.get("HF_TOKEN") or None)


def _try(errors, what, fn):
    """Run one candidate entry point; collect the failure and fall through."""
    try:
        return fn()
    except Exception as e:                       # noqa: BLE001 - every failure goes into the report
        errors.append(f"{what}: {type(e).__name__}: {str(e)[:200]}")
        return None


# ============================================================ builders
def build_orb_mpa():
    return _build_orb("orb_v3_conservative_inf_mpa")

def build_orb_omat():
    return _build_orb("orb_v3_conservative_inf_omat")

def _build_orb(name):
    dev = _device()
    import importlib
    import pkgutil
    import orb_models
    from orb_models.forcefield import pretrained
    loaded = getattr(pretrained, name)(device=dev)
    model = loaded[0] if isinstance(loaded, tuple) else loaded   # some versions return (model, adapter)
    # The ORBCalculator module has moved between orb-models releases
    # (2026-09: 'orb_models.forcefield.calculator' is gone in the latest wheel).
    # Try the known homes first, then scan the whole package for any module defining it.
    errors = []
    candidates = ["orb_models.forcefield.calculator", "orb_models.ase.calculator",
                  "orb_models.calculator", "orb_models.ase"]
    try:
        candidates += [m.name for m in pkgutil.walk_packages(orb_models.__path__, prefix="orb_models.")
                       if "calc" in m.name.rsplit(".", 1)[-1] and m.name not in candidates]
    except Exception as e:                        # noqa: BLE001
        errors.append(f"package scan failed: {type(e).__name__}")
    for cand in candidates:
        try:
            mod = importlib.import_module(cand)
        except Exception as e:                    # noqa: BLE001
            errors.append(f"{cand}: {type(e).__name__}")
            continue
        cls = getattr(mod, "ORBCalculator", None)
        if cls is not None:
            print(f"[orb] ORBCalculator found in {cand}", flush=True)
            return cls(model, device=dev)
        errors.append(f"{cand}: imported but no ORBCalculator attribute")
    raise AdapterError(f"ORBCalculator not found in installed orb-models "
                       f"(version {getattr(orb_models, '__version__', '?')}).  Either pin the version the "
                       f"repo README documents (pip install 'orb-models==<ver>') or point this builder at "
                       f"the new class.  Tried:\n  " + "\n  ".join(errors))

def build_mace_mpa():
    from mace.calculators import mace_mp
    return mace_mp(model="medium-mpa-0", device=_device(), default_dtype="float64")

def build_sevennet_mf_ompa():
    dev = _device(); errors = []
    def _a():
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model="7net-mf-ompa", modal="mpa", device=dev)
    def _b():                                    # older sevenn: keyword 'model_type'/'modal' drift
        from sevenn.sevennet_calculator import SevenNetCalculator
        return SevenNetCalculator("7net-mf-ompa", modal="mpa", device=dev)
    for what, fn in (("sevenn.calculator", _a), ("sevenn.sevennet_calculator", _b)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError("SevenNet-MF-ompa load failed (modal='mpa' is REQUIRED - the omat head is the "
                       "wrong currency).  Tried:\n  " + "\n  ".join(errors))

def build_esen_30m_oam():
    dev = _device(); errors = []
    def _load(unit_name):
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        unit = pretrained_mlip.get_predict_unit(unit_name, device=dev)
        return FAIRChemCalculator(unit, task_name="omat")
    # 2026-09-17: the installed fairchem-core v2 registry carries only UMA + esen-*-omol names -
    # no OAM-era eSEN.  HEO_ESEN_MODEL substitutes the participant (e.g. "uma-m-1p1", HF-gated:
    # HF_TOKEN + license acceptance needed); the substitution is recorded in the model tag.
    prefer = os.environ.get("HEO_ESEN_MODEL", "")
    names = ([prefer] if prefer else []) + ["esen-30m-oam", "eSEN-30M-OAM", "esen_30m_oam"]
    for name in names:
        c = _try(errors, f"get_predict_unit({name!r})", lambda n=name: _load(n))
        if c is not None: return c
    raise AdapterError("eSEN-30M-OAM load failed: the installed fairchem-core registry has no OAM-era "
                       "eSEN checkpoint.  Set HEO_ESEN_MODEL to one of the names the error below lists "
                       "(a UMA checkpoint substitutes the M05 participant and needs HF_TOKEN + license "
                       "acceptance), or leave M05 as not-runnable.  Tried:\n  " + "\n  ".join(errors))


def _esen_tag():
    m = os.environ.get("HEO_ESEN_MODEL", "")
    return f"bench-{m}" if m else "bench-esen-30m-oam"

def build_equiformer_v3_oam():
    """Vendored-fairchem route (repo README): the checkpoint should load through the fairchem
    machinery installed from packages/fairchem-core.  The DeNS-OAM .pt is discovered on
    hf.co/mirror-physics/equiformer_v3 unless HEO_EQV3_CKPT points at a local file."""
    dev = _device(); errors = []
    ckpt = os.environ.get("HEO_EQV3_CKPT", "")
    if not ckpt:
        from huggingface_hub import list_repo_files
        files = [f for f in list_repo_files("mirror-physics/equiformer_v3")
                 if f.endswith((".pt", ".ckpt", ".pth"))]
        # actual files on the mirror repo (checked 2026-09-18): checkpoint/{mptrj_gradient,
        # omat24-mptrj-salex_gradient, omat24_direct, omat24_gradient}.pt -- no 'oam'/'dens' in the
        # names.  OAM currency == OMat24+MPtrj+sAlex, and 'gradient' == conservative forces.
        low = [(f, f.lower()) for f in files]
        pref = ([f for f, l in low if "salex" in l and "gradient" in l]
                or [f for f, l in low if "oam" in l]
                or [f for f, l in low if "gradient" in l]
                or files)
        if not pref:
            raise AdapterError("no checkpoint files found in hf.co/mirror-physics/equiformer_v3 - "
                               "download one manually and set HEO_EQV3_CKPT")
        name = sorted(pref)[0]
        print(f"[eqv3] checkpoint: {name} (of {len(files)} files on the mirror repo)", flush=True)
        ckpt = _hf_download("mirror-physics/equiformer_v3", name)
    def _ocp():
        from fairchem.core import OCPCalculator
        return OCPCalculator(checkpoint_path=ckpt, cpu=(dev == "cpu"))
    def _ocp_old():
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        return OCPCalculator(checkpoint_path=ckpt, cpu=(dev == "cpu"))
    def _fc():
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        unit = pretrained_mlip.load_predict_unit(ckpt, device=dev)
        return FAIRChemCalculator(unit, task_name="omat")
    for what, fn in (("fairchem.core.OCPCalculator", _ocp),
                     ("fairchem ase_utils.OCPCalculator", _ocp_old),
                     ("FAIRChemCalculator(load_predict_unit)", _fc)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError("EquiformerV3+DeNS-OAM: the vendored fairchem loaded but no calculator route "
                       "accepted the checkpoint.  Possible missing deps from experimental/env/"
                       "conda_requirements.txt (add them to pip_steps), or a different loader API - "
                       "check packages/fairchem-core in the repo.  Tried:\n  " + "\n  ".join(errors))

def build_mattersim():
    from mattersim.forcefield import MatterSimCalculator
    return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=_device())

def build_chgnet():
    dev = _device(); errors = []
    def _a():
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device=dev)
    def _b():
        from chgnet.model import CHGNetCalculator
        return CHGNetCalculator(use_device=dev)
    for what, fn in (("chgnet.model.dynamics", _a), ("chgnet.model", _b)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError("CHGNet load failed.  Tried:\n  " + "\n  ".join(errors))

def build_equflashv2_oam():
    """2026-09-18, verified from the repo source: the ASE entry point is
    GGNN.common.calculator.UCalculator (subclass of fairchem v1 OCPCalculator; no `device` kwarg,
    GPU is selected with cpu=False).  fairchem.core==1.10.0 comes from requirements-no-deps.txt."""
    dev = _device(); errors = []
    ckpt = os.environ.get("HEO_EQUFLASH_CKPT", "")
    if not ckpt:
        raise AdapterError("EquFlashV2-45M-OAM needs HEO_EQUFLASH_CKPT=<path to the Figshare checkpoint>. "
                           "Download link: the model page matbench-discovery.materialsproject.org/models/"
                           "equflashv2-45m-oam ('Model files' section); code: github.com/SamsungDS/GGNN.")
    def _u():
        from GGNN.common.calculator import UCalculator
        return UCalculator(checkpoint_path=ckpt, cpu=(dev == "cpu"))
    c = _try(errors, "GGNN.common.calculator.UCalculator", _u)
    if c is not None: return c
    raise AdapterError("EquFlashV2 (GGNN) UCalculator failed - if the repo layout changed, re-check "
                       "GGNN/common/calculator.py on github.com/SamsungDS/GGNN.  Tried:\n  "
                       + "\n  ".join(errors))

def build_prophet_oame():
    dev = _device(); errors = []
    ckpt = os.environ.get("HEO_PROPHET_CKPT", "")
    if not ckpt:
        ckpt = _hf_download("kairosmaterial/prophet", "prophet-oame-mbd.pt")
    def _a():
        from prophet.calculator import KairosCalculator
        return KairosCalculator(ckpt, device=dev)
    def _b():
        from prophet_mlip.calculator import KairosCalculator
        return KairosCalculator(ckpt, device=dev)
    def _c():
        from kairos.calculator import KairosCalculator
        return KairosCalculator(ckpt, device=dev)
    for what, fn in (("prophet.calculator", _a), ("prophet_mlip.calculator", _b), ("kairos.calculator", _c)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError("Prophet-OAME-MBD: KairosCalculator import failed - check the package layout of "
                       "github.com/kairosmaterial/prophet and extend this builder.  Tried:\n  "
                       + "\n  ".join(errors))

def build_tece_oam_rra():
    dev = _device(); errors = []
    ckpt = os.environ.get("HEO_TECE_CKPT", "")
    if not ckpt:
        ckpt = _hf_download("xvzemin/tace-foundations", "TECE-OAM-RRA-1.0.pt")
    def _a():
        from tace.calculator import TACECalculator
        return TACECalculator(model=ckpt, device=dev)
    def _b():
        from tace.ase_interface import TACECalculator
        return TACECalculator(ckpt, device=dev)
    for what, fn in (("tace.calculator", _a), ("tace.ase_interface", _b)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError("TECE-OAM-RRA-1.0: TACE calculator import failed.  This model needs its own venv "
                       "(python 3.13, torch 2.13, openequivariance - see models_registry.yml).  Docs: "
                       "tace.readthedocs.io.  Tried:\n  " + "\n  ".join(errors))

def build_pet_oam():
    dev = _device(); errors = []
    # 2026-09-18: upet's registry names sizes explicitly - 'pet-oam' alone is invalid,
    # 'pet-oam-l' is the large OAM checkpoint (confirmed from upet's own error listing).
    name = os.environ.get("HEO_PET_MODEL", "pet-oam-l")
    def _a():
        from upet.calculator import UPETCalculator
        return UPETCalculator(model=name, device=dev)
    def _b():
        from upet import UPETCalculator
        return UPETCalculator(model=name, device=dev)
    def _c():                                    # PET-MAD-era layout, PBEsol currency -> contrast only
        from pet_mad.calculator import PETMADCalculator
        return PETMADCalculator(version="latest", device=dev)
    for what, fn in (("upet.calculator", _a), ("upet", _b), ("pet_mad.calculator (PBEsol fallback!)", _c)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError("PET-OAM load failed.  upet (lab-cosmo/upet) ships PET-OAM/PET-MAD checkpoints on HF; "
                       "check the calculator class name in its README and extend this builder.  Tried:\n  "
                       + "\n  ".join(errors))

def build_mock():
    return None       # the runner branches to heo_worker.relax_mock for this adapter


# ============================================================ registry (mirrors models_registry.yml)
# isolated=True: the model pins its own torch/python -> full venv with own torch, no system site-packages.
ADAPTERS = {
    "orb_mpa": dict(
        model_id="M01", name="ORB-v3 conservative mpa", tag="bench-orb-v3-mpa",
        # <0.6 pin (2026-09-17): the latest orb-models wheel dropped/moved
        # orb_models.forcefield.calculator.ORBCalculator; 0.5.x is the documented orb-v3 API.
        pip=["orb-models<0.6"], isolated=False, python=None,
        currency="MP (MPtrj+sAlex)", role="baseline (production main model)",
        build=build_orb_mpa),
    "orb_omat": dict(
        model_id="M02", name="ORB-v3 conservative omat", tag="bench-orb-v3-omat",
        pip=["orb-models<0.6"], isolated=False, python=None,   # same pin as orb_mpa
        currency="OMat24 (PBE, no +U) - CURRENCY CONTRAST", role="currency control",
        build=build_orb_omat),
    "mace_mpa": dict(
        model_id="M03", name="MACE-MPA-0 medium", tag="bench-mace-mpa-0",
        pip=["mace-torch"], isolated=False, python=None,
        currency="MP (MPtrj+sAlex)", role="current contrast model",
        build=build_mace_mpa),
    "sevennet_mf_ompa": dict(
        model_id="M04", name="SevenNet-MF-ompa (mpa head)", tag="bench-7net-mf-ompa-mpa",
        pip=["sevenn"], isolated=False, python=None,
        currency="MP (mpa modal head)", role="equivariant multi-fidelity",
        build=build_sevennet_mf_ompa),
    "esen_30m_oam": dict(
        model_id="M05", name="eSEN-30M-OAM (HEO_ESEN_MODEL로 대체 가능)", tag=_esen_tag(),
        pip=["fairchem-core>=2"], isolated=False, python=None,
        currency="MP finish (OMat pretrain -> MPtrj+sAlex)", role="leaderboard upper tier",
        build=build_esen_30m_oam),
    "equiformer_v3_oam": dict(
        model_id="M06", name="EquiformerV3+DeNS-OAM", tag="bench-eqv3-dens-oam",
        # The repo VENDORS a modified fairchem-core in packages/fairchem-core (README env setup,
        # 2026-09-18): install THAT subdirectory, on the exact torch/PyG stack the README pins.
        # conda_requirements.txt is skipped - if imports fail on a missing dep, add it to a step.
        pip=[],
        pip_steps=[
            ["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
             "--index-url", "https://download.pytorch.org/whl/cu128"],
            ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
             "-f", "https://data.pyg.org/whl/torch-2.7.0+cu128.html"],
            ["torch_geometric"],
            ["git+https://github.com/atomicarchitects/equiformer_v3#subdirectory=packages/fairchem-core"],
            ["huggingface_hub"],
        ],
        isolated=True, python="3.11",
        currency="MP finish (claimed)", role="leaderboard F1 top (vendored-fairchem route)",
        build=build_equiformer_v3_oam),
    "mattersim": dict(
        model_id="M07", name="MatterSim-v1 (5M)", tag="bench-mattersim-v1-5m",
        pip=["mattersim"], isolated=False, python=None,
        currency="MP-compatible (claimed; +U handling to verify)", role="independent data lineage",
        build=build_mattersim),
    "chgnet": dict(
        model_id="M08", name="CHGNet", tag="bench-chgnet",
        pip=["chgnet"], isolated=False, python=None,
        currency="MP (MPtrj)", role="only magmom-aware participant",
        build=build_chgnet),
    "equflashv2_oam": dict(
        model_id="M09", name="EquFlashV2-45M-OAM", tag="bench-equflashv2-45m-oam",
        # official repo requirements (user-supplied 2026-09-18): torch 2.9.1+cu126 + pinned PyG wheels
        # + cuequivariance 0.6.0 trio + two --no-deps extras, then the GGNN package itself.
        # GGNN goes in --no-deps too so its metadata cannot upgrade the pinned torch.
        pip=[],
        pip_steps=[
            ["--extra-index-url", "https://download.pytorch.org/whl/cu126", "torch==2.9.1+cu126"],
            ["--find-links", "https://data.pyg.org/whl/torch-2.9.1+cu126.html",
             "torch_scatter==2.1.2+pt29cu126", "torch_sparse==0.6.18+pt29cu126"],
            ["torch-geometric==2.6.1", "scipy==1.16.1", "ase==3.26.0", "e3nn==0.5.6",
             "huggingface-hub", "hydra-core", "lmdb==1.6.2", "numba", "numpy==1.26.4",
             "orjson", "pydantic", "pymatgen==2025.10.7", "pyyaml", "requests",
             "submitit==1.5.3", "tensorboard", "torchtnt", "tqdm", "wandb", "wheel",
             "scikit-learn"],
            ["cuequivariance-torch==0.6.0", "cuequivariance==0.6.0",
             "cuequivariance-ops-torch-cu12==0.6.0",
             "ase_db_backends==0.11.0", "nvalchemi-toolkit-ops==0.3.0"],
            # repo requirements-no-deps.txt == exactly this one line (verified 2026-09-18)
            ["--no-deps", "fairchem.core==1.10.0"],
            # pinned tarball, NOT git+: the flashTP submodule URL points at Samsung's internal
            # git server, so a recursive git clone fails; the tarball has no submodules and
            # EquFlashV2's model code never imports flashTP (verified in the source).
            ["--no-deps",
             "https://github.com/SamsungDS/GGNN/archive/a25dfa4735582ade72b3e3a2dd33074c43d9fe73.tar.gz"],
        ],
        isolated=True, python="3.12",
        currency="MP (MPtrj+OMat24+sAlex)", role="equivariant + FlashTP speed (F1 0.929)",
        build=build_equflashv2_oam),
    "prophet_oame": dict(
        model_id="M10", name="Prophet-OAME-MBD", tag="bench-prophet-oame-mbd",
        # openequivariance: prophet.calculator imports it but the git package does not declare it
        pip=["prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"
             "@dcfdc0b143978652d573630f1d2a6ce09e0c610a", "openequivariance", "huggingface_hub"],
        isolated=True, python=None,
        currency="MP (OMat24+ELEMENTA pretrain -> MPtrj+sAlex finish)", role="new upper tier",
        build=build_prophet_oame),
    "tece_oam_rra": dict(
        model_id="M11", name="TECE-OAM-RRA-1.0", tag="bench-tece-oam-rra-1.0",
        # tace is on PyPI (repo README: `pip install tace`) - cleaner than the git ref
        pip=["tace", "openequivariance==0.6.4", "huggingface_hub"],
        isolated=True, python="3.13", torch="torch==2.13.*",
        currency="MP finish (-oam suffix; verify on the card)", role="TACE lineage upper tier",
        build=build_tece_oam_rra),
    "pet_oam": dict(
        model_id="M12", name="PET-OAM (upet)", tag="bench-pet-oam",
        pip=["upet"], isolated=False, python=None,
        currency="MP finish (OAM); PET-MAD fallback is PBEsol -> contrast only", role="PET lineage",
        build=build_pet_oam),
    "mock": dict(
        model_id="MOCK", name="mock (heo_worker.relax_mock)", tag="bench-mock-v2",
        pip=[], isolated=False, python=None,
        currency="none", role="wiring test only - NEVER in the results table",
        build=build_mock),
}

DEFAULT_MODELS = [k for k in ADAPTERS if k != "mock"]


def check_import(key):
    """Cheap install check: import the model package(s), no checkpoint download, no CUDA init."""
    mods = {
        "orb_mpa": ["orb_models"], "orb_omat": ["orb_models"], "mace_mpa": ["mace"],
        "sevennet_mf_ompa": ["sevenn"], "esen_30m_oam": ["fairchem.core"],
        "equiformer_v3_oam": ["fairchem.core"], "mattersim": ["mattersim"], "chgnet": ["chgnet"],
        "equflashv2_oam": ["GGNN"], "prophet_oame": ["prophet", "prophet_mlip", "kairos"],
        "tece_oam_rra": ["tace"], "pet_oam": ["upet", "pet_mad"], "mock": [],
    }[key]
    if not mods:
        return "ok"
    import importlib
    errs = []
    for m in mods:                                # any one of the candidates importing is a pass
        try:
            importlib.import_module(m)
            return f"ok ({m})"
        except Exception as e:                    # noqa: BLE001
            errs.append(f"{m}: {type(e).__name__}")
    return "FAIL " + "; ".join(errs)


def build(key):
    if key not in ADAPTERS:
        raise AdapterError(f"unknown adapter {key!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[key]["build"]()


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    if key == "--list":
        for k, a in ADAPTERS.items():
            print(f"{a['model_id']:>4}  {k:<18} isolated={a['isolated']!s:<5} {a['currency']}")
    elif key:
        print(key, "->", check_import(key))
    else:
        print("usage: bench_models.py <adapter>|--list")
