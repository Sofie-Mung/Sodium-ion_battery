"""bench_models.py - uMLIP adapters for the 12-model benchmark (MODEL_BENCHMARK_SPEC).

One entry per participant M01-M12 plus a mock for wiring tests.  Each adapter knows how to
install itself (pip specs, isolation needs) and how to build an ASE calculator.  Every model
runs through the SAME relax path (ASE FIRE + FrechetCellFilter, bench_anchor_runner.py), so
the calculator is the only model-specific code.

Install specs and call signatures were re-checked against each package's released source on
2026-09-29 (bench_install.py installs + smoke-tests them).  A load failure raises AdapterError
saying what to check: the runner records that as status='adapter_error' instead of dying.  Do NOT silently drop a model that fails here -
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
    """orb-models API, read from both wheels (2026-09-29):
      0.5.x : pretrained.<name>() -> model;  orb_models.forcefield.calculator.ORBCalculator(model, device=)
      >=0.6 : pretrained.<name>() -> (model, atoms_adapter);
              orb_models.forcefield.inference.calculator.ORBCalculator(model, atoms_adapter, device=)
    The venv pins 0.5.5 (same API as the v4 production run); the >=0.6 route is kept so a
    version drift fails loudly instead of as a vague 'not found'."""
    import importlib
    dev = _device()
    try:
        import orb_models
        from orb_models.forcefield import pretrained
    except Exception as e:                        # noqa: BLE001
        raise AdapterError(f"orb-models is not importable in this venv ({type(e).__name__}: {e}) - "
                           "the install did not finish; see logs/install_<adapter>.log") from e
    loaded = getattr(pretrained, name)(device=dev)
    model, adapter = (loaded if isinstance(loaded, tuple) else (loaded, None))
    errors = []
    for cand in ("orb_models.forcefield.calculator", "orb_models.forcefield.inference.calculator"):
        try:
            cls = getattr(importlib.import_module(cand), "ORBCalculator")
        except Exception as e:                    # noqa: BLE001
            errors.append(f"{cand}: {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"[orb] orb-models {getattr(orb_models, '__version__', '?')}: ORBCalculator from {cand}",
              flush=True)
        return cls(model, adapter, device=dev) if adapter is not None else cls(model, device=dev)
    raise AdapterError(f"ORBCalculator import failed (orb-models "
                       f"{getattr(orb_models, '__version__', '?')}).  Tried:\n  " + "\n  ".join(errors))

def build_mace_mpa():
    from mace.calculators import mace_mp
    return mace_mp(model="medium-mpa-0", device=_device(), default_dtype="float64")

def build_sevennet_omni():
    # 2026-09-29: M04 switched from 7net-mf-ompa to SevenNet-Omni (15 open datasets, multi-task,
    # SevenNet's recommended model).  Omni ONLY - no silent fallback to MF-ompa.
    # HEO_7NET_MODEL overrides the checkpoint keyword if the installed sevenn spells it differently.
    dev = _device(); errors = []
    name = os.environ.get("HEO_7NET_MODEL", "7net-omni")
    def _a():
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model=name, modal="mpa", device=dev)
    def _b():                                    # older sevenn: keyword 'model_type'/'modal' drift
        from sevenn.sevennet_calculator import SevenNetCalculator
        return SevenNetCalculator(name, modal="mpa", device=dev)
    for what, fn in (("sevenn.calculator", _a), ("sevenn.sevennet_calculator", _b)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError(f"SevenNet-Omni load failed (checkpoint {name!r}; modal='mpa' is REQUIRED - the "
                       "other heads (r2SCAN, omat, omol...) are the wrong currency).  Upgrade sevenn if the "
                       "keyword is unknown, or set HEO_7NET_MODEL to the name sevenn's pretrained list uses.  "
                       "Tried:\n  " + "\n  ".join(errors))

def build_esen_30m_oam():
    """eSEN-30M-OAM loads ONLY through fairchem-core 1.10.0 (the one 1.x release with the esen model
    code; v2's get_predict_unit registry has no OAM eSEN).  Checkpoint: hf.co/facebook/OMAT24
    esen_30m_oam.pt - GATED (manual approval on the model page, then HF_TOKEN).
    Loading call = matbench-discovery's test_esen_discovery.py."""
    dev = _device()
    ckpt = os.environ.get("HEO_ESEN_CKPT", "")
    if not ckpt:
        if not os.environ.get("HF_TOKEN"):
            raise AdapterError("eSEN-30M-OAM: hf.co/facebook/OMAT24 is gated - request access on the model "
                               "page, wait for approval, then `export HF_TOKEN=...` (or set HEO_ESEN_CKPT "
                               "to a downloaded esen_30m_oam.pt)")
        try:
            ckpt = _hf_download("facebook/OMAT24", "esen_30m_oam.pt")
        except Exception as e:                    # noqa: BLE001
            raise AdapterError(f"esen_30m_oam.pt download failed ({type(e).__name__}: {str(e)[:300]}) - "
                               "401/403 = access to facebook/OMAT24 not yet approved for this token") from e
    from fairchem.core import OCPCalculator
    return OCPCalculator(checkpoint_path=ckpt, cpu=(dev == "cpu"), seed=0)

def build_equiformer_v3_oam():
    """Repo route (experimental/docs/env_setup.md): the repo's vendored fairchem-core, installed
    EDITABLE from a git clone, exposes the v1 OCPCalculator; its setup_imports() follows symlinks
    into experimental/models|trainers, which is what registers 'equiformer_v3' - a tarball/wheel
    install does not carry them.  Checkpoint (not gated): hf.co/mirror-physics/equiformer_v3
    checkpoint/omat24-mptrj-salex_gradient.pt (OMat24 -> MPtrj+sAlex, conservative)."""
    dev = _device()
    ckpt = os.environ.get("HEO_EQV3_CKPT", "") or _hf_download(
        "mirror-physics/equiformer_v3", "checkpoint/omat24-mptrj-salex_gradient.pt")
    from fairchem.core import OCPCalculator
    calc = OCPCalculator(checkpoint_path=ckpt, cpu=(dev == "cpu"), seed=0)
    calc.trainer.scaler = None                    # as in the repo's test_discovery.py
    return calc

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

EQUFLASH_URL = "https://api.figshare.com/v2/file/download/65435007"   # equflashv2_oam.pt
EQUFLASH_BYTES = 898_917_921

def _fetch_equflash():
    """figshare.com/ndownloader/... answers curl/wget with an AWS WAF challenge (HTTP 202 + HTML),
    which is how a 'checkpoint' can silently be an HTML page.  The api.figshare.com URL redirects
    to S3 and works; the size check rejects anything that is not the real file."""
    dst = os.path.join(CKPT_DIR, "equflashv2_oam.pt")
    if os.path.exists(dst) and os.path.getsize(dst) == EQUFLASH_BYTES:
        return dst
    os.makedirs(CKPT_DIR, exist_ok=True)
    import urllib.request
    print(f"[equflash] downloading {EQUFLASH_URL} -> {dst}", flush=True)
    urllib.request.urlretrieve(EQUFLASH_URL, dst + ".part")
    os.replace(dst + ".part", dst)
    return dst

def build_equflashv2_oam():
    """Verified from the repo source: GGNN.common.calculator.UCalculator (fairchem v1 OCPCalculator
    subclass; default cpu=True -> must pass cpu=False)."""
    dev = _device()
    ckpt = os.environ.get("HEO_EQUFLASH_CKPT", "")
    if not ckpt or not os.path.exists(ckpt) or os.path.getsize(ckpt) < 100_000_000:
        if ckpt:
            print(f"[equflash] {ckpt} missing or too small to be the checkpoint (a saved WAF page?) "
                  "-> re-downloading", flush=True)
        ckpt = _fetch_equflash()
    if os.path.getsize(ckpt) != EQUFLASH_BYTES:
        print(f"[equflash] warning: {ckpt} is {os.path.getsize(ckpt)} bytes, expected {EQUFLASH_BYTES}",
              flush=True)
    from GGNN.common.calculator import UCalculator
    return UCalculator(checkpoint_path=ckpt, cpu=(dev == "cpu"))

def build_prophet_oame():
    """prophet-mlip (git only, import name `prophet`): KairosCalculator(model_path, use_kernel=True, ...).
    use_kernel=True needs openequivariance, which JIT-builds a C++/CUDA extension (nvcc, CUDA
    headers - a -devel image).  Default here is the plain PyTorch path; HEO_PROPHET_KERNEL=1 opts in."""
    dev = _device(); errors = []
    ckpt = os.environ.get("HEO_PROPHET_CKPT", "") or _hf_download("kairosmaterial/prophet",
                                                                  "prophet-oame-mbd.pt")
    kernel = os.environ.get("HEO_PROPHET_KERNEL", "0") == "1"
    def _a():
        from prophet import KairosCalculator
        return KairosCalculator(model_path=ckpt, use_kernel=kernel, device=dev)
    def _b():
        from prophet.calculator import KairosCalculator
        return KairosCalculator(model_path=ckpt, use_kernel=kernel, device=dev)
    for what, fn in (("prophet", _a), ("prophet.calculator", _b)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError("Prophet-OAME-MBD: KairosCalculator failed.  Tried:\n  " + "\n  ".join(errors))

def build_tece_oam_rra():
    """tace 0.2.2 (PyPI): tace.interface.ase.TACEAseCalc(model=<path>, device=, dtype=, fidelity_idx=).
    py3.13/torch 2.13 were matbench-discovery's TEST pins, not requirements (py>=3.9, torch>=2.4).
    Acceleration backends (OEQ/cuEq) are opt-in - the reference e3nn path needs no nvcc."""
    dev = _device()
    fid = os.environ.get("HEO_TECE_FIDELITY", "")
    kw = dict(fidelity_idx=int(fid)) if fid else {}
    from tace.interface.ase import TACEAseCalc
    return TACEAseCalc(model=_tece_ckpt(), device=dev, dtype="float32", **kw)

def _tece_ckpt():
    return str(os.environ.get("HEO_TECE_CKPT", "") or _hf_download("xvzemin/tace-foundations",
                                                                   "TECE-OAM-RRA-1.0.pt"))

def build_pet_oam():
    """upet 0.3.1 (read from the sdist 2026-09-29): UPETCalculator lives in upet.ase; the size is
    the model-name suffix ('pet-oam-l' | 'pet-oam-xl', no separate size kwarg), device is
    keyword-only.  PET-OAM = OMat pretrain -> sAlex+MPtrj finish (MP currency).
    No PET-MAD fallback any more: it is PBEsol, and a silent currency swap under the
    PET-OAM tag would poison the table."""
    dev = _device(); errors = []
    name = os.environ.get("HEO_PET_MODEL", "pet-oam-l")
    version = os.environ.get("HEO_PET_VERSION", "latest")    # l -> 0.1.0, xl -> 1.0.0 on HF
    def _a():
        from upet.ase import UPETCalculator
        return UPETCalculator(model=name, version=version, device=dev, uncertainty_threshold=None)
    def _b():                                    # older upet layouts
        from upet.calculator import UPETCalculator
        return UPETCalculator(model=name, device=dev)
    for what, fn in (("upet.ase", _a), ("upet.calculator", _b)):
        c = _try(errors, what, fn)
        if c is not None: return c
    raise AdapterError(f"PET-OAM ({name!r}) load failed - upet>=0.3 needs python>=3.11 (older pythons "
                       "silently resolve to upet 0.1.0, which lacks this API).  Tried:\n  "
                       + "\n  ".join(errors))

def build_mock():
    return None       # the runner branches to heo_worker.relax_mock for this adapter


# ============================================================ torch-sim (batched engine) models
# 2026-09-30: every adapter also gets a torch-sim ModelInterface so the benchmark (and a later full
# screening) can relax hundreds of cells per GPU pass, as the v3/v4 ORB production did.  Rules:
#   - the ASE calculator stays the reference: build_ts(key, calc) REUSES the calculator the ASE
#     builder made (same weights, same preprocessing), and bench_install.py's smoke test asserts
#     batch output == ASE output before the runner may use the batched engine;
#   - torch-sim is pinned to 0.5.2 everywhere (0.6 needs torch>=2.8 through nvalchemi[torch];
#     MatterSim alone pulls >=0.6 itself - fire.py / cell_filters.py are the same algorithm);
#   - wrappers return ASE conventions: energy eV per system, forces eV/A, stress eV/A^3 (3x3).
TS_PIN = "torch-sim-atomistic==0.5.2"
_TS_CLASSES = {}


def _ts_wrappers():
    """Lazy class factory: ModelInterface can only be subclassed once torch_sim is importable."""
    if _TS_CLASSES:
        return _TS_CLASSES
    import torch
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface

    class AtomsBatchModel(ModelInterface):
        """Batched model over an existing ASE calculator's network: state -> list[Atoms] ->
        one batched forward (subclass _predict) -> torch-sim tensors."""

        def __init__(self, calc, device, dtype=torch.float32):
            super().__init__()
            self.calc = calc
            self._device = torch.device(device)
            self._dtype = dtype
            self._compute_forces = True
            self._compute_stress = True

        def forward(self, state, **_kw):
            atoms = ts.io.state_to_atoms(state)
            E, F, S = self._predict(atoms)
            kw = dict(device=self._device, dtype=self._dtype)
            return {"energy": torch.as_tensor(E).to(**kw).reshape(-1),
                    "forces": torch.as_tensor(F).to(**kw).reshape(-1, 3),
                    "stress": torch.as_tensor(S).to(**kw).reshape(-1, 3, 3)}

    class OCPBatchModel(AtomsBatchModel):
        """fairchem v1 OCPCalculator family (M05 eSEN, M06 EquiformerV3, M09 EquFlashV2's
        UCalculator): the calculator's own a2g + data_list_collater + trainer.predict - exactly
        OCPCalculator.calculate, which already accepts a Batch - on many structures at once."""

        def _predict(self, atoms):
            from fairchem.core.datasets import data_list_collater
            batch = data_list_collater([self.calc.a2g.convert(a) for a in atoms], otf_graph=True)
            p = self.calc.trainer.predict(batch, per_image=False, disable_tqdm=True)
            if "stress" not in p:
                raise RuntimeError(f"checkpoint predicts {sorted(p)} - no stress -> no cell relaxation")
            return p["energy"].detach(), p["forces"].detach(), p["stress"].detach()

    class CHGNetBatchModel(AtomsBatchModel):
        """CHGNetCalculator.calculate, batched: graph_converter per structure, one predict_graph.
        CHGNet energies are eV/atom (is_intensive) and stresses GPa -> the calculator's own
        factors (n_atoms, stress_weight = ase.units.GPa) are applied, as in the ASE path."""

        def _predict(self, atoms):
            import numpy as np
            from pymatgen.io.ase import AseAtomsAdaptor
            m = self.calc.model
            structs = [AseAtomsAdaptor.get_structure(a) for a in atoms]
            graphs = [m.graph_converter(s) for s in structs]
            preds = m.predict_graph(graphs, task="efs", batch_size=len(graphs))
            preds = [preds] if isinstance(preds, dict) else preds
            ext = [len(s) if m.is_intensive else 1 for s in structs]
            E = np.array([float(p["e"]) * n for p, n in zip(preds, ext)])
            F = np.concatenate([p["f"] for p in preds])
            S = np.stack([p["s"] * self.calc.stress_weight for p in preds])
            return E, F, S

    class ProphetBatchModel(AtomsBatchModel):
        """KairosCalculator.calculate, batched.  Prophet's top-level forward is batch-native
        (n_graph = per-atom system index, cell [G,3,3], stress = virial/|det| per system), so the
        per-structure graphs are concatenated with node offsets on edge_index."""

        def _predict(self, atoms):
            from prophet.graph import dict_to_pytorch_geometric, preprocess_graph
            from prophet.model import scatter
            c, dev = self.calc, self.calc.device
            gs = [dict_to_pytorch_geometric(preprocess_graph(a, c.atom_indices, c.cutoff)) for a in atoms]
            n_node = torch.cat([g.n_node for g in gs]).long()
            off = torch.cumsum(n_node, 0) - n_node
            ei = torch.cat([g.edge_index + o for g, o in zip(gs, off)], dim=1)
            n_graph = torch.repeat_interleave(torch.arange(len(gs)), n_node)
            e_atom, f, s = c.model(
                torch.cat([g.x for g in gs]).to(dev), torch.cat([g.positions for g in gs]).to(dev),
                torch.cat([g.edge_attr for g in gs]).to(dev), ei.to(dev),
                torch.cat([g.cell for g in gs]).to(dev), n_node.to(dev),
                torch.cat([g.n_edge for g in gs]).to(dev), n_graph.to(dev))
            E = scatter(e_atom, n_graph.to(dev), dim=0, dim_size=len(gs))
            if s is None:
                raise RuntimeError("Prophet returned no stress (non-periodic cell?)")
            return E.detach(), f.detach(), s.detach()

    _TS_CLASSES.update(OCPBatchModel=OCPBatchModel, CHGNetBatchModel=CHGNetBatchModel,
                       ProphetBatchModel=ProphetBatchModel)
    return _TS_CLASSES


def _ts_orb(calc):
    import torch
    from torch_sim.models.orb import OrbModel         # torch-sim 0.5.2: orb-models 0.5.x API
    return OrbModel(model=calc.model, device=_device(), dtype=torch.float32)

def _ts_mace(calc):
    import torch
    from torch_sim.models.mace import MaceModel       # = production v4 (heo_worker.get_model)
    return MaceModel(model=calc.models[0], device=_device(), dtype=torch.float64)

def _ts_sevennet(calc):
    from sevenn.torchsim import SevenNetModel
    return SevenNetModel(os.environ.get("HEO_7NET_MODEL", "7net-omni"), modal="mpa", device=_device())

def _ts_mattersim(calc):
    from mattersim.torchsim import TorchSimWrapper
    return TorchSimWrapper(calc.potential, device=_device())

def _ts_tece(calc):
    import torch
    from tace.interface.torchsim import TACETorchSimCalc
    fid = os.environ.get("HEO_TECE_FIDELITY", "")
    return TACETorchSimCalc(model=_tece_ckpt(), device=torch.device(_device()), dtype=torch.float32,
                            **(dict(fidelity_idx=int(fid)) if fid else {}))

def _ts_pet(calc):
    from metatomic_torchsim import MetatomicModel
    from upet import get_upet
    from upet._models import upet_resolve_model
    name, size = os.environ.get("HEO_PET_MODEL", "pet-oam-l").rsplit("-", 1)
    version = os.environ.get("HEO_PET_VERSION", "latest")
    size, ver = upet_resolve_model(name, requested_size=size,
                                   requested_version=None if version == "latest" else version)
    return MetatomicModel(get_upet(model=name, size=size, version=ver), device=_device(),
                          uncertainty_threshold=None)

def _ts_wrapped(cls_name):
    def _b(calc):
        import torch
        return _ts_wrappers()[cls_name](calc, _device(), torch.float32)
    return _b

TS_BUILDERS = {
    "orb_mpa": _ts_orb, "orb_omat": _ts_orb, "mace_mpa": _ts_mace, "sevennet_omni": _ts_sevennet,
    "esen_30m_oam": _ts_wrapped("OCPBatchModel"), "equiformer_v3_oam": _ts_wrapped("OCPBatchModel"),
    "mattersim": _ts_mattersim, "chgnet": _ts_wrapped("CHGNetBatchModel"),
    "equflashv2_oam": _ts_wrapped("OCPBatchModel"), "prophet_oame": _ts_wrapped("ProphetBatchModel"),
    "tece_oam_rra": _ts_tece, "pet_oam": _ts_pet,
}


# ============================================================ registry (mirrors models_registry.yml)
# isolated=True: the model pins its own torch/python -> full venv with own torch, no system site-packages.
ADAPTERS = {
    # Every isolated venv pins its torch first (cu126 wheels: broad driver range) and
    # bench_install.py freezes it for the later steps, so no model package can re-resolve it.
    "orb_mpa": dict(
        model_id="M01", name="ORB-v3 conservative mpa", tag="bench-orb-v3-mpa",
        # 0.5.5 = the last release with the v4-production API (pretrained -> model only);
        # dm-tree==0.1.8 has cp312 wheels.  Own venv: the 09-17 run died in the shared env.
        pip=["orb-models==0.5.5"], pip_steps=[["torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu126"]],
        isolated=True, python="3.12",
        currency="MP (MPtrj+sAlex)", role="baseline (production main model)",
        build=build_orb_mpa),
    "orb_omat": dict(
        model_id="M02", name="ORB-v3 conservative omat", tag="bench-orb-v3-omat",
        pip=["orb-models==0.5.5"], pip_steps=[["torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu126"]],
        isolated=True, python="3.12",
        currency="OMat24 (PBE+U, VASP 54 / different PSP set, not MP2020-compatible) - CURRENCY CONTRAST",
        role="currency control",
        build=build_orb_omat),
    "mace_mpa": dict(
        model_id="M03", name="MACE-MPA-0 medium", tag="bench-mace-mpa-0",
        pip=["mace-torch"], isolated=False, python=None,
        currency="MP (MPtrj+sAlex)", role="current contrast model",
        build=build_mace_mpa),
    "sevennet_omni": dict(
        model_id="M04", name="SevenNet-Omni (mpa head)", tag="bench-7net-omni-mpa",
        # Omni landed in 0.12.0; 0.13.0 fixes loading its FlashTP-trained ckpt without FlashTP
        pip=["sevenn>=0.13.0"], isolated=False, python=None,
        currency="MP (mpa head = PBE+U; backbone trained on 15 open datasets)",
        role="equivariant multi-task (SevenNet lineage)",
        build=build_sevennet_omni),
    "esen_30m_oam": dict(
        model_id="M05", name="eSEN-30M-OAM", tag="bench-esen-30m-oam",
        # fairchem-core 1.10.0: torch~=2.4.0, numpy<2, python<3.13 (PyPI metadata);
        # torch-extras = torch_scatter/sparse/cluster from the PyG wheel index
        pip=["fairchem-core[torch-extras]==1.10.0", "numpy<2", "scipy<1.15", "huggingface_hub",
             "-f", "https://data.pyg.org/whl/torch-2.4.0+cu121.html"],
        pip_steps=[["torch==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu121"]],
        isolated=True, python="3.12",
        currency="MP finish (OMat24 pretrain -> MPtrj+sAlex)", role="leaderboard upper tier",
        build=build_esen_30m_oam),
    "equiformer_v3_oam": dict(
        model_id="M06", name="EquiformerV3+DeNS-OAM", tag="bench-eqv3-dens-oam",
        # experimental/docs/env_setup.md, cu128 -> cu126 (torch 2.7.1+cu126 cp311 and the PyG
        # torch-2.7.0+cu126 wheels exist).  {SRC} = the git clone bench_install.py makes.
        git="https://github.com/atomicarchitects/equiformer_v3.git",
        pip=[],
        pip_steps=[
            ["torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
             "--index-url", "https://download.pytorch.org/whl/cu126"],
            ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
             "-f", "https://data.pyg.org/whl/torch-2.7.0+cu126.html"],
            ["torch_geometric"],
            ["-r", "{SRC}/experimental/env/conda_requirements.txt"],
            ["-e", "{SRC}/packages/fairchem-core"],
            ["huggingface_hub"],
        ],
        # 2026-09-30: 3.11 -> 3.12 for torch-sim (PEP 695 syntax, python>=3.12).  The vendored
        # fairchem allows <3.13; torch 2.7.1+cu126 and the PyG pt27cu126 wheels exist for cp312.
        isolated=True, python="3.12",
        currency="MP finish (OMat24 -> MPtrj+sAlex)", role="leaderboard F1 top (vendored-fairchem route)",
        build=build_equiformer_v3_oam),
    "mattersim": dict(
        model_id="M07", name="MatterSim-v1 (5M)", tag="bench-mattersim-v1-5m",
        pip=["mattersim"], isolated=False, python=None,   # ran fine on 09-17 - left as it was
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
        # prophet-mlip: python>=3.11, torch>=2.7, ase>=3.29 (pyproject); commit = matbench-discovery's
        pip=["prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"
             "@dcfdc0b143978652d573630f1d2a6ce09e0c610a", "huggingface_hub"],
        pip_steps=[["torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu126"]],
        isolated=True, python="3.12",
        currency="MP (OMat24+ELEMENTA pretrain -> MPtrj+sAlex finish)", role="new upper tier",
        build=build_prophet_oame),
    "tece_oam_rra": dict(
        model_id="M11", name="TECE-OAM-RRA-1.0", tag="bench-tece-oam-rra-1.0",
        pip=["tace==0.2.2", "huggingface_hub"],
        pip_steps=[["torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu126"]],
        isolated=True, python="3.12",
        currency="MP finish (-oam suffix; verify on the card)", role="TACE lineage upper tier",
        build=build_tece_oam_rra),
    "pet_oam": dict(
        model_id="M12", name="PET-OAM (upet)", tag="bench-pet-oam",
        # upet 0.3.x needs python>=3.11 (metatrain 2026.4, nvalchemi-toolkit-ops) -> own py3.12 venv;
        # torch comes first from the cu126 index (metatomic-torch accepts torch>=2.3,<2.14).
        pip=["upet==0.3.1"],
        pip_steps=[["torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu126"]],
        isolated=True, python="3.12",
        currency="MP finish (OMat pretrain -> sAlex+MPtrj)", role="PET lineage",
        build=build_pet_oam),
    "mock": dict(
        model_id="MOCK", name="mock (heo_worker.relax_mock)", tag="bench-mock-v2",
        pip=[], isolated=False, python=None,
        currency="none", role="wiring test only - NEVER in the results table",
        build=build_mock),
}

DEFAULT_MODELS = [k for k in ADAPTERS if k != "mock"]

# torch-sim install step per adapter (bench_install.py runs it AFTER the ASE stack and never lets
# its failure break the ASE install: a model whose torch-sim step or batch check fails still runs
# on the ASE engine).  MatterSim >= 1.2 already depends on torch-sim-atomistic >= 0.6.
TS_PIP = {k: [TS_PIN] for k in DEFAULT_MODELS}
TS_PIP["mattersim"] = []
TS_PIP["pet_oam"] = [TS_PIN, "metatomic-torchsim"]
for _k, _extra in TS_PIP.items():
    ADAPTERS[_k]["ts_pip"] = _extra


def check_import(key):
    """Cheap install check: import the model package(s), no checkpoint download, no CUDA init."""
    mods = {
        "orb_mpa": ["orb_models"], "orb_omat": ["orb_models"], "mace_mpa": ["mace"],
        "sevennet_omni": ["sevenn"], "esen_30m_oam": ["fairchem.core"],
        "equiformer_v3_oam": ["fairchem.core"], "mattersim": ["mattersim"], "chgnet": ["chgnet"],
        "equflashv2_oam": ["GGNN"], "prophet_oame": ["prophet"],
        "tece_oam_rra": ["tace"], "pet_oam": ["upet"], "mock": [],
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


def build_ts(key, calc):
    """torch-sim ModelInterface for `key`, built on top of the ASE calculator `calc` from build()."""
    if key not in TS_BUILDERS:
        raise AdapterError(f"{key}: no torch-sim builder")
    try:
        import torch_sim  # noqa: F401
    except Exception as e:                        # noqa: BLE001
        raise AdapterError(f"{key}: torch_sim not importable in this venv ({type(e).__name__}: {e}) - "
                           "the ts_pip install step failed; see logs/install_<adapter>.log") from e
    return TS_BUILDERS[key](calc)


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
