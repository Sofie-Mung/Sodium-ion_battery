"""runner.py - STEP 3: resident worker. One sbatch job holds a node and executes manifest
rows sequentially until the queue drains or the walltime deadline approaches (the
FireWorks-free analog of `rlaunch rapidfire --timeout`, spec sections 2/6).

State model (spec 8-6): manifest.csv is the single durable state store and this process
NEVER writes it.  A worker leaves only filesystem evidence -
    calcs/{row_id}/.claim/claim.json   atomic runtime lock (os.mkdir: atomic incl. NFS)
    calcs/{row_id}/run-*/              atomate2/jobflow run directories
    calcs/{row_id}/DONE | FIZZLED      outcome sentinels
    raw/{row_id}.json                  serialized relax+static TaskDocs
- which sync_manifest.py (the single manifest writer) folds back into manifest.csv.

atomate2 API note (spec 10-#2, verified at pilot P1): MPGGARelaxMaker/MPGGAStaticMaker with
MPGGA*SetGenerator(user_incar_settings=...) and BaseVaspMaker.run_vasp_kwargs={"vasp_cmd": ...}.
If the installed version rejects run_vasp_kwargs["vasp_cmd"], fall back to exporting
ATOMATE2_VASP_CMD / ATOMATE2_VASP_GAMMA_CMD before Python starts (see RUNBOOK P1).

Usage:  python runner.py --root $RUN_ROOT --config config.yaml [--sets pilot] [--once]
"""
import argparse, datetime, json, os, socket, subprocess, sys, time, traceback

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_common as C

MAIN_SETS = "top30,anchor,E_Na"


# ============================================================ walltime deadline
def _parse_walltime(s):
    d, rest = (s.split("-", 1) + ["0"])[:2] if "-" in s else ("0", s)
    h, m, sec = (rest.split(":") + ["0", "0"])[:3]
    return int(d) * 86400 + int(h) * 3600 + int(m) * 60 + int(sec)


def deadline_ts(cfg):
    """Job end time from SLURM, minus the safety margin (spec 10-#5 analog)."""
    end = None
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        try:
            out = subprocess.check_output(["squeue", "-h", "-j", jid, "-O", "EndTime:64"],
                                          text=True, timeout=30).strip()
            if out and out not in ("N/A", "NONE", "UNLIMITED", "UNKNOWN"):
                end = datetime.datetime.strptime(out, "%Y-%m-%dT%H:%M:%S").timestamp()
        except Exception as e:
            print(f"[warn] squeue EndTime failed ({e}); using config walltime", flush=True)
    if end is None:
        end = time.time() + _parse_walltime(str(cfg["walltime"]))
    return end - 60 * float(cfg.get("safety_margin_min", 30))


# ============================================================ claiming
def try_claim(calc_dir, row_id, vasp_mode):
    """Atomic claim via os.mkdir - atomic on POSIX filesystems including NFS, unlike
    O_CREAT|O_EXCL on old NFS or flock without a configured lockd.  Exactly one of up
    to n_workers concurrent workers wins; the dir stays behind for post-mortem."""
    os.makedirs(calc_dir, exist_ok=True)
    try:
        os.mkdir(os.path.join(calc_dir, ".claim"))
    except FileExistsError:
        return False
    with open(os.path.join(calc_dir, ".claim", "claim.json"), "w") as f:
        json.dump({"row_id": row_id, "hostname": socket.gethostname(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "pid": os.getpid(), "vasp_mode": vasp_mode,
                   "started_at": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=1)
    return True


# ============================================================ flow construction + execution
def _serialize(obj):
    from monty.json import jsanitize
    for attempt in (lambda o: jsanitize(o, strict=True, enum_values=True),
                    lambda o: jsanitize(o.model_dump(), strict=False),
                    lambda o: jsanitize(o.dict(), strict=False)):
        try:
            return attempt(obj)
        except Exception:
            continue
    raise RuntimeError(f"cannot serialize TaskDoc of type {type(obj)}")


def build_structure(root, row):
    from pymatgen.core import Structure
    if row["set"] == "E_Na":
        st = C.bcc_na_2atom()
    else:
        st = Structure.from_file(os.path.join(root, row["poscar"]))
        got = C.struct_hash(st)
        assert got == row["struct_hash"], \
            f"{row['row_id']}: POSCAR hash {got} != manifest {row['struct_hash']} (bundle corruption)"
    mm = json.loads(row["magmom_json"] or "{}")
    st.add_site_property("magmom", [float(mm.get(s.specie.symbol, 0.0)) for s in st])
    return st


def run_row(root, row, cfg):
    """Relax -> static MP flow for one manifest row.  Returns the raw-json payload."""
    from atomate2.vasp.jobs.mp import MPGGARelaxMaker, MPGGAStaticMaker
    from atomate2.vasp.sets.mp import MPGGARelaxSetGenerator, MPGGAStaticSetGenerator
    from jobflow import Flow, run_locally

    mode = row["vasp_mode"] if row["vasp_mode"] in ("std", "gam") else cfg["kmode"]
    if row["set"] == "E_Na":
        mode = "std"                                      # metal: never Gamma-only
    vasp_cmd = str(cfg["vasp_cmd_gam" if mode == "gam" else "vasp_cmd_std"]).format(**cfg)

    relax_incar = dict(C.INCAR_OVERRIDES_RELAX)
    if "custodian_test" in str(row.get("note", "")):
        relax_incar["NELM"] = 8                           # spec 4: force a custodian recovery
    gen_kw = {}
    if mode == "gam":
        from pymatgen.io.vasp.inputs import Kpoints
        gen_kw["user_kpoints_settings"] = Kpoints.gamma_automatic((1, 1, 1))

    relax = MPGGARelaxMaker(
        input_set_generator=MPGGARelaxSetGenerator(user_incar_settings=relax_incar, **gen_kw),
        run_vasp_kwargs={"vasp_cmd": vasp_cmd})
    static = MPGGAStaticMaker(
        input_set_generator=MPGGAStaticSetGenerator(
            user_incar_settings=dict(C.INCAR_OVERRIDES_STATIC), **gen_kw),
        run_vasp_kwargs={"vasp_cmd": vasp_cmd})

    st = build_structure(root, row)
    j1 = relax.make(st)
    j2 = static.make(j1.output.structure, prev_dir=j1.output.dir_name)
    for j in (j1, j2):
        j.update_metadata({k: row[k] for k in
                           ("row_id", "set", "comp_id", "phase", "x_tag", "na_count",
                            "k_best", "struct_hash")})     # spec 5 metadata keys
    flow = Flow([j1, j2], name=row["row_id"])

    run_dir = os.path.join(root, "calcs", row["row_id"],
                           "run-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        responses = run_locally(flow, create_folders=True, ensure_success=True)
    finally:
        os.chdir(cwd)

    docs = {}
    for j, name in ((j1, "relax"), (j2, "static")):
        r = responses[j.uuid][max(responses[j.uuid])]
        docs[name] = _serialize(r.output)
    return {"row_id": row["row_id"], "vasp_mode": mode, "run_dir": run_dir,
            "metadata": {k: row[k] for k in ("set", "comp_id", "phase", "x_tag",
                                             "na_count", "k_best", "struct_hash",
                                             "mlip_E", "mlip_dE")},
            **docs}


# ============================================================ main loop
def candidates(root, sets, cfg):
    df = pd.read_csv(os.path.join(root, "manifest.csv"), dtype=str).fillna("")
    df["priority"] = df["priority"].astype(float)
    df["attempts"] = df["attempts"].replace("", "0").astype(int)
    df = df[df["set"].isin(sets)]
    df = df[df["status"].isin(["pending", "fizzled", ""]) & (df["attempts"] < int(cfg["max_attempts"]))]
    return df.sort_values("priority", ascending=False, kind="stable")


def pilot_gate_ok(root):
    df = pd.read_csv(os.path.join(root, "manifest.csv"), dtype=str).fillna("")
    p = df[df["set"] == "pilot"]
    return len(p) > 0 and (p["status"] == "done").all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sets", default=MAIN_SETS, help="comma list: pilot | top30,anchor,E_Na")
    ap.add_argument("--once", action="store_true", help="run a single row then exit (debug)")
    ap.add_argument("--skip-pilot-gate", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    cfg = yaml.safe_load(open(os.path.join(root, args.config)
                              if not os.path.isabs(args.config) else args.config))
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]

    if set(sets) != {"pilot"} and not args.skip_pilot_gate:
        assert pilot_gate_ok(root), \
            "pilot gate: all 8 pilot rows must be done before the main run (spec section 4). " \
            "Run './submit.sh --pilot 1' first, or pass --skip-pilot-gate deliberately."

    deadline = deadline_ts(cfg)
    t_est = 3600.0 * float(cfg["t_est_hours"])
    print(f"[worker] root={root} sets={sets} deadline={datetime.datetime.fromtimestamp(deadline)}",
          flush=True)

    n_done = 0
    while True:
        if time.time() + t_est > deadline:
            print(f"[worker] < t_est ({cfg['t_est_hours']} h) left before deadline; "
                  f"exiting cleanly after {n_done} rows", flush=True)
            return
        cand = candidates(root, sets, cfg)
        claimed = None
        for _, row in cand.iterrows():
            calc_dir = os.path.join(root, "calcs", row["row_id"])
            if os.path.exists(os.path.join(calc_dir, "DONE")):
                continue                                   # done but manifest not yet synced
            mode = row["vasp_mode"] if row["vasp_mode"] in ("std", "gam") else cfg["kmode"]
            if try_claim(calc_dir, row["row_id"], mode):
                claimed = row
                break
        if claimed is None:
            print(f"[worker] no claimable rows; exiting after {n_done} rows", flush=True)
            return

        row_id = claimed["row_id"]
        calc_dir = os.path.join(root, "calcs", row_id)
        print(f"[worker] {row_id} claimed ({time.strftime('%F %T')})", flush=True)
        try:
            payload = run_row(root, claimed, cfg)
            os.makedirs(os.path.join(root, "raw"), exist_ok=True)
            raw_fp = os.path.join(root, "raw", f"{row_id}.json")
            C.atomic_write(raw_fp, json.dumps(payload))
            C.atomic_write(os.path.join(calc_dir, "DONE"), json.dumps(
                {"finished_at": datetime.datetime.now().isoformat(timespec='seconds'),
                 "raw": raw_fp, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}))
            n_done += 1
            print(f"[worker] {row_id} DONE", flush=True)
        except Exception:
            C.atomic_write(os.path.join(calc_dir, "FIZZLED"),
                           time.strftime("%F %T\n") + traceback.format_exc())
            print(f"[worker] {row_id} FIZZLED:\n{traceback.format_exc()}", flush=True)
        if args.once:
            return


if __name__ == "__main__":
    main()
