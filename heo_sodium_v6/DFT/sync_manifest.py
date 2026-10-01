"""sync_manifest.py - the SINGLE writer of manifest.csv (spec 8-6).

Reconciles filesystem evidence left by workers (claim dirs, DONE/FIZZLED sentinels,
raw/*.json) into manifest.csv.  Idempotent - safe to run any time, by hand or cron.
Replaces `lpad detect_lostruns --rerun` (stale claims of dead SLURM jobs are released)
and the FIZZLED 2-strike -> manual rule (spec section 6).

Usage:  python sync_manifest.py --root $RUN_ROOT [--config config.yaml]
"""
import argparse, glob, json, os, re, subprocess, sys, time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_common as C


def job_alive(jid):
    """True if the SLURM job still exists in the queue; None when squeue is unusable
    (then we conservatively keep the row as running)."""
    if not jid:
        return False
    try:
        out = subprocess.check_output(["squeue", "-h", "-j", str(jid)], text=True,
                                      stderr=subprocess.DEVNULL, timeout=30)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False                                   # unknown job id -> gone
    except Exception:
        return None


def _bump_note(note, key):
    """'lost=2' style counters inside the free-text note column."""
    m = re.search(rf"{key}=(\d+)", note)
    n = int(m.group(1)) + 1 if m else 1
    note = re.sub(rf"{key}=\d+", "", note).strip("; ")
    return (note + f"; {key}={n}").strip("; "), n


def sync(root, cfg):
    mpath = os.path.join(root, "manifest.csv")
    df = pd.read_csv(mpath, dtype=str).fillna("")
    df["attempts"] = df["attempts"].replace("", "0").astype(int)
    max_att, max_lost = int(cfg["max_attempts"]), int(cfg.get("max_lost", 3))
    changes = []

    for i, row in df.iterrows():
        if row["status"] in ("done", "manual"):
            continue
        rid = row["row_id"]
        calc = os.path.join(root, "calcs", rid)
        claim = os.path.join(calc, ".claim")
        done_fp, fizz_fp = os.path.join(calc, "DONE"), os.path.join(calc, "FIZZLED")
        raw_fp = os.path.join(root, "raw", f"{rid}.json")

        if os.path.exists(done_fp) and os.path.exists(raw_fp):
            jid = ""
            try:
                jid = json.load(open(os.path.join(claim, "claim.json"))).get("slurm_job_id", "")
            except Exception:
                pass
            df.loc[i, ["status", "task_id", "fw_id"]] = ["done", os.path.relpath(raw_fp, root), jid]
            changes.append(f"{rid}: done")

        elif os.path.exists(fizz_fp):
            att = row["attempts"] + 1
            head = open(fizz_fp).read().strip().splitlines()
            err = head[-1][:120] if head else "unknown"
            os.replace(fizz_fp, os.path.join(calc, f"FIZZLED.{att}"))
            if os.path.isdir(claim):
                os.replace(claim, os.path.join(calc, f".claim.failed-{att}"))
            status = "manual" if att >= max_att else "fizzled"
            note = (row["note"] + f"; attempt {att} fizzled: {err}").strip("; ")
            df.loc[i, ["status", "attempts", "note"]] = [status, att, note]
            changes.append(f"{rid}: {status} (attempt {att})")

        elif os.path.isdir(claim):
            jid = ""
            try:
                jid = json.load(open(os.path.join(claim, "claim.json"))).get("slurm_job_id", "")
            except Exception:
                pass
            alive = job_alive(jid)
            if alive or alive is None:                 # None: squeue unusable -> don't touch
                df.loc[i, ["status", "fw_id"]] = ["running", jid]
            else:                                      # walltime-cut / node death: release the claim
                note, n_lost = _bump_note(row["note"], "lost")
                os.replace(claim, os.path.join(calc, f".claim.lost-{n_lost}-{int(time.time())}"))
                status = "manual" if n_lost >= max_lost else "pending"
                if status == "manual":
                    note += "; too many lost runs - check node/walltime"
                df.loc[i, ["status", "note"]] = [status, note]
                changes.append(f"{rid}: lost run #{n_lost} released -> {status}")

        elif row["status"] not in ("pending", "fizzled", ""):
            df.loc[i, "status"] = "pending"

    if os.path.exists(mpath):
        df_old = open(mpath).read()
        with open(mpath + ".bak", "w") as f:
            f.write(df_old)
    tmp = mpath + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, mpath)

    tally = df.groupby(["set", "status"]).size().unstack(fill_value=0)
    print(tally.to_string())
    if changes:
        print(f"\n{len(changes)} change(s):")
        for c in changes:
            print("  -", c)
    manual = df[df["status"] == "manual"]
    if len(manual):
        print(f"\nMANUAL rows needing human review ({len(manual)}):")
        for _, r in manual.iterrows():
            print(f"  - {r['row_id']}: {r['note']}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    cfg = yaml.safe_load(open(os.path.join(root, args.config)
                              if not os.path.isabs(args.config) else args.config))
    sync(root, cfg)


if __name__ == "__main__":
    main()
