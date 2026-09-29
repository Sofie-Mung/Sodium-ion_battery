"""orb_static.py - F0 softening measurement, GPU machine side.

Fixed-geometry ORB single points on every DFT trajectory frame - NO relaxation may be
mixed in (spec 8-4): the softening coefficient c compares forces on IDENTICAL geometries.

Interface (the only two files that cross machines):
    in : frames.extxyz      from collect.py (info dft_energy/dft_e0, arrays dft_forces)
    out: orb_on_dft.extxyz  same frames + info E_orb [eV] + arrays orb_forces [eV/A]
audit.py consumes only the output file.

Usage (GPU machine):
    python orb_static.py --frames frames.extxyz --out orb_on_dft.extxyz \
        [--model orb_v3_conservative_inf_mpa] [--device cuda]
"""
import argparse, sys, time

import numpy as np


def make_calculator(model_name, device):
    """ORB ASE calculator.  The GPU machine already runs this model for the screening
    (heo_worker MODE=orb, same MODEL_TAG currency) - keep the model name identical so
    c is measured for the model that produced the ranking."""
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    fn = getattr(pretrained, model_name, None)
    if fn is None:
        avail = [n for n in dir(pretrained) if n.startswith("orb")]
        sys.exit(f"unknown ORB model '{model_name}'; available: {avail}")
    return ORBCalculator(fn(device=device), device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="orb_v3_conservative_inf_mpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N frames")
    args = ap.parse_args()

    from ase.io import read, write
    frames = read(args.frames, index=":")
    if args.limit:
        frames = frames[:args.limit]
    calc = make_calculator(args.model, args.device)

    t0 = time.time()
    for i, at in enumerate(frames):
        at.calc = calc
        at.info["E_orb"] = float(at.get_potential_energy())   # static only - never at.get positions changed
        at.new_array("orb_forces", np.asarray(at.get_forces(), dtype=float))
        at.info["orb_model"] = args.model
        at.calc = None
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(frames)}  ({(time.time()-t0)/(i+1):.2f} s/frame)", flush=True)

    write(args.out, frames)
    print(f"wrote {args.out}: {len(frames)} frames, model {args.model}")


if __name__ == "__main__":
    main()
