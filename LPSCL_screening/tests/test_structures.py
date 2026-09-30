"""Spec 12-2 / 12-3: charge ledger + structure generation + invariants (CPU, no MLIP)."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import sse_worker as W

cfg = W.load_config(overrides={"workdir": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "allow_24g_host": True})
print(W.check_pilot_ledger().to_string(index=False))
b, m = W.stage2_compositions()
print("stage II:", len(b), "baselines", len(m), "mixtures")
df = pd.DataFrame([W.comp_summary_row(c) for c in m])
print(df.groupby("valence_combo").n_li_per_uc.agg(["min", "max", "count"]))

t0 = time.time()
host = W.load_host(cfg, log_dir=os.path.join(cfg["workdir"], "logs"))
print(f"host loaded in {time.time()-t0:.1f}s: mode={host.li_site_mode} a={host.a:.4f} pairs={host.li_pairs.shape} stats={host.pair_stats}")
print(pd.Series(host.labels).value_counts().to_dict())

t0 = time.time()
rows = []
for comp in W.pilot_compositions():
    for cid, atoms, meta in W.generate_configs(host, comp, "I", cfg):
        rows.append(dict(comp=comp["id"], cid=cid, mode=meta["config_mode"], n=len(atoms), min_d=round(meta["min_dist"], 3),
                         vac=meta["n_vacancy"], near=meta["n_vacancy_near"], int16e=meta["n_interstitial_16e"],
                         int_other=meta["n_interstitial_other"], labels=meta["interstitial_labels"][:40],
                         d_dop_ge=round(meta["d_dopant_ge_min"], 2), d_dop_vac=round(meta["d_dopant_vacancy_min"], 2)))
print(f"{len(rows)} configs generated in {time.time()-t0:.1f}s, all invariants passed")
print(pd.DataFrame(rows).to_string(index=False))
# write/read round trip keeps labels
a, meta = W.build_config(host, W.pilot_compositions()[1], "random", 1, cfg)
p = "/tmp/_sse_test.extxyz"; W.write_atoms(a, p, info={"x": 1}); b = W.read_atoms(p)
assert (b.get_array("site_label") == a.get_array("site_label")).all() and len(b) == len(a)
W.check_invariants(b, W.pilot_compositions()[1], dict(meta), cfg)
# a few stage-II mixtures
for comp in m[:3] + m[-3:]:
    for cid, atoms, meta in W.generate_configs(host, comp, "II", cfg):
        pass
    print("stage II ok:", comp["id"], W.n_li_of(comp), meta["n_vacancy"], meta["n_interstitial"], round(W.configurational_entropy_eV_per_K(comp)*1000, 4), "meV/K")
print("ALL STRUCTURE TESTS PASSED")
