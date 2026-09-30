"""End-to-end wiring test with the mock (Lennard-Jones) model: generate -> relax -> MD -> kinisi -> tables -> gates -> pptx -> export."""
import sys, os, time, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import sse_worker as W

wd = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sse_e2e_mock"
if "--keep" not in sys.argv: shutil.rmtree(wd, ignore_errors=True)
os.makedirs(wd, exist_ok=True)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = W.load_config(overrides={"workdir": wd, "host_cif": os.path.join(root, "inputs/Li6PS5Cl.cif"), "allow_24g_host": True,
                               "relax_max_steps": 20, "relax_chunk": 4, "md_save_every_fs": 10.0, "kinisi_n_samples": 150, "kinisi_n_burn": 60,
                               "models": ["mock", "mock2"], "n_configs_md_pilot": 2, "md_temperatures_pilot": [500, 700], "n_mc_samples": 500})
W.atomic_write_json(cfg, W.wpath(cfg, "outputs", "config.json"))
host = W.load_host(cfg, log_dir=os.path.join(wd, "logs"))
comps = W.pilot_compositions()
metas = W.ensure_generated(host, comps, "I", cfg)
keys = list(metas)
print("generated", len(keys))
# two "models": both the mock LJ (second one with a different name to exercise the two-model bookkeeping)
handles = {"mock": W.load_model("mock", cfg)}
h2 = W.load_model("mock", cfg); h2.name = "mock2"; handles["mock2"] = h2
relax_dfs, hull_dfs, rs_all, md_jobs_all = {}, {}, [], {}
for m, h in handles.items():
    t0 = time.time(); W.run_relax_jobs(h, W.relax_jobs_for(cfg, metas, keys), cfg); print(m, "relax", round(time.time()-t0, 1), "s")
    rel = W.relax_table(cfg, m, metas, keys); rel["converged"] = True  # mock never converges in 20 steps; pretend for the wiring test
    relax_dfs[m] = rel
    hd = rel.copy(); hd["e_above_hull"] = np.random.default_rng(1).uniform(0, 0.05, len(hd)); hd["e_above_hull_host"] = 0.01; hd["de_hull_vs_host"] = hd.e_above_hull - 0.01
    hull_dfs[m] = hd
    jobs = []
    for c in comps:
        cids = W.select_md_configs(rel, c["id"], cfg["n_configs_md_pilot"])
        jobs += W.md_jobs_for(cfg, m, c["id"], cids, cfg["md_temperatures_pilot"], npt_ps=0.1, nvt_ps=0.4)
    md_jobs_all[m] = jobs
    t0 = time.time(); W.run_md_jobs(h, jobs, cfg); print(m, "md", len(jobs), "jobs", round(time.time()-t0, 1), "s")
    t0 = time.time(); W.run_analyze_jobs(W.analyze_jobs_for(cfg, m, jobs), cfg); print(m, "analyze", round(time.time()-t0, 1), "s")
    rs = W.results_single_table(cfg, m, comps, hd, cfg["md_temperatures_pilot"], "I"); rs_all.append(rs)
rs = pd.concat(rs_all); rs.to_csv(W.wpath(cfg, "outputs", "results_single.csv"), index=False)
print("results_single rows", len(rs), "cols", len(rs.columns)); print(rs[["model","composition_id","config_id","T","sigma_Scm","sigma_ci_low","sigma_ci_high"]].head(6).to_string())
pt = W.per_T_table(cfg, rs); pt.to_csv(W.wpath(cfg, "outputs", "results_single_perT.csv"), index=False)
print(pt[["model","composition_id","T","sigma_Scm","sigma_ci_low","sigma_ci_high","ratio_vs_A","ratio_vs_A_ci_low"]].head(8).to_string())
summ = W.summary_table(cfg, pt, comps, hull_dfs); summ.to_csv(W.wpath(cfg, "outputs", "results_single_summary.csv"), index=False)
print(summ[["model","composition_id","sigma_600K","Ea","Ea_ci_low","Ea_ci_high","sigma_300K","ratio_vs_A_300K","de_hull_vs_host_mean"]].to_string())
# 600 K missing here (temps 500/700) -> sigma_600K NaN by design; model selection uses 600 K -> exercise with T_ref override
pt600 = pt.copy(); pt600.loc[pt600["T"] == 700.0, "T"] = 600.0
summ600 = W.summary_table(cfg, pt600, comps, hull_dfs)
msel, choice = W.model_selection_table(cfg, summ600, pt600, relax_dfs, {"mock": 5.0, "mock2": 4.0}, {"mock": handles["mock"].tag, "mock2": h2.tag},
                                       pd.read_csv(os.path.join(root, "inputs/exp_reference.csv")))
msel.to_csv(W.wpath(cfg, "outputs", "results_model_selection.csv"), index=False)
print(msel[["model","a_relaxed","sigma_A_600K","ratio_F_A_600K","ratio_F_A_600K_ci_low","passes_ge_trend","role"]].to_string()); print(choice)
gates = W.evaluate_gates(cfg, summ600, pt600, "mock", "mock2"); W.atomic_write_json(gates, W.wpath(cfg, "outputs", "gates.json"))
# single-model run (ORB-only pilot, 2026-09-28): no contrast model, G-model skipped
cfg1 = dict(cfg, models=["mock"])
msel1, choice1 = W.model_selection_table(cfg1, summ600[summ600.model == "mock"], pt600[pt600.model == "mock"], {"mock": relax_dfs["mock"]}, {"mock": 5.0},
                                         {"mock": handles["mock"].tag}, None)
assert choice1["contrast"] is None and len(msel1) == 1, choice1
gates1 = W.evaluate_gates(cfg1, summ600, pt600, "mock", None); assert "G-model" not in gates1
print("single-model:", choice1)
print(json.dumps(gates, indent=1, default=str)[:1500])
# stage II pieces on a tiny subset: use the pilot A/N0/M0 as fake baselines through the same functions
base, mix = W.stage2_compositions()
stab = W.stage2_stability_table(dict(cfg, stab_threshold_eV=0.03), "mock", mix[:5], hull_dfs["mock"].iloc[0:0], 0.01)
print(stab[["composition_id","li_group","dS_conf_meV_per_K","de_hull_corr_vs_host","pass_stability"]].to_string())
fig = W.make_figures(cfg, summ600, msel, None, None, pt600, gates, W.wpath(cfg, "outputs", "figures.pptx")); print("pptx:", fig, os.path.getsize(fig))
out = W.export_dft_handoff(cfg, "mock", ["A", "D"], [], {"A": ["A__c0"], "D": ["D__c0"]}, {"A": [("A__c0__500K", 500)]}, {"mock": []},
                           {"models": {"mock": {"tag": handles["mock"].tag}}}, n_frames=5)
print("handoff:", sorted(os.listdir(os.path.join(out, "pilot", "A"))))
print("E2E MOCK OK")
