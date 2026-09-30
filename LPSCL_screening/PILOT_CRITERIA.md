# PILOT_CRITERIA.md — Stage I gate criteria (frozen BEFORE any result is inspected)

Written 2026-09-27, copied verbatim from SSE_PIPELINE_SPEC.md Section 9.
This file must not be edited after Stage I results exist. `sse_worker.py` stores the
md5 of this file in `outputs/run_manifest.json` at the start of Stage I and re-checks it
at the end; a mismatch aborts the run.

| ID | Criterion | Rule | On failure |
|---|---|---|---|
| G-a | Ge trend reproduced | main model: 95% CI lower bound of σ_F/σ_A > 1 at 600 K | STOP, report to user |
| G-b | P-site compensation effect | for each valence X ∈ {N, M, L}: σ_X2/σ_X0 > 1 (CI lower bound) | record only; used when interpreting Stage II |
| G-c | Dopant intrinsic effect | 95% CI of σ_D/σ_A does not contain 1 | record only (if it contains 1: report "no effect at equal Li") |
| G-lin | Linearity of ln σ in N_Li | A, F, N0, N2, M0, M2, L0, L2: weighted linear regression of ln σ(600 K) vs N_Li gives R² ≥ 0.8 and no systematic curvature (CI of the quadratic coefficient contains 0) | replace the S_syn denominator by interpolation of single-dopant ln σ at the mixture's N_Li |
| G-model | Two-model agreement | Spearman ρ ≥ 0.6 between the two models' ratio_vs_A rankings over the 9 pilot compositions | **not evaluated** in the ORB-only pilot (user decision 2026-09-28); SevenNet may later rescore the top candidates |
| G-stab | Stage II stability threshold | proposed after Stage I, confirmed by the user, then recorded here | — |

## Definitions used by the code

* Pilot protocol (user decision 2026-09-28, set before any Stage I result exists): ORB v3 only,
  2 fs timestep, 600/750/900 K, NPT 10 ps + NVT 100 ps, 2 configurations per composition.
  All gates use 600 K ratios; the 300 K Arrhenius extrapolation is not used for any decision
  and is not reported.
* σ at a given (model, composition, T) is the equal-weight average over the 2 lowest-energy
  configurations of ln σ; the variance is (mean within-config posterior variance) +
  (between-config variance). Boltzmann-weighted values are reported as a sensitivity check only.
* All CIs are 95% (`CONFIG["ci_level"] = 0.95`).
* Ratios are formed from the posterior of ln σ of the two compositions computed with the SAME
  model at the SAME temperature.
* G-lin regression is weighted by 1/var(ln σ).

## G-stab (filled in after Stage I, before Stage II)

de_hull_corr_vs_host threshold (eV/atom): **not yet set** — proposal will be based on the
pilot values of D and N2; default proposal 0.03 eV/atom. Set `CONFIG["stab_threshold_eV"]`
explicitly before Stage II; the notebook refuses to start Stage II while it is `None`.
