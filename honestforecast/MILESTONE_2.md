# Milestone 2 completion record

Completed locally on 2026-11-09.

## Delivered

- origin-aware as-of joins across revised feature observations;
- explicit rejection of future availability and later snapshots used as historical vintages;
- a fixed transform registry: identity, nonnegative and signed `log1p`, lags 1/2/12, trailing means 2/3, one-period momentum, and two Fourier harmonics;
- fold-local median imputation and quantile clipping whose fitted parameters ignore later rows;
- complete lineage for source feature, family, version, availability rule, transform, fit window, and fitted parameters;
- development-only screening with coverage, tied-rank Spearman correlation, early/later sign stability, circular-shift null comparison, and BH-FDR;
- a warning when tested variants exceed the registered search;
- exact-grid feature-family ablation with paired WIS/MAE, improved-origin proportion, origin bootstrap interval, coverage change, horizon loss, and drop-best-origin sensitivity;
- promotion logic that requires robustness, frozen confirmation, and explicit passage of external audit gates;
- append-only JSONL decisions, including rejected families;
- synthetic S2 vintage mismatch, S3 post-hoc null lag search, and S8 one-outbreak concentration scenarios;
- 18 unit/integration tests across Python 3.9-compatible code.

## Verified commands

From the repository root:

```bash
python3 -m venv /private/tmp/honestforecast-m2-venv-20261109
/private/tmp/honestforecast-m2-venv-20261109/bin/python -m pip install -e ./honestforecast
PYTHONPYCACHEPREFIX=/private/tmp/honestforecast-m2-pycache \
  /private/tmp/honestforecast-m2-venv-20261109/bin/python \
  -m unittest discover -s honestforecast/tests -v
/private/tmp/honestforecast-m2-venv-20261109/bin/honestforecast-synthetic \
  --output-dir honestforecast/build/synthetic
```

Observed result: all 18 tests passed. The deterministic synthetic receipt recorded:

- S0 clean process: passed;
- S1 future availability: detected;
- S2 later snapshot substituted for an old vintage: detected;
- S3 post-hoc null lags: detected;
- S5 unequal grids: detected;
- S6 crossing quantiles: detected;
- S8 gain concentrated in one outbreak: detected;
- naive null false-promotion rate: 17/40 (42.5%);
- registered/FDR/stability pipeline false-promotion rate: 0/40 (0%).

The 40-run simulation is a deterministic software known-answer check, not a publishable Monte Carlo estimate.

## Exit criteria

- No fitted preprocessing parameter uses data after `train_end`; covered by a changed-future-value invariance test.
- Every constructed feature has a lineage record with source, version, availability rule, and fit window.
- Family comparisons fail on unequal origin/target/horizon grids.
- Rejected families remain in an append-only decision log.
- Null-feature false promotion is measured and the post-hoc search warning is deterministic.
- S2, S3, and S8 findings are stable under the fixed master seeds.

## Boundaries retained

- No existing dengue model, input table, updater, API, client, dashboard, or checked-in model output was changed.
- No private-project artifact is imported or used.
- The synthetic result is E4 evidence for software behavior, not evidence of applied dengue forecast skill.
- Historical snapshot extraction, baseline refitting, and the public-data Dengue adapter remain Milestone 3.
