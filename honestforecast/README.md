# HonestForecast

HonestForecast is a small, standard-library toolkit for feature-aware and audit-friendly forecasting studies. It validates forecast and feature contracts, reconstructs origin-available feature values, fits preprocessing inside training folds, screens candidates only on development data, compares feature families on identical forecast cases, and emits machine-readable evidence.

This directory is clean-room implementation work. It does not import private-project data, results, dashboards, or code.

## Quick start

```bash
cd honestforecast
python3 -m pip install --upgrade pip
python3 -m pip install -e .
honestforecast-synthetic --output-dir build/synthetic
python3 -m unittest discover -s tests -v
```

The synthetic command produces:

- `predictions.csv`: clean prediction-level evidence;
- `metrics.json`: metrics regenerated from those rows;
- `audit_receipt.json`: known-answer outcomes for S0, S1, S2, S3, S5, S6, and S8;
- `feature_lineage.json`: source, version, transform, availability rule, fit window, and fitted parameters;
- `evidence_card.md`: a compact result and gate record.

Audit the Git-preserved India dengue data vintages from the repository root:

```bash
honestforecast-vintage-audit \
  --repo . \
  --data-path dengue_updater/data/processed/monthly_data.csv \
  --update-log dengue_updater/logs/update_log.csv \
  --as-of 2026-11-09 \
  --output-dir honestforecast/case_studies/india_dengue/vintage_audit
```

Run the minimal applied pilot on identical repository-vintage origins:

```bash
honestforecast-dengue-pilot \
  --repo . \
  --data-path dengue_updater/data/processed/monthly_data.csv \
  --output-dir honestforecast/case_studies/india_dengue/pilot
```

## What is new—and what is not

WIS, rolling evaluation, point-in-time joins, and leakage checks already exist in strong open-source projects. HonestForecast's contribution is the enforced evidence chain connecting feature availability and lineage to development-only screening, identical-grid probabilistic ablation, robustness gates, append-only decisions, and reproducible evidence cards.

See [the landscape and novelty boundary](docs/LANDSCAPE.md) for the nearest projects, overlaps, citations, and claims this project deliberately does not make.

## Implemented scope

- `ForecastContract` and `FeatureSpec` validation;
- CSV feature-manifest loading;
- origin-aware as-of revision joins and availability/vintage audits;
- registered identity, log, lag, trailing-mean, momentum, and Fourier transforms;
- fold-local median imputation and quantile clipping with lineage;
- development cutoff enforcement, coverage, Spearman, time-block stability, circular-shift nulls, and BH-FDR;
- exact common-grid comparison;
- prediction key and quantile validation;
- MAE, WIS, 50/80/95% coverage, and interval width;
- paired feature-family WIS/MAE ablation, origin bootstrap, coverage, drop-best-origin, and promotion decisions;
- append-only decision logs that retain rejected families;
- deterministic synthetic known-answer scenarios and measurable null-feature false promotion;
- Git-history vintage extraction, hashes, cell-level revision evidence, updater-health summaries, and schema-change records;
- a minimal India dengue pilot with origin-matched repository vintages, seasonal-naive and ridge baselines, digital-proxy ablation, and latest-revised sensitivity;
- Markdown evidence-card generation.

The current applied pilot is deliberately small and exploratory. A longer prospective confirmation grid and the full registered feature-family study remain future work.

## Real-data vintage finding

The current India dengue case contains 18 Git-preserved snapshots from 2026-03-31 through 2026-11-05. Across consecutive snapshots, 1,543 revision events affect 203 unique feature-period pairs, while zero previously non-null values are overwritten with null. The evidence is explicitly classified as repository-state history, not source-native publication timestamps.

See the [vintage audit report](case_studies/india_dengue/vintage_audit/REPORT.md) and machine-readable receipt in the same directory.

The seven-origin exploratory applied pilot rejects both the small target-history ridge model and its digital-proxy increment against seasonal naive. Latest-revised proxy histories perform worse than the origin-matched versions in this pilot. These are deliberately retained negative results, not evidence that digital traces can never help. See the [applied pilot report](case_studies/india_dengue/pilot/REPORT.md).

## Core API sequence

1. Use `as_of_join` to select the last source revision actually available at each origin.
2. Use `fit_fold_transform` to fit imputation/clipping only through `train_end`; save its `FeatureLineage`.
3. Use `screen_candidates` with an explicit `development_end`; confirmation rows are rejected.
4. Use `compare_feature_family` on baseline/candidate prediction rows. Exact grid equality is mandatory.
5. Use `append_decision` to retain every challenger, promotion, and rejection.

`promoted` is possible only when performance/robustness gates pass and the caller explicitly confirms both a frozen confirmation analysis and all external audit gates. Otherwise a favorable result remains `challenger`.

## Design boundary

HonestForecast supervises forecasting research; it is not a model zoo. A result that fails task, leakage, comparison, or probabilistic-format checks is not eligible for promotion regardless of its headline metric.
