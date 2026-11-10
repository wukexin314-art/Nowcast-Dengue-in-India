# Landscape and novelty boundary

Last searched: 2026-11-09.

## Bottom line

HonestForecast does **not** claim that forecast scoring, leakage checks, point-in-time data, rolling evaluation, or epidemic forecasting are individually new. Mature projects already cover each of those components.

The project's novelty hypothesis is narrower and testable:

> An executable evidence chain for deciding whether a feature family deserves to enter a small-sample probabilistic forecasting system when predictors are delayed, revised, or operationally fragile.

The chain is:

`feature mechanism and manifest -> origin-available vintage -> fold-local transform -> development-only screening -> common-grid probabilistic ablation -> robustness gates -> append-only promotion decision -> evidence card`

The intended contribution is the integration and enforcement of that chain, plus defect-injection benchmarks that measure false promotion—not a new forecasting model.

## Nearest public projects and methods

| Project or source | What it already does well | Overlap with HonestForecast | Boundary retained by HonestForecast |
| --- | --- | --- | --- |
| [Bank of England Forecast Evaluation](https://bank-of-england.github.io/forecast_evaluation/api/data.html) | Forecast/outturn loading, evaluation, and multiple outturn vintages | Vintage-aware evaluation | Does not govern exogenous feature construction, development screening, feature-family promotion, or public-health proxy reliability |
| [hubEvals](https://github.com/hubverse-org/hubEvals) and [scoringutils](https://github.com/epiforecasts/scoringutils) | High-quality epidemic forecast scoring, WIS, coverage, and relative skill | Probabilistic metrics and epidemic use case | HonestForecast should interoperate with—not duplicate—these scoring ecosystems; its focus is pre-score feature evidence and post-score promotion decisions |
| [EpiBenchmark](https://accidda.github.io/EpiBenchmark/) | Standardized infectious-disease benchmark design and revision-aware comparability | Common evaluation protocols and data-version concerns | HonestForecast focuses on feature-family provenance, selection, ablation, and decision records rather than benchmark hosting |
| [skforecast](https://github.com/skforecast/skforecast) | Forecast models, backtesting, exogenous variables, and feature engineering | Rolling evaluation and transformations | HonestForecast is not a model library; it constrains what was knowable, how a feature was selected, and what evidence permits promotion |
| [tsauditor](https://github.com/imann128/tsauditor) | Time-series data-quality and leakage auditing, including `available_at` semantics | Availability and leakage detection | HonestForecast constructs versioned features and connects them to probabilistic ablation, robustness, and promotion states |
| [walkforward-audit](https://github.com/mospira/walkforward-audit) | Agent-readable checklist for walk-forward leakage review | Temporal contract and leakage review | HonestForecast provides executable data contracts, metrics, synthetic known-answer failures, and decision artifacts rather than only a review workflow |
| [quant-feature-store](https://github.com/sophie-nguyenthuthuy/quant-feature-store) | Bitemporal point-in-time feature storage and leakage-safe financial backtests | Knowledge-time joins, versions, append-only data | HonestForecast is domain-neutral/public-health-oriented and evaluates whether feature families add stable probabilistic forecast value |
| [EPIFORGE guidelines](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1003793) | Reporting guidance for epidemic forecasting studies | Prospective/retrospective labeling and reproducibility | HonestForecast turns part of the reporting guidance into failable software gates and machine-readable evidence |
| [Delphi Revision Forecast](https://pmc.ncbi.nlm.nih.gov/articles/PMC12646461/) | Forecasting finalized values from revision/backfill dynamics | Public-health revisions and real-time surveillance | HonestForecast audits revision effects on predictors and feature claims; it does not replace revision-nowcasting models |
| [Recommendations for leading indicators](https://pmc.ncbi.nlm.nih.gov/articles/PMC13637922/) | Distinguishes statistical, predictive, and operational lead in epidemic surveillance | Operational availability of digital indicators | HonestForecast operationalizes this distinction with availability manifests, vintage evidence, common-grid ablations, and promotion states |
| [Robust digital disease nowcasting](https://pmc.ncbi.nlm.nih.gov/articles/PMC13630223/) | Stability and interpretability in digital-trace feature selection | Digital proxy selection and overfitting risk | HonestForecast adds origin-time versions, revision sensitivity, probabilistic comparison, and append-only negative decisions |

## What is demonstrably different today

The current implementation links capabilities that the nearest projects generally expose separately:

1. A feature output must resolve to a source, version, availability rule, transform, and training-fit window.
2. Screening rejects rows after a declared development cutoff and reports time-block sign stability, circular-shift null evidence, and BH-FDR.
3. A feature-family comparison fails unless baseline and candidate share the exact origin/target/horizon grid.
4. Promotion considers paired WIS, MAE, origin-level improvement, bootstrap uncertainty, coverage degradation, horizon loss, and drop-best-origin sensitivity.
5. Rejected families remain in an append-only log, reducing repeated unreported searches.
6. Synthetic injected defects measure detection and false-promotion behavior.
7. The India dengue case reconstructs Git-preserved feature vintages and quantifies repeated historical rewrites by source family.

## Current real-data finding

The India dengue updater preserved 18 committed monthly-data snapshots between 2026-03-31 and 2026-11-05. The audit finds 1,543 revision events across 203 unique feature-period pairs and zero cases where a previously non-null value was overwritten with null. Google Trends contributes most revision events; Wikipedia pageview definitions also rewrite history. These are repository-state vintages, not proof of upstream publication timestamps.

This finding makes the practical question concrete: a digital proxy can be available quickly yet still be unsuitable for retrospective feature selection if later downloads renormalize or redefine its history.

In the first seven-origin applied pilot, seasonal naive outperforms both a small target-history ridge model and the same model plus digital proxies on MAE and WIS. Replacing origin-matched proxy histories with the latest-revised histories changes the result further and makes it worse in this sample. The toolkit records both feature increments as rejected. This is exploratory evidence because the vintage window is short, but it demonstrates the intended behavior: complex or digital features are not promoted merely because they are available.

## Claims we will not make

- “No other project overlaps with HonestForecast.”
- “The individual metrics or leakage rules are novel.”
- “Git commit time equals source publication time.”
- “A synthetic false-promotion rate is a population estimate.”
- “The dengue features improve forecasts” until the registered common-grid applied experiments are complete.

## Search limitations

This is a structured landscape review, not a systematic review or patent search. Searches covered GitHub repositories, official package documentation, and recent epidemic forecasting/revision literature using combinations of forecasting, vintage, point-in-time, leakage, feature engineering, public health, WIS, and audit terms. Exact-name searches found no established forecasting package named `HonestForecast` on the search date, but that is not a trademark clearance.
