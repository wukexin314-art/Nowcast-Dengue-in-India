# Prespecified experiment matrix

## 1. Design principle

The first paper minimizes degrees of freedom. It tests a small number of meaningful feature increments under multiple information regimes. It does not rank a large model zoo.

## 2. Evaluation regimes

| ID | Regime | Purpose | Claim level |
| --- | --- | --- | --- |
| R0 | Latest revised values, ordinary time split | Quantify the result a typical retrospective workflow would report | Diagnostic |
| R1 | Latest values with explicit reporting-lag masking | Separate simple availability control from true vintage effects | Sensitivity |
| R2 | Audited Git/file historical snapshots | Pilot reconstructed historical as-of evaluation | Historical or reconstructed, depending on audit |
| R3 | Prospectively saved snapshots after freeze | Confirmation and ongoing monitoring | Prospective |

R0 is not allowed to supply features that cross the target boundary. It is intentionally less strict about revisions, not about direct target leakage.

## 3. Feature families

| ID | Family | Candidate contents | Role |
| --- | --- | --- | --- |
| F0 | Seasonality only | intercept, month/Fourier terms | Minimal baseline |
| F1 | Target history | lags 1, 2, 12; forecast-safe recursive use | Core AR baseline |
| F2 | Google fever | level and one development-selected transform | Digital proxy increment |
| F3 | Google vaccine | level and one development-selected transform | Digital proxy increment |
| F4 | Wikipedia dengue | versioned raw or normalized series, not both selected on test | Digital proxy increment |
| F5 | Wikipedia mosquito | level and one development-selected transform | Related-topic increment |
| F6 | Digital combined | only families individually eligible before combination | Combined increment |
| F7 | Revision/status | source age, missing-source, revision magnitude where available | Monitoring/sensitivity, not primary biological signal |

Each family is versioned. Changing language sets, normalization, query terms, or aggregation rules creates a new version.

## 4. Models

| ID | Model | Status | Rationale |
| --- | --- | --- | --- |
| M0 | Seasonal naive | Required | Hard-to-beat public-health baseline |
| M1 | Regularized ARX | Primary | Interpretable and suitable for the small monthly sample |
| M2 | Poisson or negative-binomial regression | Challenger | Count-aware specification; use only if diagnostics support it |
| M3 | LightGBM with conservative complexity | Challenger | Nonlinear comparison; not eligible if effective sample is inadequate |
| M4 | Simple predeclared ensemble of eligible models | Optional | Only after individual models are frozen |

Deep neural and foundation models are excluded from paper 1.

## 5. Primary model-feature comparisons

Run on M1 first:

1. `F0` versus seasonal naive M0.
2. `F0 + F1` versus `F0`.
3. `F0 + F1 + F2` versus `F0 + F1`.
4. `F0 + F1 + F3` versus `F0 + F1`.
5. `F0 + F1 + F4` versus `F0 + F1`.
6. `F0 + F1 + F5` versus `F0 + F1`.
7. `F0 + F1 + eligible(F2:F5)` versus `F0 + F1`.

Repeat the frozen feature specifications with M2 and M3 only after M1 development decisions are recorded. Do not reopen feature selection separately for every model unless the comparison is explicitly labeled model-specific and remains inside development data.

## 6. Horizons and recursive design

- Horizon 1: one month ahead.
- Horizon 2: two months ahead, generated recursively when lagged outcomes are required.
- Report each horizon separately.
- A pooled result is secondary and weights horizons equally unless otherwise registered.
- Prediction intervals for horizon 2 must account for recursive uncertainty; a coefficient-only interval that treats the predicted lag as fixed is labeled incomplete.

## 7. Metrics

### Primary

- Weighted Interval Score using the registered quantiles.

### Secondary

- MAE in cases/activity units.
- Relative WIS and MAE versus seasonal naive.
- 50%, 80%, and 95% empirical coverage.
- Mean interval width.
- Bias and median error.
- Optional RMSLE for skew sensitivity, never as the only metric.

MAPE is descriptive only and must report zero-target exclusions.

## 8. Tuning

- All tuning occurs inside the historical development period.
- Use a small grid or fixed defaults; record every tried value.
- Select on mean inner-fold WIS.
- Feature transform and lag selection are tuning decisions.
- A failed configuration is retained with its error; it is not silently removed.
- Random seeds are fixed before confirmation and all registered seeds are reported.

## 9. Synthetic benchmark scenarios

| ID | Injected condition | Expected audit |
| --- | --- | --- |
| S0 | Clean delayed seasonal count process | No false critical findings |
| S1 | Future covariate copied into origin row | Leakage failure |
| S2 | Latest revised history substituted for old vintages | Vintage mismatch warning |
| S3 | Null feature with many post-hoc lags | Selection multiplicity warning; lower false promotion under honest pipeline |
| S4 | Two method names with identical predictions | Duplicate implementation/prediction warning |
| S5 | Seven favorable origins versus 100 common origins | Incomparable-grid warning |
| S6 | Crossing forecast quantiles | Probabilistic-format failure |
| S7 | Narrow under-covering intervals | Coverage failure |
| S8 | One extreme outbreak produces all gain | Event-concentration warning |
| S9 | Fetch failure overwrites history with null | Persistence/failure-handling failure |

Use repeated simulations with fixed master seeds. Report detection rate and false-promotion rate, not only one worked example.

## 10. Required result tables

1. Data/vintage coverage by source and snapshot.
2. Feature manifest and eligibility decisions.
3. Development screening and later stability.
4. Common-origin model/feature comparison by horizon.
5. WIS, MAE, coverage, and interval width.
6. Vintage optimism gap.
7. Robustness and drop-best-origin sensitivity.
8. Synthetic defect-detection and false-promotion rates.
9. Promotion decision table including negative results.

## 11. Analysis freeze

Before confirmation outcomes are opened, freeze and hash:

- source code commit;
- environment lock;
- forecast contract;
- source and feature manifests;
- experiment registry;
- model hyperparameters;
- transformations and lags;
- origins and target cases;
- random seeds;
- metrics and promotion rules.
