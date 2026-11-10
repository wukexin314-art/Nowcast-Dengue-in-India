# Feature engineering protocol

## 1. Principle

A feature is not eligible because it correlates with the target. It must pass a chain of evidence:

`mechanism -> source identity -> forecast availability -> development association -> later stability -> conditional contribution -> incremental forecast gain -> robustness`

## 2. Feature states

Every candidate has exactly one current state:

- `proposed`: mechanism documented, not yet audited;
- `ineligible`: unavailable, unverifiable, prohibited, or structurally invalid;
- `screening`: availability passed, development evidence in progress;
- `candidate`: screening and stability passed;
- `challenger`: incremental point estimate is favorable but promotion evidence is incomplete;
- `promoted`: all promotion gates passed;
- `rejected`: tested and failed the registered criteria;
- `retired`: source definition changed or operational support ended.

State transitions are append-only records. Do not erase prior decisions after a source or transform changes.

## 3. Required manifest fields

- stable feature and family IDs;
- version;
- human description and unit;
- source, query/endpoint, geography, and aggregation;
- retrieval and publication semantics;
- earliest and latest coverage;
- missingness and revision behavior;
- minimum reporting lag;
- availability class: `known_ahead`, `observed_with_lag`, `forecast_required`, or `unverified`;
- expected mechanism, target, and direction;
- permitted horizons;
- raw-to-feature lineage;
- transformation parameters and fitting scope;
- privacy/publication class;
- current state and reason.

Use `templates/feature_manifest.csv` as the initial applied manifest.

## 4. Construction rules

### 4.1 Fold locality

Fit scaling, imputation, clipping, transformation choice, lag choice, and feature selection inside each training fold. A global z-score over the full study window is diagnostic only unless the transformation parameters are known in advance.

### 4.2 Reporting delays

At origin `o`, an observed value dated `d` is usable only when its actual publication timestamp is no later than `o`. When only a documented minimum lag is known, use the lag conservatively and label it an assumed-lag analysis.

### 4.3 Revised sources

- Use the snapshot that existed at the origin when available.
- Do not backfill old origins with values from a later snapshot in an as-of analysis.
- Version Google Trends queries and window definitions because full-window values can renormalize.
- Version Wikipedia language sets and normalization references.

### 4.4 Missingness

- Distinguish fetch failure, not-yet-published, structural absence, and genuine missing value.
- A failed fetch must not overwrite stored history with null.
- Missingness indicators may be features only if they would have been observable at the origin and were prespecified.

### 4.5 Recursive outcomes

For horizon 2, lagged target features use the horizon-1 prediction, not the future observation. Uncertainty propagation must reflect this substitution.

## 5. Allowed first-release transforms

Keep the search small and interpretable:

- identity;
- `log1p` for nonnegative skewed signals;
- signed `log1p` only if negative values are meaningful;
- one- and two-period lags where source timing permits;
- trailing 2- or 3-month mean;
- one-period momentum/difference;
- 12-month seasonal analogue when coverage permits;
- Fourier seasonality constructed from the target date;
- source age and revision magnitude for monitoring analyses.

Do not simultaneously search many windows, lags, nonlinear transforms, and interactions on the same small monthly sample.

## 6. Screening

Screening uses development data only and reports:

- usable observations and coverage;
- raw Spearman association;
- calendar/trend-adjusted association where estimable;
- direction by time block;
- circular-shift or time-preserving null comparison;
- FDR-adjusted results when multiple p-values are presented;
- source-lag eligibility;
- result sensitivity to latest-revised versus available-as-of values.

Screening nominates candidates. It cannot promote a feature.

## 7. Stability check

A candidate proceeds when:

- development coverage is at least 80%, unless a lower threshold was justified in advance;
- the sign does not reverse materially in the later development block;
- the effect is not isolated to one month/event;
- the source definition is consistent or explicitly versioned;
- the feature remains available for the intended horizon.

Small-sample uncertainty must be shown. Failure to meet a heuristic threshold is a reason for review, not permission to search for a more favorable unregistered threshold.

## 8. Conditional contribution

For eligible candidates:

- compare the same model with and without the feature family;
- use identical origins, target cases, preprocessing, and hyperparameters except where the increment logically requires a registered change;
- compute permutation or ablation effects only on untouched later-development predictions;
- preserve time structure in permutation tests;
- report model dependence and correlated-feature substitution.

SHAP or feature importance is descriptive. It does not replace an incremental forecast comparison.

## 9. Incremental forecast test

The primary comparison is baseline versus baseline plus one feature family. Combined-family models are evaluated only after individual eligibility decisions are recorded.

Report:

- paired WIS and MAE differences by origin and horizon;
- empirical coverage and interval width;
- proportion of origins improved;
- moving-block/origin bootstrap interval;
- drop-best-origin sensitivity;
- missing-source and revision sensitivity;
- latest-revised versus as-of difference.

## 10. Feature promotion

A feature family is `promoted` only when all rules in `EVIDENCE_AND_PROMOTION_RULES.md` pass. A favorable but uncertain family is a `challenger`; it is not silently included in the champion model.

Negative results remain in the feature registry with enough detail to prevent repeated unregistered searches.
