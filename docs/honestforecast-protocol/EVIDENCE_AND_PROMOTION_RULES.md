# Evidence and promotion rules

## 1. Audit gates

The pipeline evaluates gates in order. Failure of a higher-severity gate stops promotion even if later metrics look favorable.

### Gate A — task and data validity

- Forecast contract is complete.
- Target, unit, issue time, and horizon are unambiguous.
- Outcome-maturity rule is enforced.
- Source definitions and vintages are identified.
- Public/private boundary is satisfied.

Failure status: `not_comparable` or `ineligible`.

### Gate B — leakage and preprocessing

- No target-date or later observation is available at the origin.
- All preprocessing is fitted inside folds.
- Recursive lags use predictions rather than future outcomes.
- Revision and assumed-lag analyses are correctly labeled.
- Missing-source failures cannot become future-informed imputation.

Failure status: result invalid; no performance interpretation.

### Gate C — comparison integrity

- Candidate and baseline share identical origin, target, and horizon keys.
- Origin count and target-case count are disclosed.
- Method identity, source, package/version, and implementation type are recorded.
- Duplicate predictions and aliases are identified.
- No reduced-origin result is ranked as equal evidence to a full-grid result.

Failure status: `not_comparable` until recomputed on the intersection.

### Gate D — probabilistic validity

- Required quantiles exist and are monotone.
- Intervals are finite and ordered.
- Coverage and width are reported.
- A nominal interval is labeled by its true quantile endpoints.
- Recursive uncertainty limitations are stated.

Failure status: point forecast may remain evaluable; probabilistic promotion fails.

### Gate E — performance and robustness

- Primary paired WIS point estimate favors the candidate.
- At least 60% of registered origins favor the candidate.
- The 95% uncertainty interval for the mean paired difference excludes no improvement for `promoted`; otherwise the result can be `challenger`.
- Removing the best origin does not reverse the overall point estimate for `promoted`.
- Coverage degradation is no more than 5 percentage points at the registered nominal level unless the interval-width tradeoff was prespecified.
- Horizon-specific results do not hide a material loss in an operationally important horizon.
- Results are not dependent on one unregistered hyperparameter or transform cell.

### Gate F — confirmation

- Configuration was frozen before confirmation outcomes were opened.
- The confirmation vintage class is stated.
- All registered models are reported.
- No post-confirmation feature, threshold, or model selection is folded back into the same claim.

## 2. Decision states

| State | Required evidence | Permitted language |
| --- | --- | --- |
| `promoted` | Gates A–F pass for the intended claim | Improves the registered baseline in the confirmation design |
| `challenger` | A–D pass; favorable point estimate; uncertainty or confirmation incomplete | Promising candidate requiring more confirmation |
| `exploratory` | Valid descriptive/development result but selection or information regime is not confirmatory | Exploratory association or development gain |
| `rejected` | Valid comparison fails performance/robustness criteria | Did not improve under the tested representation |
| `not_comparable` | Keys, origins, task, or implementation do not support direct ranking | Cannot be directly compared |

`Rejected` applies to the tested representation, not every possible use of the source or model family.

## 3. Evidence levels

| Level | Evidence | What it establishes |
| --- | --- | --- |
| E0 | Narrative only | Claim exists; not independently checked |
| E1 | Hashes and manifest | Artifact bytes and membership are preserved |
| E2 | Aggregate arithmetic regeneration | Summary follows saved aggregate inputs |
| E3 | Report regeneration | Tables/figures follow the registered analysis code |
| E4 | Synthetic known-answer execution | Core code detects and handles controlled cases |
| E5 | Public-data refit | Public applied analysis can be rerun |
| E6 | Restricted-data refit | Authorized environment reproduces private analysis |
| E7 | Frozen prospective confirmation | Claim survives future, previously unopened outcomes |

The first public release targets E5 for Dengue and E4 for every critical audit. It must not imply E6 for CHOA or ERCOT.

## 4. Deterministic checks required in CI

- contract schema validation;
- feature lineage completeness;
- origin/target/horizon key uniqueness;
- common-grid intersection size;
- duplicated prediction hashes across method identities;
- quantile presence and monotonicity;
- finite metrics and prediction lengths;
- zero-target handling disclosure;
- availability timestamp not later than origin;
- train/tune/test date separation;
- report metrics recomputed from prediction-level evidence;
- public bundle denylist for known private project names and raw clinical artifact patterns.

The denylist is a safety net, not a substitute for manual review.

## 5. Evidence card contents

Each primary result receives an evidence card containing:

- forecast contract and data vintage;
- feature family and lineage;
- model identity and implementation provenance;
- evaluation grid and scored-case count;
- primary and secondary metrics;
- uncertainty and sensitivity results;
- gate outcomes and final decision state;
- reproducibility level;
- limitations and prohibited claims.

Use `templates/evidence_card.md`.

## 6. Stop rules

- Stop historical-vintage claims if available snapshots cannot establish what was known at origin; continue only as reconstructed/latest-revised sensitivity.
- Stop feature expansion after the registered families are tested; do not add a new family because the primary families are negative.
- Stop M3/LightGBM promotion if effective training size or fold stability is inadequate.
- Stop model/feature tuning when the registered grid is exhausted.
- Stop confirmation analysis after outcomes are opened; any subsequent change begins a new protocol version and future confirmation window.
- Stop public packaging if any private data, prediction-level clinical record, credential, or ambiguous-license artifact is present.
