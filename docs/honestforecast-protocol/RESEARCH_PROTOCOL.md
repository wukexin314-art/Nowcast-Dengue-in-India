# Research protocol

## 1. Working title

**Honest feature engineering for public-health nowcasting under delayed and revised surveillance data**

Software artifact: **HonestForecast**.

## 2. Research objective

Estimate how much apparent predictive value from digital-proxy features remains after enforcing the information set that would have existed at each forecast origin, selecting features only in development data, and evaluating point and probabilistic forecasts on common rolling-origin cases.

The applied study uses monthly dengue activity in India. The controlled study uses synthetic delayed surveillance with known data-generating mechanisms and injected failure modes.

## 3. Primary research questions

1. How different are feature and model conclusions under latest-revised, lag-safe reconstructed-vintage, and prospectively captured-vintage evaluation?
2. Which digital-proxy feature families add out-of-sample value beyond target history and seasonality?
3. How stable are those gains across one- and two-month horizons, forecast origins, and plausible preprocessing choices?
4. Can an evidence-aware audit detect leakage, incomparable origin grids, duplicated implementations, interval defects, and event-concentrated gains?

## 4. Hypotheses

- **H1 — vintage optimism:** evaluation on the latest revised history will be at least as favorable as evaluation on as-of snapshots for one or more feature families. Direction and magnitude are empirical; no nonzero effect is assumed.
- **H2 — screening attrition:** some features with favorable development-period association will fail later-period stability or incremental forecast tests.
- **H3 — proxy value:** at least one prespecified digital-proxy family may improve the ARX baseline, but it will be promoted only if the improvement passes the full gate.
- **H4 — comparison integrity:** unequal-origin and post-hoc comparisons will produce more apparent wins than common-grid, development-only comparisons in the synthetic benchmark.
- **H5 — audit effectiveness:** deterministic injected defects will be detected by the corresponding audit checks before model promotion.

H3 is deliberately noncommittal. A well-supported null or negative result remains a valid outcome.

## 5. Intended contributions

### Scientific

- Quantify the optimism gap introduced by revised or unavailable-at-origin covariates.
- Evaluate digital-proxy feature families through a mechanism-to-promotion funnel rather than correlation alone.
- Report probabilistic skill, calibration, and horizon degradation under a common rolling-origin design.

### Methodological

- A machine-readable forecast contract and feature availability manifest.
- A common-grid comparison rule with explicit method identity and implementation provenance.
- Prespecified promotion gates resistant to post-hoc minima and single-event dominance.
- A layered evidence model for public and restricted-data forecasting studies.

### Software

- A small Python package and CLI that validates contracts, constructs as-of features, scores common-origin predictions, runs robustness checks, and emits evidence cards.
- A Codex skill that uses the same contracts and does not embed project-specific private material.

## 6. Scope and exclusions

### Included

- Monthly India dengue nowcasting.
- One- and two-month horizons.
- WHO monthly observations, Google Trends, Wikipedia dengue, Wikipedia mosquito, and OpenDengue annual constraints where available.
- Seasonal naive, regularized ARX, a count regression challenger, and a tree-based challenger if sample-size gates are met.
- Synthetic delayed/revised surveillance with controlled injected defects.

### Excluded from the first paper

- CHOA or ERCOT empirical data and results.
- Patient-level or clinical decision prediction.
- Large neural/foundation model benchmarking.
- Automatic discovery of hundreds of transformations.
- Claims of causal effects from digital proxies.
- Claims that nowcasts replace official surveillance.
- Multi-country generalization unless added in a later, separately registered extension.

## 7. Forecast contract

The applied task is defined by `templates/forecast_contract.example.yaml`.

At every origin, the pipeline must distinguish:

- `who_observed_through`: last month with an official WHO value available in that vintage;
- `external_data_through`: last month with all features required by the current specification;
- `nowcast_available_through`: last month for which a forecast can be generated;
- `outcome_mature_through`: last target month permitted for model fitting after the outcome-maturity rule.

The values may differ by vintage and experiment. They must be derived, not hard-coded from the current repository state.

## 8. Data and vintage design

### 8.1 Source ledger

For every acquisition, save or record:

- source and retrieval endpoint;
- retrieval timestamp in UTC;
- raw or canonical snapshot path;
- SHA-256 checksum;
- parser version or commit;
- first and last usable period;
- non-null coverage by variable;
- revision relative to the prior snapshot;
- fetch status and failure reason;
- whether the snapshot was truly observed as-of or reconstructed later.

### 8.2 Vintage classes

| Class | Meaning | Permitted claim |
| --- | --- | --- |
| `prospective_snapshot` | Saved before later outcomes were examined and retained unchanged | Prospective/as-of evidence |
| `historical_snapshot` | Verifiable file or Git object created at that historical date | Historical as-of evidence |
| `reconstructed_snapshot` | Rebuilt later from records that preserve some, but not all, historical state | Quasi-as-of sensitivity |
| `latest_revised` | Current source values used for all historical dates | Retrospective diagnostic only |
| `assumed_lag` | Latest history shifted by an assumed reporting delay | Availability sensitivity, not vintage replay |

The five existing processed-data Git commits are candidates for `historical_snapshot` only after their contents, dates, and source semantics are audited.

### 8.3 Data freezing

- Hash each input vintage and experiment configuration.
- Never overwrite a prior snapshot.
- Store data-source failure separately from real missing values.
- Google Trends full-window renormalization is treated as a revision event, not an ordinary appended observation.
- A source definition change, such as Wikipedia language coverage, starts a new feature version.

## 9. Evaluation periods

The current repository has already exposed rolling results through 2026-02. Those dates are development evidence.

### Retrospective development

- Use WHO-observed targets through 2026-02 for pipeline development, debugging, feature-family screening, and model selection.
- Inner splits must preserve time order.
- Previously inspected results cannot be relabeled as untouched confirmation.

### Sealed confirmation

- After implementation and configuration freeze, acquire the first six newly available WHO target months that were not used for tuning in this project.
- If any target values were already publicly available before freeze, label the evaluation `sealed retrospective` rather than `prospective`.
- Do not revise features, transformations, hyperparameters, thresholds, or promotion rules after confirmation outcomes are opened.

### Continuing prospective evaluation

- Append future vintages and predictions without editing prior records.
- Report all registered models, not only the best current model.
- A minimum confirmation window is six observed target months. Longer observation is preferable because monthly samples are small.

## 10. Primary estimands

### Applied feature increment

For feature family `F` over common scored cases:

`Delta_WIS(F) = mean(WIS_baseline - WIS_baseline_plus_F)`

Positive values indicate improvement. The report must also provide paired per-origin differences, MAE change, coverage change, interval-width change, and one- versus two-month results.

### Vintage optimism gap

For the same model, feature set, and target cases:

`Optimism_gap = Skill_latest_revised - Skill_as_of`

The metric definition and denominator must be identical across regimes. A reconstructed snapshot must not be presented as a prospective vintage.

### Synthetic false-promotion rate

Across repeated simulated datasets:

`false_promotion_rate = promoted null/noise feature families / evaluated null/noise feature families`

Compare naive and honest pipelines using the same simulations.

## 11. Statistical analysis

- Unit of pairing: forecast origin and horizon on an identical target case.
- Preserve repeated/overlapping forecast structure; do not count every horizon row as independent.
- Report the distribution of paired origin-level differences.
- Use moving-block or origin-level bootstrap appropriate to the evaluation grid.
- Report confidence intervals without making significance the sole promotion criterion.
- Always run drop-best-origin and, when relevant, drop-best-event sensitivity.
- Report results by horizon before pooling.
- Correct multiple feature-screening tests with FDR when inferential p-values are shown.

## 12. Reproducibility and privacy

The public release must reproduce the synthetic and dengue studies from public inputs. Restricted projects motivate the protocol but contribute no public empirical artifacts.

Reproducibility claims are layered: byte integrity, aggregate arithmetic, report regeneration, synthetic execution, public-data refit, private-data refit, and prospective confirmation. Passing a lower level does not imply passing a higher one.

## 13. Ethics and communication

- Describe the output as research-stage situational awareness.
- Do not present digital traces as direct disease counts without model uncertainty and limitations.
- Discuss selection, language, internet-access, media-attention, and platform-normalization biases.
- Do not imply clinical or individual-level validity.
- Retain negative results and source failures.
