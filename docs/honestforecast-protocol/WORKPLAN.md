# Work plan and workload

## 1. Planning assumption

This document separates the current protocol sprint from later implementation. Hour estimates are focused-work estimates, not elapsed-time claims.

## 2. Protocol sprint: 12–18 hours equivalent

| Work package | Hours | Deliverable |
| --- | ---: | --- |
| Audit project structure, data coverage, Git snapshot depth, and publication boundaries | 2.0–3.0 | Repository audit in `README.md` |
| Freeze research questions, hypotheses, estimands, and exclusions | 2.0–2.5 | `RESEARCH_PROTOCOL.md` |
| Define feature families, models, regimes, horizons, and metrics | 2.0–2.5 | `EXPERIMENT_MATRIX.md` |
| Define feature lifecycle and availability rules | 1.5–2.5 | `FEATURE_ENGINEERING_PROTOCOL.md` |
| Define evidence levels, audit gates, promotion and stop rules | 2.0–2.5 | `EVIDENCE_AND_PROMOTION_RULES.md` |
| Create machine-readable starter templates | 1.0–1.5 | `templates/` |
| Sequence implementation, risks, decisions, and review checklist | 1.5–3.5 | `WORKPLAN.md`, `OPEN_DECISIONS.md` |
| **Total** | **12.0–18.0** | Reviewable protocol package |

## 3. Implementation roadmap

### Milestone 0 — protocol approval

Effort: 4–8 hours of author/collaborator review.

Exit criteria:

- approve paper scope and wording;
- assign authorship/data-source owners;
- decide repository name and license;
- accept promotion thresholds;
- resolve the open decisions marked blocking.

### Milestone 1 — clean-room core and synthetic vertical slice

Effort: 45–65 hours.

Build only:

- project skeleton and CI;
- Pydantic/dataclass schemas for contracts and manifests;
- rolling-origin key generation;
- common-grid comparison;
- WIS, MAE, coverage, and quantile checks;
- one synthetic dataset with S0, S1, S5, and S6;
- evidence-card generation.

Exit criteria:

- `pip install -e .` succeeds in a clean environment;
- synthetic example runs from one command;
- injected leakage, unequal origins, and crossing quantiles fail deterministically;
- metrics regenerate from prediction-level evidence.

### Milestone 2 — feature engineering engine

Effort: 45–65 hours.

Status: completed locally on 2026-11-09; verification and scope are recorded in `honestforecast/MILESTONE_2.md`.

Build:

- feature manifest validation;
- as-of availability joins;
- fold-local preprocessing;
- limited transforms;
- development screening;
- feature-family ablation;
- lineage output and decision log.

Exit criteria:

- no transform is fitted on later data;
- every output feature maps to source, version, and availability rule;
- rejected families remain in the registry;
- synthetic null-feature false promotion can be measured.

### Milestone 3 — Dengue vintage pilot

Effort: 45–70 hours.

Status: repository-vintage extraction, updater health, minimal M0/M1 comparison, and revision sensitivity completed on 2026-11-09. Full reproduction of the legacy notebook outputs remains separate because the pilot intentionally uses a smaller registered model.

Tasks:

- audit all 18 Git-preserved snapshots currently available;
- extract source coverage and revision deltas;
- classify each snapshot as historical or reconstructed;
- create an append-only prospective snapshot process;
- adapt M0 and M1 to the new contracts;
- reproduce current rolling metrics on an explicitly labeled development grid;
- identify discrepancies between current inputs and checked-in model outputs.

Exit criteria:

- all data artifacts have vintage class and hash;
- no claim exceeds the available vintage evidence;
- baseline reproduction differences are explained;
- the current input/output freshness mismatch is surfaced automatically.

### Milestone 4 — applied feature experiments

Effort: 55–80 hours.

Tasks:

- run registered M1 feature increments;
- freeze selected transforms inside development data;
- add M2/M3 only if data-size gates pass;
- compute horizon-specific WIS/MAE/coverage;
- run paired and drop-best-origin sensitivity;
- compare R0/R1/R2 regimes.

Exit criteria:

- all primary tables are generated from registered experiments;
- common-grid counts are identical within each comparison;
- every feature family has a decision state;
- negative results are retained.

### Milestone 5 — complete synthetic benchmark

Effort: 30–45 hours.

Add S2–S4 and S7–S9, repeated seeds, detection rates, false-promotion rates, and failure documentation.

Exit criteria:

- each critical audit has a known-answer test;
- clean scenarios do not trigger unacceptable false critical errors;
- naive and honest pipelines are compared on identical simulations.

### Milestone 6 — skill, documentation, and public artifact

Effort: 30–45 hours.

Build:

- concise `honest-forecast-research` skill;
- references for design, feature audit, comparison, and packaging modes;
- quickstart and two end-to-end examples;
- CI, environment lock, license, contribution guidance;
- public-safe artifact scan and verification receipt.

Exit criteria:

- a new user can run the synthetic example in 15 minutes;
- the skill produces or validates the same contracts as the package;
- the repository contains no CHOA/ERCOT artifacts.

### Milestone 7 — manuscript and release

Effort: 65–95 hours.

Tasks:

- freeze analysis and regenerate all figures/tables;
- draft methods, results, limitations, and artifact appendix;
- internal review and reruns;
- tag v0.1, archive release, and produce reproducibility instructions;
- begin sealed/prospective confirmation tracking.

## 4. Total workload

| Scope | Estimated hours |
| --- | ---: |
| Protocol sprint | 12–18 |
| Minimal public vertical slice through Milestone 2 | 90–138 after protocol |
| GitHub MVP through Dengue pilot | 135–208 after protocol |
| Paper-ready retrospective package | 315–465 total including protocol |
| Prospective confirmation | Calendar-dependent; approximately 2–4 hours per update plus final analysis |

At 15 focused hours/week, a paper-ready retrospective package is roughly five to eight months. Prospective confirmation adds waiting time even when labor is small.

## 5. Priority and cut lines

### Must ship in v0.1

- contract and feature-manifest schemas;
- availability/vintage classification;
- rolling-origin common-grid evaluator;
- WIS/MAE/coverage;
- feature-family ablation;
- core robustness and promotion decisions;
- synthetic known-answer study;
- Dengue development case;
- tests, CI, license, and evidence cards.

### Defer if time is constrained

- LightGBM and ensembles;
- interactive dashboard;
- plugin architecture;
- additional countries/diseases;
- automated source acquisition beyond the current updater;
- extensive CLI styling;
- large method registry;
- publication-quality web application.

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Historical snapshots are too shallow for a strong vintage claim | High | High | Treat as pilot; start prospective capture; use synthetic benchmark for causal demonstration |
| Monthly sample is too small for complex models | High | Medium | Keep M0/M1 primary; use conservative challengers and simple grids |
| WHO source changes or remains fragile | Medium | High | Save raw/canonical snapshots, parser version, failure status, and preserve old values |
| Google Trends renormalization changes history | High | High | Version every fetch; compare revisions; never overwrite prior snapshots |
| Framework scope expands into AutoML | Medium | High | Enforce v0.1 cut lines and small experiment matrix |
| Null feature results tempt post-hoc searching | Medium | High | Freeze families/transforms; new ideas require protocol v0.2 and future confirmation |
| Private artifacts enter public repo | Low–medium | Critical | Clean-room repo, denylist scan, manifest review, no copying from private folders |
| Probabilistic intervals are under-dispersed | High | Medium | Report coverage/width; add bootstrap or simulation; label incomplete recursion intervals |

## 7. Practical value

The MVP is useful if it lets a researcher answer, with evidence:

- what was knowable at each origin;
- which feature transformation was selected where;
- whether two results share the same scored cases;
- whether uncertainty is calibrated;
- whether a gain is stable or event-concentrated;
- what level of reproduction has actually been achieved;
- whether a model is champion, challenger, exploratory, rejected, or incomparable.

Its differentiation is the research control plane around forecasting models, not new forecasting algorithms.
