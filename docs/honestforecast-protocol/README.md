# HonestForecast protocol sprint

Status: protocol v0.1, frozen for review on 2026-11-05.

Implementation update: Milestones 1 and 2 plus the Milestone 3 repository-vintage pilot were completed on 2026-11-09 in the isolated `honestforecast/` subproject. The applied pilot retains negative feature-family results and remains exploratory because only seven scored origins exist.

This folder is the planning record for a public, clean-room forecasting research project. It does not modify the current dengue models, data updater, API, dashboard, or checked-in outputs.

## Proposed project

**GitHub project:** `HonestForecast` — a feature-aware, vintage-aware, and auditable forecasting research toolkit.

**Paper scope:** honest feature engineering for public-health nowcasting under delayed and revised surveillance data, demonstrated with public India dengue data and a synthetic delayed-surveillance benchmark.

The paper is intentionally narrower than the software. It asks whether digital-proxy feature gains survive realistic information-availability, development-only selection, rolling-origin probabilistic evaluation, and robustness gates.

## Frozen decisions

- CHOA and ERCOT data, results, predictions, dashboards, and internal code are excluded from the public project.
- India dengue is the public applied case; a synthetic hospital-surveillance study is the controlled case.
- Feature engineering is a first-class contribution, not a side effect of model comparison.
- The first release will use a small prespecified model set. It will not be a model zoo or AutoML framework.
- The primary probabilistic metric is WIS. MAE, coverage, interval width, and relative skill against seasonal naive are secondary.
- Hyperparameters and feature choices are selected only inside development data.
- Previously inspected rolling results are development evidence, not untouched confirmation.
- A real historical vintage is a saved as-of snapshot. Applying an assumed lag to the latest revised history is not equivalent.
- Claims are assigned one of: `promoted`, `challenger`, `exploratory`, `rejected`, or `not_comparable`.

## Repository audit informing this protocol

As of this review:

- `monthly_data.csv` has 64 rows from 2021-01 through 2026-04.
- WHO monthly cases are non-null for 26 of 64 rows.
- Both Google Trends columns are non-null for 63 rows; the three Wikipedia-derived columns are non-null for all 64 rows.
- `yearly_data.csv` has four national rows, 2021 through 2024.
- Git contains five commits touching the processed monthly/yearly tables between 2026-03-31 and 2026-04-20.
- `preprocessing_outputs/params_step1.json` records WHO through 2026-02 and external data through 2026-03, while the current processed table extends to 2026-04. Existing model artifacts therefore cannot be assumed current with the latest checked-in input.

These facts make a small reconstructed-vintage pilot feasible, but not a complete historical as-of replay. Prospective snapshots must begin at implementation freeze.

## Protocol package

- [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md): question, hypotheses, scope, data design, and estimands.
- [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md): prespecified models, feature families, evaluation regimes, and comparisons.
- [FEATURE_ENGINEERING_PROTOCOL.md](FEATURE_ENGINEERING_PROTOCOL.md): feature lifecycle and leakage-safe construction rules.
- [EVIDENCE_AND_PROMOTION_RULES.md](EVIDENCE_AND_PROMOTION_RULES.md): audit checks, uncertainty, evidence levels, and promotion gates.
- [WORKPLAN.md](WORKPLAN.md): implementation work packages, estimates, milestones, and cut lines.
- [OPEN_DECISIONS.md](OPEN_DECISIONS.md): decisions that require author or collaborator confirmation.
- `templates/`: initial machine-readable contracts and reporting templates.

## Exit criteria for this sprint

This protocol is ready to move into implementation only when:

1. the paper question remains narrower than the software;
2. publication/private-data boundaries are accepted;
3. the primary estimand and promotion rules are accepted before new model runs;
4. the vintage audit is performed before claiming historical as-of evaluation;
5. the first implementation milestone is limited to contract validation, feature availability, rolling-origin scoring, and one synthetic example.
