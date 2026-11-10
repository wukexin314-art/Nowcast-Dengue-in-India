# Open decisions

Blocking decisions should be resolved before implementation or before the specified milestone. Defaults below are recommendations, not hidden assumptions.

| ID | Decision | Recommended default | Needed by | Status |
| --- | --- | --- | --- | --- |
| D01 | Public repository name | `honestforecast` | Milestone 1 | Open |
| D02 | License | Apache-2.0 for code; CC BY 4.0 for documentation where appropriate | Milestone 1 | Open |
| D03 | Paper emphasis | Health feature engineering and vintage-aware evaluation; toolkit as artifact | Protocol approval | Recommended |
| D04 | Confirmation window | First six newly acquired WHO-observed target months after freeze, correctly labeled retrospective/prospective | Before analysis freeze | Recommended |
| D05 | Primary metric | WIS, with MAE and coverage secondary | Protocol approval | Recommended |
| D06 | Promotion threshold | Favorable WIS, >=60% origins improved, 95% interval supports improvement, drop-best-origin remains favorable, coverage loss <=5 pp | Protocol approval | Recommended |
| D07 | Historical Git snapshots | Audit each commit before assigning `historical_snapshot`; otherwise classify reconstructed | Milestone 3 | Required |
| D08 | Google Trends storage | Preserve full raw response or canonical snapshot per run; never replace prior vintages | Milestone 3 | Required |
| D09 | Wikipedia normalized feature | Choose raw versus normalized in development; version the normalization reference | Milestone 3 | Open |
| D10 | OpenDengue role | Annual constraint/sensitivity only unless release vintage is verified | Milestone 3 | Recommended |
| D11 | Count model | Negative binomial preferred if dispersion diagnostics support it; otherwise Poisson/quasi-Poisson | Milestone 4 | Open |
| D12 | LightGBM eligibility | Include only after a minimum effective-sample and stability review | Milestone 4 | Open |
| D13 | Authorship and data-source ownership | Record lead, software, methods, public-health interpretation, and maintenance roles | Before manuscript | Open |
| D14 | Public archive/DOI | Archive tagged release after v0.1 and artifact verification | Milestone 7 | Open |

## Review questions

1. Is the main paper question narrow enough to remain a public-health methods/application paper rather than a generic software paper?
2. Is six newly acquired WHO target months sufficient for the first confirmation report, or should the project wait for twelve?
3. Is WIS acceptable as primary given the current interval-generation limitations, or should MAE remain co-primary until probabilistic calibration improves?
4. Should the first applied release include only M0/M1, with M2/M3 deferred to avoid small-sample model shopping?
5. Are the promotion thresholds intentionally strict enough for the number of origins available?
6. Who will provide domain review of surveillance-source semantics and public-health claims?

## Nonblocking future questions

- Add state-level India dengue data if a stable, versioned public source becomes available.
- Add a second public disease only after the first paper is frozen.
- Extend evidence packaging to authorized private-data studies without copying their empirical content.
- Add model identity/alias discovery after the core common-grid evaluator is stable.
