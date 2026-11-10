# Milestone 3 pilot record

Completed locally on 2026-11-09. This is the repository-vintage pilot portion of Milestone 3, not a confirmation study.

## Vintage evidence

- 18 committed `monthly_data.csv` snapshots from 2026-03-31 through 2026-11-05;
- every snapshot hashed and classified;
- 1,543 historical revision events across 203 unique feature-period pairs;
- zero updates that changed a previously non-null value to null;
- updater current as of 2026-11-09, with the last successful scheduled run on 2026-11-05;
- explicit limitation: Git state proves repository bytes, not upstream source publication time.

## Minimal applied comparison

Seven origins have both a saved repository state and later-matured WHO outcomes. Every method uses the same origin/target/horizon grid.

| Method | MAE | WIS | 80% coverage | Decision |
| --- | ---: | ---: | ---: | --- |
| Seasonal naive | 910.7 | 692.4 | 1.000 | Reference |
| Lag-12 + Fourier ridge | 1846.4 | 1292.8 | 0.429 | Rejected |
| Ridge + as-of digital proxies | 2653.5 | 1921.9 | 0.286 | Rejected |
| Ridge + latest-revised proxies | 3116.9 | 2430.0 | 0.286 | Diagnostic only |

Latest-revised minus origin-matched digital-proxy performance is +508.2 WIS and +463.4 MAE, so using later revisions changes and worsens the result in this pilot.

## Important implementation finding

Lag-1 WHO activity is not always available at issue time because reporting delays sometimes exceed one month. The pilot therefore uses lag 12 plus fixed Fourier terms. It does not silently substitute a future lag-1 observation or an unregistered recursive estimate.

## Evidence level and limitations

- Status: exploratory.
- Evidence: E4 software known-answer checks plus repository-state historical pilot.
- Only seven scored origins are available.
- Prediction intervals use centered in-sample residual quantiles.
- No feature family is eligible for confirmation-level promotion.
- A longer prospective window is required before a paper-level effectiveness claim.

## Reproduction

```bash
honestforecast-vintage-audit \
  --repo . \
  --data-path dengue_updater/data/processed/monthly_data.csv \
  --update-log dengue_updater/logs/update_log.csv \
  --as-of 2026-11-09 \
  --output-dir honestforecast/case_studies/india_dengue/vintage_audit

honestforecast-dengue-pilot \
  --repo . \
  --data-path dengue_updater/data/processed/monthly_data.csv \
  --output-dir honestforecast/case_studies/india_dengue/pilot
```
