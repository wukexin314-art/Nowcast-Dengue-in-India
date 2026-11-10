# India dengue repository-vintage pilot

Status: **exploratory**; scored origins: 7.

| Method | MAE | WIS | 80% coverage |
| --- | ---: | ---: | ---: |
| seasonal_naive | 910.7 | 692.4 | 1.000 |
| ar | 1846.4 | 1292.8 | 0.429 |
| ar_digital_asof | 2653.5 | 1921.9 | 0.286 |
| ar_digital_latest_revised | 3116.9 | 2430.0 | 0.286 |

## Registered decisions

- Target history versus seasonal naive: `rejected`.
- Digital proxies versus target history: `rejected`.

## Revision sensitivity

- Latest-revised minus as-of WIS: 508.2.
- Latest-revised minus as-of MAE: 463.4.

Negative values mean a retrospective latest-revised feature history looks better; positive values mean it looks worse. Either direction shows that feature revisions change the result.

## Limitations

- repository commit time is not source-native publication time
- only seven scored origins are available
- prediction intervals use in-sample residual quantiles
- no feature family is eligible for confirmation-level promotion
