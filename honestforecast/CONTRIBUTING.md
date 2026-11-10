# Contributing

HonestForecast accepts focused changes that strengthen point-in-time validity, feature evidence, probabilistic comparison, or reproducibility.

## Before opening a change

1. State the forecast origin and information-availability contract.
2. Add a failing test for new audit logic or a known-answer scenario for a new failure mode.
3. Keep model-specific code outside the core unless it enforces a general evidence rule.
4. Do not include private data, credentials, patient-level records, or artifacts with unclear redistribution rights.
5. Preserve negative feature decisions and comparable evaluation grids.

## Local checks

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
honestforecast-synthetic --output-dir build/synthetic
python scripts/public_safety_scan.py
python setup.py check --strict
```

New applied examples must state whether their data are latest-revised, lag-masked, reconstructed from saved snapshots, or prospectively captured.
