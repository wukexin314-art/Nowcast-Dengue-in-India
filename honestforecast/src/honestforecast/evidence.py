"""Minimal human-readable evidence card generation."""

from __future__ import annotations

from typing import Mapping


def render_evidence_card(
    study_id: str,
    experiment_id: str,
    metrics: Mapping[str, float],
    scenario_results: Mapping[str, str],
) -> str:
    scenarios = "\n".join(
        f"- {scenario}: {result}" for scenario, result in sorted(scenario_results.items())
    )
    return f"""# Forecast evidence card

## Identity

- Study: `{study_id}`
- Experiment: `{experiment_id}`
- Decision state: `exploratory`
- Evidence level: `E4 synthetic known-answer execution`

## Evaluation grid

- Common scored cases: {int(metrics['n_cases'])}
- Horizons: 1 and 2 months

## Metrics regenerated from prediction rows

- MAE: {metrics['mae']:.6f}
- WIS: {metrics['wis']:.6f}
- 50% coverage: {metrics['coverage_50']:.3f}
- 80% coverage: {metrics['coverage_80']:.3f}
- 95% coverage: {metrics['coverage_95']:.3f}
- 95% mean interval width: {metrics['width_95']:.3f}

## Known-answer audit outcomes

{scenarios}

## Interpretation

This card verifies core mechanics on synthetic evidence. It does not establish public-health forecast skill or a clinical claim.
"""
