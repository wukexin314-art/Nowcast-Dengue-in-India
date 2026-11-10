"""Audit Git-preserved data vintages and quantify historical revisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class VintageError(ValueError):
    """Raised when vintage evidence is missing or internally inconsistent."""


@dataclass(frozen=True)
class GitSnapshot:
    commit: str
    committed_at: str
    subject: str
    content: bytes


@dataclass(frozen=True)
class SnapshotAudit:
    commit: str
    committed_at: str
    subject: str
    sha256: str
    row_count: int
    first_period: str
    last_period: str
    added_rows: int
    removed_rows: int
    added_cells: int
    removed_cells: int
    revised_cells: int
    retroactive_revised_cells: int
    schema_added_fields: Tuple[str, ...]
    schema_removed_fields: Tuple[str, ...]
    vintage_class: str
    claim_limit: str
    non_null_by_field: Mapping[str, int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _git(repo: Path, arguments: Sequence[str]) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments], stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", b"").decode("utf-8", errors="replace").strip()
        raise VintageError(f"git command failed: {detail or exc}") from exc


def collect_git_snapshots(repo: Path, data_path: str) -> List[GitSnapshot]:
    """Read every committed version of one CSV, oldest first."""
    repo = Path(repo).resolve()
    log = _git(
        repo,
        ["log", "--reverse", "--format=%H%x1f%cI%x1f%s", "--", data_path],
    ).decode("utf-8")
    snapshots = []
    for line in log.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f", 2)
        if len(parts) != 3:
            raise VintageError(f"unexpected git log row: {line}")
        commit, committed_at, subject = parts
        content = _git(repo, ["show", f"{commit}:{data_path}"])
        snapshots.append(GitSnapshot(commit, committed_at, subject, content))
    if not snapshots:
        raise VintageError(f"no committed snapshots found for {data_path}")
    return snapshots


def _csv_rows(content: bytes, key_field: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise VintageError("snapshot CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or key_field not in reader.fieldnames:
        raise VintageError(f"snapshot CSV requires key field {key_field}")
    rows: Dict[str, Dict[str, str]] = {}
    for row in reader:
        key = str(row.get(key_field, "")).strip()
        if not key:
            raise VintageError(f"blank {key_field} in snapshot")
        if key in rows:
            raise VintageError(f"duplicate {key_field}: {key}")
        rows[key] = {field: str(row.get(field, "")).strip() for field in reader.fieldnames}
    if not rows:
        raise VintageError("snapshot CSV is empty")
    return list(reader.fieldnames), rows


def audit_snapshots(
    snapshots: Iterable[GitSnapshot], key_field: str = "date"
) -> Tuple[List[SnapshotAudit], List[Dict[str, object]]]:
    """Return per-snapshot coverage and per-field revision evidence."""
    audits: List[SnapshotAudit] = []
    revisions: List[Dict[str, object]] = []
    previous_fields: Optional[List[str]] = None
    previous_rows: Optional[Dict[str, Dict[str, str]]] = None

    for snapshot in snapshots:
        fields, rows = _csv_rows(snapshot.content, key_field)
        previous_field_set = set(previous_fields or fields)
        current_field_set = set(fields)
        schema_added = tuple(sorted(current_field_set - previous_field_set))
        schema_removed = tuple(sorted(previous_field_set - current_field_set))
        added_keys = set(rows) if previous_rows is None else set(rows) - set(previous_rows)
        removed_keys = set() if previous_rows is None else set(previous_rows) - set(rows)
        counts = {"added": 0, "removed": 0, "revised": 0, "retroactive": 0}
        if previous_rows is not None:
            prior_last_period = max(previous_rows)
            for key in sorted(set(rows) & set(previous_rows)):
                for field in sorted(current_field_set & previous_field_set):
                    if field == key_field:
                        continue
                    before, after = previous_rows[key][field], rows[key][field]
                    if before == after:
                        continue
                    if not before and after:
                        change_type = "added"
                    elif before and not after:
                        change_type = "removed"
                    else:
                        change_type = "revised"
                    counts[change_type] += 1
                    retroactive = key <= prior_last_period
                    if retroactive and change_type == "revised":
                        counts["retroactive"] += 1
                    revisions.append(
                        {
                            "commit": snapshot.commit,
                            "committed_at": snapshot.committed_at,
                            "period": key,
                            "field": field,
                            "change_type": change_type,
                            "before": before,
                            "after": after,
                            "retroactive": retroactive,
                            "numeric_absolute_change": _absolute_change(before, after),
                        }
                    )
        non_null = {
            field: sum(bool(row[field]) for row in rows.values())
            for field in fields
            if field != key_field
        }
        scheduled = snapshot.subject.startswith("auto: update dengue data")
        audits.append(
            SnapshotAudit(
                commit=snapshot.commit,
                committed_at=snapshot.committed_at,
                subject=snapshot.subject,
                sha256=hashlib.sha256(snapshot.content).hexdigest(),
                row_count=len(rows),
                first_period=min(rows),
                last_period=max(rows),
                added_rows=len(added_keys),
                removed_rows=len(removed_keys),
                added_cells=counts["added"],
                removed_cells=counts["removed"],
                revised_cells=counts["revised"],
                retroactive_revised_cells=counts["retroactive"],
                schema_added_fields=schema_added,
                schema_removed_fields=schema_removed,
                vintage_class=(
                    "scheduled_git_snapshot" if scheduled else "historical_git_snapshot"
                ),
                claim_limit="repository state; source-native publication time not proven",
                non_null_by_field=non_null,
            )
        )
        previous_fields, previous_rows = fields, rows
    return audits, revisions


def _absolute_change(before: str, after: str) -> object:
    try:
        result = abs(float(after) - float(before))
    except ValueError:
        return ""
    return result if math.isfinite(result) else ""


def summarize_revisions(
    revisions: Iterable[Mapping[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in revisions:
        grouped.setdefault(str(row["field"]), []).append(row)
    summary = []
    for field, rows in sorted(grouped.items()):
        revised = [row for row in rows if row["change_type"] == "revised"]
        magnitudes = [
            float(row["numeric_absolute_change"])
            for row in revised
            if row["numeric_absolute_change"] != ""
        ]
        summary.append(
            {
                "field": field,
                "revision_events": len(revised),
                "unique_periods_revised": len({str(row["period"]) for row in revised}),
                "added_events": sum(row["change_type"] == "added" for row in rows),
                "removed_events": sum(row["change_type"] == "removed" for row in rows),
                "mean_absolute_revision": (
                    statistics.mean(magnitudes) if magnitudes else ""
                ),
                "median_absolute_revision": (
                    statistics.median(magnitudes) if magnitudes else ""
                ),
                "max_absolute_revision": max(magnitudes) if magnitudes else "",
            }
        )
    return summary


def summarize_update_log(
    path: Path, *, as_of: str, expected_interval_days: int = 20
) -> Dict[str, object]:
    """Summarize source reliability and scheduled-update freshness."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise VintageError("update log is empty")
    try:
        as_of_date = date.fromisoformat(as_of)
        run_times = [datetime.fromisoformat(str(row["run_time"])) for row in rows]
    except (KeyError, ValueError) as exc:
        raise VintageError("update log requires ISO run_time values") from exc
    last_run = max(run_times)
    stale_days = (as_of_date - last_run.date()).days
    by_source: Dict[str, Dict[str, int]] = {}
    for row in rows:
        source = str(row.get("source_name", "unknown"))
        status = str(row.get("status", "unknown"))
        by_source.setdefault(source, {})[status] = by_source.setdefault(source, {}).get(status, 0) + 1
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "as_of": as_of,
        "unique_runs": len({str(row["run_time"]) for row in rows}),
        "log_rows": len(rows),
        "last_run_at": last_run.isoformat(),
        "stale_days": stale_days,
        "expected_interval_days": expected_interval_days,
        "freshness_status": "stale" if stale_days > expected_interval_days else "current",
        "status_counts": status_counts,
        "source_status_counts": by_source,
    }


def _write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    data_path: str,
    audits: Sequence[SnapshotAudit],
    revisions: Sequence[Mapping[str, object]],
    updater_health: Optional[Mapping[str, object]] = None,
) -> str:
    removed_cells = sum(audit.removed_cells for audit in audits)
    revised_cells = sum(audit.revised_cells for audit in audits)
    lines = [
        "# Dengue vintage audit",
        "",
        f"- Source path: `{data_path}`",
        f"- Git-preserved snapshots: {len(audits)}",
        f"- Snapshot window: {audits[0].committed_at} to {audits[-1].committed_at}",
        f"- Historical revision events: {revised_cells}",
        f"- Previously non-null cells removed: {removed_cells}",
        "- Vintage class: repository-state evidence, not source-native publication timestamps",
        "",
        "## Snapshot manifest",
        "",
        "| Commit | Committed at | Rows | Through | Revised cells | Removed cells | Class |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for audit in audits:
        lines.append(
            f"| `{audit.commit[:8]}` | {audit.committed_at} | {audit.row_count} | "
            f"{audit.last_period} | {audit.revised_cells} | {audit.removed_cells} | "
            f"{audit.vintage_class} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Each Git object proves that these exact bytes existed by its commit time. "
            "It does not prove when the upstream provider first published each value. "
            "Use these snapshots for a repository-state historical pilot and revision "
            "sensitivity analysis; do not call them source-native real-time vintages.",
            "",
            f"The cell-level revision file contains {len(revisions)} change events.",
        ]
    )
    if updater_health is not None:
        lines.extend(
            [
                "",
                "## Updater health",
                "",
                f"- Last logged run: {updater_health['last_run_at']}",
                f"- Freshness as of {updater_health['as_of']}: "
                f"{updater_health['freshness_status']} ({updater_health['stale_days']} days)",
                f"- Status counts: `{json.dumps(updater_health['status_counts'], sort_keys=True)}`",
            ]
        )
    return "\n".join(lines) + "\n"


def run_vintage_audit(
    repo: Path,
    data_path: str,
    output_dir: Path,
    key_field: str = "date",
    update_log: Optional[Path] = None,
    as_of: Optional[str] = None,
) -> Dict[str, object]:
    snapshots = collect_git_snapshots(repo, data_path)
    audits, revisions = audit_snapshots(snapshots, key_field)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for audit in audits:
        row = audit.to_dict()
        row["non_null_by_field"] = json.dumps(row["non_null_by_field"], sort_keys=True)
        row["schema_added_fields"] = json.dumps(row["schema_added_fields"])
        row["schema_removed_fields"] = json.dumps(row["schema_removed_fields"])
        manifest_rows.append(row)
    _write_csv(output_dir / "snapshot_manifest.csv", manifest_rows)
    _write_csv(output_dir / "cell_revisions.csv", list(revisions))
    _write_csv(output_dir / "revision_summary.csv", summarize_revisions(revisions))
    updater_health = None
    if update_log is not None:
        if as_of is None:
            raise VintageError("as_of is required when update_log is supplied")
        updater_health = summarize_update_log(update_log, as_of=as_of)
        (output_dir / "updater_health.json").write_text(
            json.dumps(updater_health, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    revision_events = [row for row in revisions if row["change_type"] == "revised"]
    receipt = {
        "data_path": data_path,
        "snapshot_count": len(audits),
        "first_snapshot_at": audits[0].committed_at,
        "last_snapshot_at": audits[-1].committed_at,
        "latest_period": audits[-1].last_period,
        "historical_revision_events": len(revision_events),
        "unique_revised_feature_periods": len(
            {(str(row["field"]), str(row["period"])) for row in revision_events}
        ),
        "removed_non_null_cells": sum(audit.removed_cells for audit in audits),
        "latest_sha256": audits[-1].sha256,
        "vintage_class": "repository_state_history",
        "source_native_publication_time_proven": False,
        "updater_freshness_status": (
            updater_health["freshness_status"] if updater_health else "not_checked"
        ),
    }
    (output_dir / "audit_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        _render_report(data_path, audits, revisions, updater_health), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--data-path",
        default="dengue_updater/data/processed/monthly_data.csv",
    )
    parser.add_argument("--key-field", default="date")
    parser.add_argument("--update-log", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_vintage_audit(
        args.repo,
        args.data_path,
        args.output_dir,
        args.key_field,
        args.update_log,
        args.as_of,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
