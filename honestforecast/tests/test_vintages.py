import json
import tempfile
import unittest
from pathlib import Path

from honestforecast.vintages import (
    GitSnapshot,
    VintageError,
    audit_snapshots,
    summarize_revisions,
    summarize_update_log,
)


def snapshot(commit, content, subject="auto: update dengue data 2026-01-01"):
    return GitSnapshot(commit, "2026-01-01T00:00:00Z", subject, content.encode())


class VintageTests(unittest.TestCase):
    def test_revision_and_removed_value_are_distinguished(self):
        first = snapshot(
            "a" * 40,
            "date,x,y\n2025-01,1,10\n2025-02,2,20\n",
            subject="initial import",
        )
        second = snapshot(
            "b" * 40,
            "date,x,y\n2025-01,3,\n2025-02,2,20\n2025-03,4,30\n",
        )
        audits, revisions = audit_snapshots([first, second])
        self.assertEqual(audits[1].added_rows, 1)
        self.assertEqual(audits[1].revised_cells, 1)
        self.assertEqual(audits[1].removed_cells, 1)
        self.assertEqual(audits[1].retroactive_revised_cells, 1)
        self.assertEqual(len(revisions), 2)
        self.assertEqual(audits[0].vintage_class, "historical_git_snapshot")
        self.assertEqual(audits[1].vintage_class, "scheduled_git_snapshot")
        summary = summarize_revisions(revisions)
        self.assertEqual(summary[0]["revision_events"], 1)
        self.assertEqual(summary[0]["max_absolute_revision"], 2.0)

    def test_schema_change_is_recorded_not_counted_as_revision(self):
        audits, revisions = audit_snapshots(
            [
                snapshot("a", "date,x\n2025-01,1\n"),
                snapshot("b", "date,x,z\n2025-01,1,9\n"),
            ]
        )
        self.assertEqual(audits[1].schema_added_fields, ("z",))
        self.assertEqual(audits[1].revised_cells, 0)
        self.assertEqual(revisions, [])

    def test_update_log_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update_log.csv"
            path.write_text(
                "run_time,source_name,status\n"
                "2026-08-20T03:51:26,WHO,SUCCESS\n"
                "2026-08-20T03:51:26,wiki,PARTIAL\n"
            )
            result = summarize_update_log(path, as_of="2026-11-09")
            self.assertEqual(result["freshness_status"], "stale")
            self.assertEqual(result["unique_runs"], 1)
            self.assertEqual(result["source_status_counts"]["wiki"]["PARTIAL"], 1)


if __name__ == "__main__":
    unittest.main()
