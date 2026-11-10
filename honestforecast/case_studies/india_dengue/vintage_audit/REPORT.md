# Dengue vintage audit

- Source path: `dengue_updater/data/processed/monthly_data.csv`
- Git-preserved snapshots: 18
- Snapshot window: 2026-03-31T14:45:03-04:00 to 2026-11-05T10:40:11Z
- Historical revision events: 1543
- Previously non-null cells removed: 0
- Vintage class: repository-state evidence, not source-native publication timestamps

## Snapshot manifest

| Commit | Committed at | Rows | Through | Revised cells | Removed cells | Class |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `86fc3d58` | 2026-03-31T14:45:03-04:00 | 63 | 2026-03 | 0 | 0 | historical_git_snapshot |
| `b04ac146` | 2026-04-04T23:29:45-04:00 | 64 | 2026-04 | 83 | 0 | historical_git_snapshot |
| `888591e6` | 2026-04-05T03:40:23Z | 64 | 2026-04 | 73 | 0 | scheduled_git_snapshot |
| `3a3d8f15` | 2026-04-13T00:15:22-04:00 | 64 | 2026-04 | 73 | 0 | historical_git_snapshot |
| `6199f784` | 2026-04-20T05:47:14Z | 64 | 2026-04 | 75 | 0 | scheduled_git_snapshot |
| `346202ab` | 2026-05-05T05:49:16Z | 65 | 2026-05 | 121 | 0 | scheduled_git_snapshot |
| `e100733a` | 2026-05-20T06:54:14Z | 65 | 2026-05 | 72 | 0 | scheduled_git_snapshot |
| `040f289f` | 2026-06-05T07:15:30Z | 66 | 2026-06 | 71 | 0 | scheduled_git_snapshot |
| `96d225d1` | 2026-06-20T07:06:10Z | 66 | 2026-06 | 77 | 0 | scheduled_git_snapshot |
| `a24e5de8` | 2026-07-05T06:48:22Z | 67 | 2026-07 | 142 | 0 | scheduled_git_snapshot |
| `256a5ce7` | 2026-07-20T06:24:20Z | 67 | 2026-07 | 80 | 0 | scheduled_git_snapshot |
| `98a48248` | 2026-08-05T05:53:16Z | 68 | 2026-08 | 156 | 0 | scheduled_git_snapshot |
| `96ac29fd` | 2026-08-20T03:53:21Z | 68 | 2026-08 | 106 | 0 | scheduled_git_snapshot |
| `cce9dde2` | 2026-09-05T07:35:27Z | 69 | 2026-09 | 112 | 0 | scheduled_git_snapshot |
| `d56d65e8` | 2026-09-20T08:29:11Z | 69 | 2026-09 | 104 | 0 | scheduled_git_snapshot |
| `e6dd8e8d` | 2026-10-05T10:16:23Z | 70 | 2026-10 | 93 | 0 | scheduled_git_snapshot |
| `e0853a21` | 2026-10-20T08:23:10Z | 70 | 2026-10 | 57 | 0 | scheduled_git_snapshot |
| `924b3128` | 2026-11-05T10:40:11Z | 71 | 2026-11 | 48 | 0 | scheduled_git_snapshot |

## Interpretation

Each Git object proves that these exact bytes existed by its commit time. It does not prove when the upstream provider first published each value. Use these snapshots for a repository-state historical pilot and revision sensitivity analysis; do not call them source-native real-time vintages.

The cell-level revision file contains 1566 change events.

## Updater health

- Last logged run: 2026-11-05T10:39:23.771288
- Freshness as of 2026-11-09: current (4 days)
- Status counts: `{"FAILED": 5, "PARTIAL": 27, "SUCCESS": 83}`
