"""
Phase 13 — Monitoring & Drift (Sections 25/26): the scheduled job
itself (.github/workflows/drift_check.yml runs this).

Reads the last DRIFT_WINDOW_SIZE check-ins (+ their already-computed
predictions) from whichever DATABASE_URL is set, compares their
feature/prediction distributions against the committed training-time
reference, and writes reports/monitoring/latest_drift_report.json.

Deliberately does NOT load the model — predicted_score is read
straight from the `predictions` table, which /checkins already wrote
at submission time. This job's only real dependencies are pandas +
sqlalchemy + psycopg2-binary, entirely independent of the
mlflow/shap/scikit-learn stack.

Scope limitation (Section 25, documented rather than glossed over):
only /checkins traffic is ever persisted — /api/v1/predict is free,
unauthenticated, and intentionally never writes to the DB (Phase 7/8
decision) — so "incoming traffic" here means check-in traffic only,
not total request volume.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.api.db import engine
from src.monitoring.drift import build_drift_report
from src.monitoring.reference import load_reference_distribution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "200"))
MIN_SAMPLE_SIZE = int(os.environ.get("DRIFT_MIN_SAMPLES", "30"))

REPORT_PATH = PROJECT_ROOT / "reports" / "monitoring" / "latest_drift_report.json"
HISTORY_PATH = PROJECT_ROOT / "reports" / "monitoring" / "drift_history.csv"

QUERY = text("""
    SELECT
        c.age AS "Age",
        c.gender AS "Gender",
        c.academic_level AS "Academic_Level",
        c.country AS "Country",
        c.avg_daily_usage_hours AS "Avg_Daily_Usage_Hours",
        c.most_used_platform AS "Most_Used_Platform",
        c.sleep_hours_per_night AS "Sleep_Hours_Per_Night",
        c.relationship_status AS "Relationship_Status",
        c.conflicts_over_social_media AS "Conflicts_Over_Social_Media",
        c.created_at,
        p.predicted_score
    FROM check_ins c
    JOIN predictions p ON p.check_in_id = c.id
    ORDER BY c.created_at DESC
    LIMIT :window_size
""")


def _fetch_window(window_size: int) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(QUERY, conn, params={"window_size": window_size})


def _write_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(markdown + "\n")


def _append_history(report: dict) -> None:
    row = pd.DataFrame([{
        "run_at": report["run_at"],
        "window_size_used": report["window_size_used"],
        "verdict": report["verdict"],
        "worst_check": report["worst_check"]["name"],
        "worst_psi": report["worst_check"]["psi"],
    }])
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)


def main() -> int:
    reference = load_reference_distribution()
    window_df = _fetch_window(WINDOW_SIZE)
    n = len(window_df)

    if n < MIN_SAMPLE_SIZE:
        report = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "window_size_used": n,
            "verdict": "insufficient_data",
            "message": (
                f"Only {n} check-in(s) available (minimum {MIN_SAMPLE_SIZE}) — "
                "skipping drift computation rather than reporting a verdict from "
                "too small a sample."
            ),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(report["message"])
        _write_step_summary(f"### Drift check\n{report['message']}")
        return 0  # not a failure — honestly can't assess yet

    report = build_drift_report(reference, window_df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    _append_history(report)

    summary = "\n".join([
        "### Drift check",
        f"- Window: last {n} check-ins",
        f"- Verdict: **{report['verdict']}**",
        f"- Worst: `{report['worst_check']['name']}` (PSI={report['worst_check']['psi']:.3f})",
    ])
    _write_step_summary(summary)
    print(summary)

    return 1 if report["verdict"] == "significant" else 0


if __name__ == "__main__":
    sys.exit(main())