"""
Deterministic data ingestion for WellPulse (Section 7 / Phase 1).

Pulls the raw "Students' Social Media Addiction" CSV (Kaggle,
adilshamim8/social-media-addiction-vs-relationships) into
data/raw/ under a fixed canonical filename, validates it against the
Phase 1 data contract (src/data/schema.py), and stamps it with a
`dataset_version` (a short content hash + ingestion month) so every
MLflow run can later be traced back to the exact file that produced it.

No manual steps once the CSV is on disk somewhere:

    python -m src.data.ingest --source ~/Downloads/students.csv
    python -m src.data.ingest                # re-stamp/re-validate the
                                              # file already at data/raw/
Kaggle requires an authenticated download, so this script does not
reach out to Kaggle itself — point --source at wherever the CSV was
downloaded (Kaggle CLI, manual browser download, etc.). This keeps the
ingestion step honest about needing that one manual download, while
making everything after it (copy, hash, validate) scripted and
deterministic per Section 7.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pandera.errors

from src.data.schema import validate_raw_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CANONICAL_FILENAME = "students_social_media_addiction.csv"
CANONICAL_PATH = RAW_DIR / CANONICAL_FILENAME
VERSION_LOG = RAW_DIR / "dataset_version.txt"


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_version(path: Path) -> str:
    """e.g. 'a1b2c3d4_2026-09' — content hash + ingestion month."""
    return f"{_sha256_of(path)[:8]}_{date.today():%Y-%m}"


def ingest(source: Path, dest: Path = CANONICAL_PATH, validate: bool = True) -> str:
    """
    Copy `source` to `dest` (unless they're already the same file),
    validate it against the data contract, and write a dataset_version
    stamp. Returns the dataset_version string.

    Raises FileNotFoundError if source doesn't exist, and
    pandera.errors.SchemaErrors if the file fails validation — in
    both cases nothing is left half-written: the version log is only
    updated after validation succeeds.
    """
    source = source.resolve()
    dest = dest.resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Source file not found: {source}\n"
            "Download the 'Students' Social Media Addiction' dataset from "
            "Kaggle (adilshamim8/social-media-addiction-vs-relationships) "
            "and pass its path with --source."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    if source != dest:
        shutil.copyfile(source, dest)
        print(f"Copied {source} -> {dest}")
    else:
        print(f"Source already at canonical path: {dest}")

    if validate:
        df = pd.read_csv(dest)
        try:
            validate_raw_dataset(df)
        except pandera.errors.SchemaErrors as e:
            print("VALIDATION FAILED — see failure cases below:", file=sys.stderr)
            print(e.failure_cases.to_string(), file=sys.stderr)
            raise
        print(f"Validated: {len(df)} rows, {len(df.columns)} columns, schema OK.")

    version = compute_dataset_version(dest)
    VERSION_LOG.write_text(
        f"dataset_version={version}\n"
        f"source_file={dest.name}\n"
        f"sha256={_sha256_of(dest)}\n"
        f"ingested_at={datetime.now(timezone.utc).isoformat()}\n"
    )
    print(f"dataset_version = {version}")
    print(f"Version stamp written to {VERSION_LOG}")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=CANONICAL_PATH,
        help="Path to the downloaded raw CSV (defaults to the canonical "
             "data/raw/ path, for re-stamping a file already in place).",
    )
    parser.add_argument(
        "--dest", type=Path, default=CANONICAL_PATH,
        help="Destination path under data/raw/.",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Copy and stamp without running the schema validation "
             "(not recommended — mainly for debugging).",
    )
    args = parser.parse_args()

    try:
        ingest(args.source, args.dest, validate=not args.skip_validation)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except pandera.errors.SchemaErrors:
        sys.exit(1)


if __name__ == "__main__":
    main()
