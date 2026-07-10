"""All of Us Researcher Workbench cloud connector.

This client runs ONLY inside the All of Us Researcher Workbench, the secure
analysis environment where user-authorized data lives. It reads the Workbench
environment variables (``WORKSPACE_CDR`` for the BigQuery CDR dataset,
``WORKSPACE_BUCKET`` for the workspace ``gs://`` bucket, ``GOOGLE_PROJECT``) and
accesses genomics (WGS), proteomics, and RNA-seq via those managed paths.

Security / isolation model: All of Us data cannot be exported from the
Workbench, network egress is locked down, and no participant-level data is ever
transmitted off-platform by omicau. When these environment variables are absent
(i.e. the tool is run outside the Workbench), every entry point degrades
gracefully to a clear, actionable error instead of attempting a network call.

Interface validated against the CDR v7/v8 Workbench variable conventions as of
2026-07; confirm the current CDR version inside your workspace.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

_ENV = {
    "cdr": "WORKSPACE_CDR",
    "bucket": "WORKSPACE_BUCKET",
    "project": "GOOGLE_PROJECT",
    "namespace": "WORKSPACE_NAMESPACE",
}


def workbench_status() -> dict[str, Any]:
    """Report which Workbench environment variables are present (no values leaked)."""
    present = {k: (os.environ.get(v) is not None) for k, v in _ENV.items()}
    return {"in_workbench": all(present[k] for k in ("cdr", "bucket", "project")),
            "variables_present": present}


def in_workbench() -> bool:
    return workbench_status()["in_workbench"]


def _require_workbench() -> None:
    if not in_workbench():
        raise EnvironmentError(
            "All of Us data is only accessible inside the Researcher Workbench. "
            "Required environment variables (WORKSPACE_CDR, WORKSPACE_BUCKET, "
            "GOOGLE_PROJECT) are not set, so this appears to be an off-platform "
            "session. Run omicau inside your Workbench analysis environment; data "
            "cannot be exported."
        )


def _require_gcs():
    try:
        from google.cloud import storage  # type: ignore
        return storage
    except ImportError as exc:  # pragma: no cover - optional dep
        from omicau._hints import extra_hint
        raise ImportError(
            f"All of Us GCS access needs 'google-cloud-storage' ({extra_hint('data')}). "
            "It is preinstalled in the Workbench."
        ) from exc


def _require_bigquery():
    try:
        from google.cloud import bigquery  # type: ignore
        return bigquery
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "All of Us BigQuery access needs 'google-cloud-bigquery' (pip install "
            "google-cloud-bigquery). It is preinstalled in the Workbench."
        ) from exc


def read_gcs_object(gs_uri: str, dest: str | Path) -> Path:
    """Download a ``gs://`` object to a local path inside the Workbench."""
    _require_workbench()
    storage = _require_gcs()
    if not gs_uri.startswith("gs://"):
        raise ValueError("gs_uri must start with 'gs://'.")
    bucket_name, _, blob_name = gs_uri[len("gs://"):].partition("/")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=os.environ.get(_ENV["project"]))
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    return dest


def query_cdr(sql: str) -> pd.DataFrame:
    """Run a BigQuery query against the workspace CDR dataset (``WORKSPACE_CDR``)."""
    _require_workbench()
    bigquery = _require_bigquery()
    client = bigquery.Client(project=os.environ.get(_ENV["project"]))
    # Callers reference the CDR dataset via the {CDR} placeholder or WORKSPACE_CDR.
    sql = sql.replace("{CDR}", os.environ.get(_ENV["cdr"], ""))
    return client.query(sql).result().to_dataframe()


def list_cdr_tables() -> pd.DataFrame:
    """List tables in the workspace CDR dataset (connectivity check).

    The exact genomic-manifest table name varies by CDR version, so this returns
    the dataset's table inventory; use it to locate the WGS/array manifest, then
    query it with :func:`query_cdr`.
    """
    _require_workbench()
    return query_cdr("SELECT table_id FROM `{CDR}.__TABLES__`")


def prepare(out_dir: str | Path) -> Path:  # pragma: no cover - Workbench-only
    """Placeholder assembler: All of Us datasets are workspace-specific.

    Because cohort definitions, phenotypes, and modality selections are chosen
    per study inside the Workbench, omicau does not hardcode a query here. This
    raises with guidance so the user builds their cohort with :func:`query_cdr`
    and :func:`read_gcs_object`, then points ``omicau run`` at the resulting
    matrices. Off-platform, it raises the Workbench-required error.
    """
    _require_workbench()
    raise NotImplementedError(
        "Assemble your All of Us cohort inside the Workbench using query_cdr() and "
        "read_gcs_object(), write the aligned matrices to CSV, then run "
        "'omicau run --config <your_config.json>'. Cohort/phenotype selection is "
        "study-specific and intentionally not hardcoded."
    )
