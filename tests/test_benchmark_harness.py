"""Execution-integrity tests for the benchmark harness.

These cover the two guarantees a long definitive run depends on and that no
result row can demonstrate on its own: a dataset is complete only when its
completion marker and every artifact it names verify, and a frozen split's
SHA-256 is reproducible from the fold payload rather than taken on trust.

Model fitting is deliberately absent. The machinery under test is promotion,
resume and verification; exercising it through real fits would test scikit-learn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "harness"))
sys.path.insert(0, str(REPO / "benchmarks" / "simulations"))

import run as harness  # noqa: E402
from artifacts import marker_path, sha256_file, split_digest, verify_marker  # noqa: E402
from generate import Spec  # noqa: E402


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Point every harness output at an isolated development root."""
    selected = harness.configure_output_paths("smoke", tmp_path / "artifacts")
    yield selected


@pytest.fixture(scope="module")
def spec():
    return Spec.load()


def _unit(family: str) -> harness.WorkUnit:
    return next(unit for unit in harness.protocol_work() if unit.family == family)


def _stage_rows(root, unit, spec, profile="smoke"):
    """Freeze splits and fabricate a complete, valid staged result set."""
    dataset = harness.build(unit, spec)
    ds_id = harness.unit_dataset_id(unit)
    arms = ("safe", "unsafe") if unit.family == "group_leakage" else (None,)
    for arm in arms:
        harness.freeze_splits(dataset, spec, arm)

    records = harness.task_index_records([unit], spec, profile)
    expected = harness.expected_keys_by_dataset(records)[ds_id]
    stage = harness.open_stage(profile, ds_id).create()
    n_samples = int(len(dataset["y"]))
    rows = []
    for record in records:
        arm = record["splitter_arm"]
        if record["method_id"] == "omicau_audit":
            rows.append({"run_id": f"{ds_id}|audit", "dataset_id": ds_id,
                         "method_id": "omicau_audit", "repeat": 0,
                         "split_id": record["split_id"], "splitter_arm": arm,
                         "oof_predictions_path": None, "n_samples": n_samples})
            continue
        split = harness.load_splits(ds_id, record["repeat"],
                                    None if arm == "standard" else arm)
        name = (f"{ds_id}_rep{record['repeat']}_{arm}_"
                f"{harness.safe_name(record['method_id'])}.npy")
        np.save(stage.oof_dir / name, np.linspace(0.1, 0.9, n_samples).astype(np.float32))
        rows.append({
            "run_id": f"{ds_id}|rep{record['repeat']}|{arm}|{record['method_id']}",
            "dataset_id": ds_id, "method_id": record["method_id"],
            "repeat": record["repeat"], "split_id": record["split_id"],
            "splitter_arm": arm, "split_sha256": split["sha256"],
            "oof_predictions_path": f"results/raw/oof/{name}", "n_samples": n_samples,
        })
    return stage, rows, expected


# --------------------------------------------------------------------------- #
# split digests
# --------------------------------------------------------------------------- #
def test_frozen_split_digest_is_reproducible_from_the_fold_payload(root, spec):
    unit = _unit("role_recovery")
    dataset = harness.build(unit, spec)
    written = harness.freeze_splits(dataset, spec, None)
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["sha256"] == split_digest(payload)


def test_a_tampered_split_no_longer_matches_its_stored_digest(root, spec):
    unit = _unit("role_recovery")
    dataset = harness.build(unit, spec)
    written = harness.freeze_splits(dataset, spec, None)
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    payload["folds"][0]["train"].append(payload["folds"][0]["test"].pop())
    assert split_digest(payload) != payload["sha256"]
    written[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match its own contents"):
        harness.load_splits(payload["dataset_id"], 0, None)


def test_paired_group_leakage_arms_get_distinct_splits_and_task_keys(root, spec):
    unit = _unit("group_leakage")
    dataset = harness.build(unit, spec)
    safe = json.loads(harness.freeze_splits(dataset, spec, "safe")[0].read_text(encoding="utf-8"))
    unsafe = json.loads(harness.freeze_splits(dataset, spec, "unsafe")[0].read_text(encoding="utf-8"))
    assert safe["splitter"] == "StratifiedGroupKFold"
    assert unsafe["splitter"] == "StratifiedKFold"
    assert safe["sha256"] != unsafe["sha256"]

    records = harness.task_index_records([unit], spec, "smoke")
    keys = [harness.logical_task_key(record) for record in records]
    assert len(keys) == len(set(keys)), "the two arms must not collapse onto one task key"
    assert {record["splitter_arm"] for record in records} == {"safe", "unsafe", "internal"}


# --------------------------------------------------------------------------- #
# atomic completion and resume
# --------------------------------------------------------------------------- #
def test_an_interrupted_dataset_never_looks_complete(root, spec):
    unit = _unit("role_recovery")
    ds_id = harness.unit_dataset_id(unit)
    stage, rows, expected = _stage_rows(root, unit, spec)
    index_sha = "0" * 64

    # Interruption: staging holds artifacts, nothing was promoted.
    assert verify_marker(marker_path(harness.WORK_DIR, stage.component), root,
                         index_sha, expected) == ["completion marker is absent"]
    assert not (harness.RESULT_DIR / "datasets").exists()
    assert harness.rebuild_aggregate(harness.RESULT_DIR / "smoke_rows.jsonl", [ds_id]) == 0

    harness.promote_dataset(stage, rows, expected, index_sha, "smoke")
    assert verify_marker(marker_path(harness.WORK_DIR, stage.component), root,
                         index_sha, expected) == []
    assert not stage.root.exists()
    assert harness.rebuild_aggregate(harness.RESULT_DIR / "smoke_rows.jsonl", [ds_id]) == len(rows)


def test_a_partial_dataset_is_refused_promotion(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    dropped = rows.pop()
    with pytest.raises(RuntimeError, match="neither a row nor a failure record"):
        harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")
    assert not marker_path(harness.WORK_DIR, stage.component).exists()
    assert dropped["dataset_id"]


def test_a_missing_out_of_fold_file_is_refused_promotion(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    victim = next(row for row in rows if row["oof_predictions_path"])
    (stage.oof_dir / Path(victim["oof_predictions_path"]).name).unlink()
    with pytest.raises(RuntimeError, match="staged out-of-fold file is missing"):
        harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")


def test_a_row_citing_the_wrong_split_digest_is_refused_promotion(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    next(row for row in rows if row.get("split_sha256"))["split_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="split_sha256 disagrees"):
        harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")


def test_a_marker_is_invalidated_by_a_damaged_artifact(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    component = stage.component
    harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")
    marker = marker_path(harness.WORK_DIR, component)

    damaged = next(harness.OOF_DIR.glob("*.npy"))
    np.save(damaged, np.zeros(3, dtype=np.float32))
    reasons = verify_marker(marker, root, "0" * 64, expected)
    assert any("does not match its SHA-256" in reason for reason in reasons)


def test_a_marker_is_invalidated_by_a_changed_task_index(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    component = stage.component
    harness.promote_dataset(stage, rows, expected, "a" * 64, "smoke")
    reasons = verify_marker(marker_path(harness.WORK_DIR, component), root, "b" * 64, expected)
    assert reasons == ["completion marker was written against a different task index"]


def test_a_deleted_row_shard_invalidates_the_marker(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    component = stage.component
    harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")
    (harness.RESULT_DIR / "datasets" / f"{component}.jsonl").unlink()
    reasons = verify_marker(marker_path(harness.WORK_DIR, component), root, "0" * 64, expected)
    assert any("promoted row file is missing" in reason for reason in reasons)


def test_promotion_clears_a_failure_record_left_by_an_earlier_attempt(root, spec):
    unit = _unit("role_recovery")
    ds_id = harness.unit_dataset_id(unit)
    harness.FAIL_DIR.mkdir(parents=True, exist_ok=True)
    stale = harness.FAIL_DIR / f"{ds_id}_rep0_standard_dead_attempt.json"
    stale.write_text(json.dumps({"run_id": "dead", "disposition": "numerical_failure"}),
                     encoding="utf-8")
    stage, rows, expected = _stage_rows(root, unit, spec)
    harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")
    assert not stale.exists(), "a superseded failure record must not survive a successful re-run"


def test_a_tampered_frozen_split_invalidates_the_marker(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    component = stage.component
    harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")

    split = next(harness.SPLIT_DIR.glob("*.json"))
    payload = json.loads(split.read_text(encoding="utf-8"))
    payload["folds"][0]["train"].append(payload["folds"][0]["test"].pop())
    split.write_text(json.dumps(payload), encoding="utf-8")
    reasons = verify_marker(marker_path(harness.WORK_DIR, component), root, "0" * 64, expected)
    assert any("does not match its SHA-256" in reason for reason in reasons)


def test_an_unreferenced_prediction_file_is_refused_promotion(root, spec):
    unit = _unit("role_recovery")
    stage, rows, expected = _stage_rows(root, unit, spec)
    np.save(stage.oof_dir / "orphan_from_a_dead_attempt.npy", np.zeros(4, dtype=np.float32))
    with pytest.raises(RuntimeError, match="named by no row"):
        harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")


def test_a_retained_failure_record_survives_a_failed_promotion(root, spec, monkeypatch):
    unit = _unit("role_recovery")
    ds_id = harness.unit_dataset_id(unit)
    harness.FAIL_DIR.mkdir(parents=True, exist_ok=True)
    retained = harness.FAIL_DIR / f"{ds_id}_rep0_standard_omicau_masked_fusion.json"
    retained.write_text(json.dumps({"run_id": "earlier", "disposition": "resource_exhausted"}),
                        encoding="utf-8")
    stage, rows, expected = _stage_rows(root, unit, spec)

    calls = {"n": 0}
    real_move = harness.atomic_move

    def failing_move(source, destination):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        return real_move(source, destination)

    monkeypatch.setattr(harness, "atomic_move", failing_move)
    with pytest.raises(OSError):
        harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")
    assert retained.exists(), "an earlier attempt's retained failure must survive a failed promotion"


def test_the_aggregate_ignores_a_shard_whose_marker_does_not_verify(root, spec):
    unit = _unit("role_recovery")
    ds_id = harness.unit_dataset_id(unit)
    stage, rows, expected = _stage_rows(root, unit, spec)
    component = stage.component
    harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")
    out = harness.RESULT_DIR / "smoke_rows.jsonl"
    assert harness.rebuild_aggregate(out, [ds_id], "0" * 64, {ds_id: expected}) == len(rows)

    marker_path(harness.WORK_DIR, component).unlink()
    assert harness.rebuild_aggregate(out, [ds_id], "0" * 64, {ds_id: expected}) == 0
    assert out.read_text(encoding="utf-8") == ""


def test_every_schema_keyword_in_the_record_is_enforced():
    from validate_rows import schema_keyword_gaps
    schemas = sorted((REPO / "benchmark_record" / "schemas").glob("*.json"))
    assert schemas, "the record must ship schemas"
    unenforced = {path.name: sorted(schema_keyword_gaps(json.loads(
        path.read_text(encoding="utf-8")))) for path in schemas}
    assert not any(unenforced.values()), f"schemas use unenforced keywords: {unenforced}"


def test_the_aggregate_is_rebuilt_only_from_promoted_shards(root, spec):
    unit = _unit("role_recovery")
    ds_id = harness.unit_dataset_id(unit)
    stage, rows, expected = _stage_rows(root, unit, spec)
    harness.promote_dataset(stage, rows, expected, "0" * 64, "smoke")
    out = harness.RESULT_DIR / "smoke_rows.jsonl"

    harness.rebuild_aggregate(out, [ds_id])
    first = sha256_file(out)
    assert harness.rebuild_aggregate(out, [ds_id]) == len(rows)
    assert sha256_file(out) == first, "rebuilding must be idempotent, never appending"
    observed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row["run_id"] for row in observed} == {row["run_id"] for row in rows}
