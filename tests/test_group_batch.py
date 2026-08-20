import json

import numpy as np
import pytest
from scipy import stats

import omicau.diagnostics.group_batch as gb


def _structure(values, batch, *, reps=99, seed=7):
    return gb.structure_batch_association(
        values,
        batch,
        standardization="global_zscore",
        representation_distance="normalized_euclidean",
        reps=reps,
        seed=seed,
        minimum_groups_per_batch=3,
    )


def _independent_pearson(batch, outcome):
    batch_levels = list(dict.fromkeys(batch))
    outcome_levels = list(dict.fromkeys(outcome))
    table = np.array(
        [
            [sum(b == row and y == column for b, y in zip(batch, outcome)) for column in outcome_levels]
            for row in batch_levels
        ],
        dtype=float,
    )
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / table.sum()
    return float(((table - expected) ** 2 / expected).sum())


def _independent_two_sample_logrank(time, event, batch):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    first = np.asarray(batch) == list(dict.fromkeys(batch))[0]
    observed = expected = variance = 0.0
    for event_time in np.unique(time[event == 1]):
        risk = time >= event_time
        deaths = (time == event_time) & (event == 1)
        n = int(risk.sum())
        d = int(deaths.sum())
        n1 = int((risk & first).sum())
        d1 = int((deaths & first).sum())
        if n <= 1:
            continue
        observed += d1
        expected += d * n1 / n
        variance += d * (n - d) * n1 * (n - n1) / (n * n * (n - 1))
    return (observed - expected) ** 2 / variance


def test_group_collapse_gives_groups_equal_weight_and_refuses_mixed_fields():
    compact = np.array([[0.0], [2.0], [10.0], [20.0], [30.0], [40.0]])
    groups = list("aabcde")
    repeated = np.repeat(compact, [9, 9, 1, 1, 1, 1], axis=0)
    repeated_groups = np.repeat(groups, [9, 9, 1, 1, 1, 1])
    expected = gb.collapse_group_representation(compact, groups, group_reducer="mean")
    observed = gb.collapse_group_representation(repeated, repeated_groups, group_reducer="mean")
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(observed[:, 0], [1, 10, 20, 30, 40])

    with pytest.raises(ValueError, match="constant within each group"):
        gb.collapse_pure_group_values([0, 1, 0, 1, 0, 1], groups, value_kind="categorical")


def test_global_standardization_is_explicit_and_fails_on_unregistered_constant_axis():
    values = np.column_stack([np.arange(5.0), np.arange(5.0) ** 2])
    z = gb.standardize_group_representation(values, standardization="global_zscore")
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(z.std(axis=0), 1.0)
    with pytest.raises(ValueError, match="positive global scale"):
        gb.standardize_group_representation(
            np.column_stack([np.arange(5.0), np.ones(5)]),
            standardization="global_median_iqr",
        )
    with pytest.raises(ValueError, match="standardization"):
        gb.standardize_group_representation(values, standardization="local")


def test_full_representation_distance_uses_later_axes_not_pc1_only():
    representation = np.column_stack(
        [np.tile(np.arange(5.0), 2), np.r_[np.zeros(5), np.ones(5) * 8]]
    )
    standardized = gb.standardize_group_representation(
        representation, standardization="global_zscore"
    )
    full = gb.full_representation_distance(
        standardized, representation_distance="normalized_euclidean"
    )
    pc1_only = np.abs(standardized[:, 0, None] - standardized[None, :, 0])
    assert pc1_only[0, 5] == 0
    assert full[0, 5] > 0


@pytest.mark.parametrize("signal", ["location", "scale", "later_axis"])
def test_registered_structure_positives_detect_location_scale_and_later_axis(signal):
    rng = np.random.default_rng(18)
    batch = np.array(["first"] * 14 + ["second"] * 14)
    values = rng.normal(size=(28, 4))
    if signal == "location":
        values[14:, :] += 4
    elif signal == "scale":
        values[:14, :] *= 0.15
        values[14:, :] *= 4
    else:
        values[:, :3] = np.tile(values[:14, :3], (2, 1))
        values[:14, 3] -= 4
        values[14:, 3] += 4
    result = _structure(values, batch, reps=199)
    assert result["statistic"] > 0
    assert result["p_value"] <= 0.05


def test_structure_null_has_identical_registered_group_distributions():
    base = np.column_stack([np.arange(7.0), np.arange(7.0) ** 2])
    values = np.repeat(base, 2, axis=0)
    batch = np.tile(["first", "second"], 7)
    result = _structure(values, batch, reps=199)
    assert result["p_value"] > 0.1


def test_nonlinear_radial_structure_positive_is_detected():
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    radius = np.r_[np.ones(15), np.full(15, 4.0)]
    values = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    batch = np.array(["inner"] * 15 + ["outer"] * 15)
    result = _structure(values, batch, reps=199, seed=9)
    assert result["p_value"] <= 0.05


def test_classification_fixed_margin_pearson_matches_independent_table_oracle():
    batch = np.array(["a"] * 8 + ["b"] * 8)
    outcome = np.array([0] * 7 + [1] + [0] + [1] * 7)
    result = gb.classification_batch_association(
        batch, outcome, reps=199, seed=3, minimum_groups_per_batch=3
    )
    assert result["statistic"] == pytest.approx(_independent_pearson(batch, outcome))
    assert result["effect"] > 0.7
    assert result["p_value"] < 0.05


def test_classification_sparse_and_perfect_tables_are_supported():
    batch = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 6)
    sparse = np.array([0] * 5 + [1] + [0] + [1] * 5 + [2] * 6)
    perfect = np.repeat([0, 1, 2], 6)
    for outcome in (sparse, perfect):
        result = gb.classification_batch_association(
            batch, outcome, reps=99, seed=4, minimum_groups_per_batch=3
        )
        assert np.isfinite(result["statistic"])
        assert 0 < result["p_value"] <= 1


def test_regression_uses_batch_delta_and_normalized_midrank_outcome_distance():
    batch = np.array(["a"] * 10 + ["b"] * 10)
    outcome = np.r_[np.arange(10), np.arange(10) + 30]
    result = gb.regression_batch_association(
        batch, outcome, reps=199, seed=5, minimum_groups_per_batch=3
    )
    ranks = stats.rankdata(outcome, method="average")
    scaled = (ranks - 1) / (len(ranks) - 1)
    independent_distance = np.abs(scaled[:, None] - scaled[None, :])
    np.testing.assert_allclose(gb.continuous_midrank_distance(outcome), independent_distance)
    assert result["p_value"] <= 0.05


def test_logrank_statistic_matches_independent_risk_set_oracle():
    batch = np.array(["a"] * 8 + ["b"] * 8)
    time = np.r_[np.arange(1, 9), np.arange(9, 17)].astype(float)
    event = np.ones(16, dtype=int)
    expected = _independent_two_sample_logrank(time, event, batch)
    assert gb.logrank_score_statistic(time, event, batch) == pytest.approx(expected)


def test_survival_event_only_positive_and_no_censoring_is_not_applicable():
    batch = np.array(["a"] * 10 + ["b"] * 10)
    time = np.r_[np.arange(1, 11), np.arange(11, 21)].astype(float)
    event = np.ones(20, dtype=int)
    result = gb.survival_batch_association(
        batch, time, event, reps=199, seed=8, minimum_groups_per_batch=3
    )
    assert result["event"]["p_value"] <= 0.05
    assert result["censoring"] == {
        "status": "not_applicable",
        "reason": "no_observed_censoring_events",
        "method_id": "k_sample_logrank_observed_censoring_process",
    }


def test_survival_censoring_process_is_separate_from_event_process():
    n = 20
    batch = np.array(["a"] * n + ["b"] * n)
    time = np.r_[
        np.r_[np.arange(1, 17), [100, 100, 110, 110]],
        np.r_[np.arange(50, 66), [100, 100, 110, 110]],
    ]
    event = np.r_[[0] * 16 + [1] * 4, [0] * 16 + [1] * 4]
    result = gb.survival_batch_association(
        batch, time, event, reps=199, seed=4, minimum_groups_per_batch=3
    )
    assert result["event"]["statistic"] == pytest.approx(0.0)
    assert result["event"]["p_value"] == pytest.approx(1.0)
    assert result["censoring"]["status"] == "tested"
    assert result["censoring"]["statistic"] > result["event"]["statistic"]
    assert "informative" not in json.dumps(result).lower()


def test_survival_refuses_no_event_or_risk_set_information():
    batch = np.array(["a"] * 5 + ["b"] * 5)
    with pytest.raises(ValueError, match="no event information"):
        gb.logrank_score_statistic(np.arange(1, 11), np.zeros(10), batch)
    with pytest.raises(ValueError, match="risk-set information"):
        gb.logrank_score_statistic(np.ones(10), np.ones(10), batch)


def test_one_primary_holm_family_cannot_be_triggered_by_secondary_evidence():
    primary = {
        "structure": {"status": "tested", "p_value": 0.04},
        "outcome": {"status": "tested", "p_value": 0.04},
        "censoring": {"status": "not_applicable", "p_value": 1e-12},
    }
    decisions = gb.apply_primary_holm(primary, alpha=0.05)
    assert decisions["structure"]["holm_p_value"] == pytest.approx(0.08)
    assert decisions["outcome"]["holm_p_value"] == pytest.approx(0.08)
    assert not decisions["structure"]["warning"]
    assert not decisions["outcome"]["warning"]
    assert not decisions["censoring"]["warning"]


@pytest.mark.parametrize(
    "hostile_status",
    [
        r"C:\private\credential.txt",
        "token=synthetic-secret-marker",
        "not_applicable\nAuthorization: Bearer synthetic-marker",
        7,
    ],
)
def test_primary_holm_rejects_unregistered_status_without_reflection(hostile_status):
    components = {
        "structure": {"status": "tested", "p_value": 0.5},
        "outcome": {"status": hostile_status, "p_value": 0.5},
    }
    with pytest.raises(ValueError, match="registered public token") as captured:
        gb.apply_primary_holm(components, alpha=0.05)
    message = str(captured.value)
    assert str(hostile_status) not in message
    assert "credential" not in message.lower()
    assert "synthetic-secret-marker" not in message.lower()


def test_public_summary_is_aggregate_seed_private_and_hostile_values_do_not_escape():
    groups = np.repeat([f"private-{i}" for i in range(10)], 2)
    batch = np.repeat([r"C:\private\batch.tsv"] * 5 + ["hostile-subject-label"] * 5, 2)
    endpoint = np.repeat([0] * 5 + [1] * 5, 2)
    representation = np.repeat(
        np.column_stack([np.arange(10.0), np.arange(10.0) ** 2]), 2, axis=0
    )
    execution_seed = 987654321
    result = gb.group_batch_diagnostics(
        representation,
        groups,
        batch,
        endpoint,
        endpoint_kind="classification",
        group_reducer="mean",
        standardization="global_zscore",
        representation_distance="normalized_euclidean",
        reps=19,
        seed=execution_seed,
        minimum_groups_per_batch=3,
        alpha=0.05,
    )
    assert set(result) == {
        "status",
        "claim_id",
        "group_count",
        "modality_count",
        "component_decisions",
        "effect_summaries",
        "eligibility_reasons",
        "method_ids",
        "multiplicity_family_id",
        "oracle_status",
        "permutation_registry_sha256",
        "permutation_registry_status",
    }
    serialized = json.dumps(result).lower()
    for forbidden in (
        "private-",
        "batch.tsv",
        "subject-label",
        "group_ids",
        "labels",
        "coordinates",
        str(execution_seed),
    ):
        assert forbidden not in serialized
    assert result["status"] == "development_only_not_production_wired"
    assert result["permutation_registry_sha256"] is None
    assert result["permutation_registry_status"] == "unavailable_pending_frozen_registry"


def test_mixed_batch_and_endpoint_groups_refuse_without_reflecting_hostile_values():
    groups = list("aabcde")
    for values in (
        [r"C:\secret\batch", "other", "x", "x", "y", "y"],
        [r"C:\secret\outcome", "other", 0, 0, 1, 1],
    ):
        with pytest.raises(ValueError, match="constant within each group") as captured:
            gb.collapse_pure_group_values(values, groups, value_kind="categorical")
        assert "secret" not in str(captured.value).lower()


def test_degenerate_distances_and_unsupported_batches_fail_closed():
    with pytest.raises(ValueError, match="positive global scale"):
        _structure(np.ones((10, 2)), np.array(["a"] * 5 + ["b"] * 5))
    with pytest.raises(ValueError, match="support is insufficient"):
        _structure(np.arange(20.0).reshape(10, 2), np.array(["a"] * 9 + ["b"]))


def test_boolean_parameters_fail_closed_and_arbitrary_registry_digest_is_not_accepted():
    batch = np.array(["a"] * 5 + ["b"] * 5)
    outcome = np.array([0] * 5 + [1] * 5)
    with pytest.raises(TypeError, match="reps"):
        gb.classification_batch_association(
            batch, outcome, reps=True, seed=1, minimum_groups_per_batch=2
        )
    with pytest.raises(TypeError, match="seed"):
        gb.classification_batch_association(
            batch, outcome, reps=1, seed=False, minimum_groups_per_batch=2
        )
    with pytest.raises(TypeError, match="unexpected keyword argument") as captured:
        gb.group_batch_diagnostics(
            np.arange(20.0).reshape(10, 2),
            np.arange(10),
            batch,
            outcome,
            endpoint_kind="classification",
            group_reducer="mean",
            standardization="global_zscore",
            representation_distance="normalized_euclidean",
            reps=9,
            seed=31,
            minimum_groups_per_batch=2,
            alpha=0.05,
            permutation_registry_sha256="ab" * 32,
        )
    assert "ab" * 32 not in str(captured.value)
