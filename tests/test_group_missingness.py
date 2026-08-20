import hashlib
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import omicau.diagnostics.group_missingness as gm


def test_collapse_is_equal_weight_per_group_despite_repeated_rows():
    compact = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bool)
    compact_groups = ["a", "a", "b", "c"]
    repeated = np.repeat(compact, [7, 7, 1, 1], axis=0)
    repeated_groups = np.repeat(compact_groups, [7, 7, 1, 1])

    expected = gm.collapse_group_missingness(compact, compact_groups)
    observed = gm.collapse_group_missingness(repeated, repeated_groups)

    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(observed, [[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_group_labels_must_be_complete_and_pure():
    missingness = np.zeros((6, 1), dtype=bool)
    with pytest.raises(ValueError, match="group_ids must not contain missing"):
        gm.collapse_group_missingness(missingness, ["a", "a", "b", None, "c", "c"])

    group_ids = ["a", "a", "b", "b", "c", "c"]
    with pytest.raises(ValueError, match="constant within each group"):
        gm.collapse_pure_group_labels(
            [0, 1, 0, 0, 1, 1], group_ids, name="endpoint", kind="categorical"
        )
    with pytest.raises(ValueError, match="endpoint must not contain missing"):
        gm.collapse_pure_group_labels(
            [0, 0, 1, 1, np.nan, np.nan], group_ids, name="endpoint", kind="categorical"
        )


def test_batch_labels_are_checked_with_the_same_group_purity_rule():
    group_ids = ["a", "a", "b", "b", "c", "c"]
    with pytest.raises(ValueError, match="batch must be constant"):
        gm.collapse_pure_group_labels(
            ["x", "y", "x", "x", "y", "y"], group_ids, name="batch", kind="categorical"
        )


def test_distance_constructors_are_finite_symmetric_and_hollow():
    profiles = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
    distances = [
        gm.normalized_l1_distance(profiles),
        gm.categorical_delta_distance(["a", "a", "b", "b", "c"]),
        gm.continuous_midrank_distance([30, 10, 20, 20, 40]),
    ]
    for distance in distances:
        assert np.isfinite(distance).all()
        np.testing.assert_allclose(distance, distance.T)
        np.testing.assert_array_equal(np.diag(distance), np.zeros(len(distance)))
    assert distances[0][0, 3] == 1.0
    assert distances[1][0, 1] == 0.0
    assert distances[1][0, 2] == 1.0
    assert distances[2][2, 3] == 0.0
    assert gm.categorical_delta_distance([True, True, False, False, True])[0, 2] == 1.0


def test_opposing_feature_signal_is_retained_when_total_burden_is_equal():
    profiles = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=float)
    labels = [0, 0, 0, 1, 1, 1]
    np.testing.assert_array_equal(profiles.mean(axis=1), np.full(6, 0.5))
    distance = gm.normalized_l1_distance(profiles)
    assert distance[0, 3] == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = gm.mgc_precomputed(
            distance,
            gm.categorical_delta_distance(labels),
            reps=19,
            seed=7,
        )
        repeated = gm.mgc_precomputed(
            distance,
            gm.categorical_delta_distance(labels),
            reps=19,
            seed=7,
        )
    assert repeated == result
    assert result["statistic"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(0.05)


def test_null_and_positive_secondary_categorical_fixtures():
    endpoint = [0, 0, 0, 0, 1, 1, 1, 1]
    null_profile = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=float)
    positive_profile = np.array([[0], [0], [0], [0], [1], [1], [1], [1]], dtype=float)

    null_stat, null_p = gm.secondary_feature_statistics(
        null_profile, endpoint, endpoint_kind="categorical", reps=99, seed=5
    )
    positive_stat, positive_p = gm.secondary_feature_statistics(
        positive_profile, endpoint, endpoint_kind="categorical", reps=199, seed=5
    )

    assert null_stat[0] == 0.0
    assert null_p[0] == 1.0
    assert positive_stat[0] > 0.0
    assert positive_p[0] < 0.1


def test_continuous_secondary_uses_absolute_spearman_and_is_seed_reproducible():
    profile = np.arange(8, dtype=float)[:, None] / 7
    endpoint = np.arange(8, dtype=float)
    first = gm.secondary_feature_statistics(
        profile, endpoint, endpoint_kind="continuous", reps=31, seed=13
    )
    second = gm.secondary_feature_statistics(
        profile, endpoint, endpoint_kind="continuous", reps=31, seed=13
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0][0] == pytest.approx(1.0)


def test_plus_one_rule_counts_ties_and_never_returns_zero():
    assert gm.plus_one_pvalue(4.0, [1.0, 2.0, 3.0]) == 0.25
    assert gm.plus_one_pvalue(3.0, [1.0, 3.0, 4.0]) == 0.75


def test_secondary_permutations_move_complete_group_endpoint_units(monkeypatch):
    lengths = []

    def recorded_statistic(values, endpoint):
        lengths.append(len(endpoint))
        return float(np.sum(values * np.arange(len(endpoint))))

    monkeypatch.setattr(gm, "_kruskal_statistic", recorded_statistic)
    gm.secondary_feature_statistics(
        np.eye(5, 2),
        [0, 0, 1, 1, 1],
        endpoint_kind="categorical",
        reps=4,
        seed=2,
    )
    assert lengths == [5] * (2 * (4 + 1))


def test_mgc_wrapper_fixes_precomputed_mode_and_one_worker(monkeypatch):
    captured = {}

    def fake_mgc(x, y, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(statistic=0.25, pvalue=0.4)

    monkeypatch.setattr(gm.stats, "multiscale_graphcorr", fake_mgc)
    x = gm.normalized_l1_distance(np.eye(5))
    y = gm.categorical_delta_distance([0, 0, 1, 1, 1])
    result = gm.mgc_precomputed(x, y, reps=17, seed=23)

    assert captured == {
        "compute_distance": None,
        "reps": 17,
        "workers": 1,
        "random_state": 23,
    }
    assert result == {
        "test": "multiscale_graphcorr",
        "statistic": 0.25,
        "p_value": 0.4,
        "reps": 17,
        "workers": 1,
    }


def test_holm_and_benjamini_yekutieli_adjustments_are_exact():
    p_values = [0.01, 0.04, 0.03]
    np.testing.assert_allclose(gm.holm_adjust(p_values), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(
        gm.benjamini_yekutieli_adjust(p_values),
        [0.055, 0.07333333333333333, 0.07333333333333333],
    )


def test_public_summary_has_fixed_aggregate_only_schema():
    execution_seed = 123456789
    group_ids = np.repeat(["private-a", "private-b", "private-c", "private-d", "private-e", "private-f"], 2)
    endpoint = np.repeat([0, 0, 0, 1, 1, 1], 2)
    row_missingness = np.repeat(np.array([[1, 0]] * 3 + [[0, 1]] * 3), 2, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = gm.group_missingness_diagnostics(
            row_missingness,
            group_ids,
            endpoint,
            endpoint_kind="categorical",
            reps=3,
            seed=execution_seed,
        )

    assert set(result) == {
        "status",
        "n_rows",
        "n_groups",
        "n_features",
        "permutation_reps",
        "permutation_registry_sha256",
        "permutation_registry_status",
        "aggregate_missing_fraction",
        "maximum_group_feature_missing_fraction",
        "endpoint_association",
        "batch_association",
    }
    assert set(result["endpoint_association"]) == {"kind", "primary", "secondary"}
    assert result["permutation_registry_sha256"] is None
    assert result["permutation_registry_status"] == "unavailable_pending_frozen_registry"
    serialized = repr(result).lower()
    for forbidden in ("private-a", "private-b", "group_ids", "sample_ids", "labels", "row_indices"):
        assert forbidden not in serialized
    assert str(execution_seed) not in serialized
    assert "number of replications is low" not in serialized

    def assert_no_seed_key(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                assert "seed" not in key.lower()
                assert_no_seed_key(nested)

    assert_no_seed_key(result)


def test_watched_fail_caller_supplied_registry_digest_keyword_is_rejected_without_reflection():
    hostile_digest = "ab" * 32
    with pytest.raises(TypeError, match="permutation_registry_sha256") as captured:
        gm.group_missingness_diagnostics(
            np.zeros((5, 1)),
            list("abcde"),
            [0, 0, 1, 1, 1],
            endpoint_kind="categorical",
            reps=1,
            seed=1,
            permutation_registry_sha256=hostile_digest,
        )
    assert hostile_digest not in str(captured.value)


def test_watched_fail_seed_derived_registry_digest_is_rejected_without_seed_reflection():
    execution_seed = 31
    seed_derived_digest = hashlib.sha256(str(execution_seed).encode("ascii")).hexdigest()
    with pytest.raises(TypeError, match="permutation_registry_sha256") as captured:
        gm.group_missingness_diagnostics(
            np.zeros((5, 1)),
            list("abcde"),
            [0, 0, 1, 1, 1],
            endpoint_kind="categorical",
            reps=1,
            seed=execution_seed,
            permutation_registry_sha256=seed_derived_digest,
        )
    message = str(captured.value)
    assert seed_derived_digest not in message
    assert str(execution_seed) not in message


def test_watched_fail_hostile_label_name_is_not_reflected():
    hostile = r"C:\private\subject_labels.tsv"
    with pytest.raises(ValueError) as captured:
        gm.collapse_pure_group_labels(
            [0, 0, 1, 1, 1],
            list("abcde"),
            name=hostile,
            kind="categorical",
        )
    message = str(captured.value)
    assert hostile not in message
    assert "private" not in message
    assert "subject_labels" not in message


def test_positive_batch_missingness_fixture_is_detected_at_group_level():
    group_ids = np.repeat(list("abcdef"), 2)
    endpoint = np.repeat([0, 1, 0, 1, 0, 1], 2)
    batch = np.repeat(["x", "x", "x", "y", "y", "y"], 2)
    row_missingness = np.repeat(np.array([[1, 0]] * 3 + [[0, 1]] * 3), 2, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = gm.group_missingness_diagnostics(
            row_missingness,
            group_ids,
            endpoint,
            endpoint_kind="categorical",
            batch=batch,
            reps=19,
            seed=7,
        )
    assert result["batch_association"]["primary"]["statistic"] == pytest.approx(1.0)
    assert result["batch_association"]["primary"]["p_value"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: gm.collapse_group_missingness([[0], [1], [0]], [True, False, True]), TypeError),
        (lambda: gm.collapse_group_missingness([[0], [2], [0]], ["a", "b", "c"]), ValueError),
        (lambda: gm.collapse_group_missingness([["0"], ["1"], ["0"]], ["a", "b", "c"]), TypeError),
        (lambda: gm.continuous_midrank_distance([True, False, True, False, True]), TypeError),
        (lambda: gm.holm_adjust([True, False]), TypeError),
        (lambda: gm.categorical_delta_distance([0, False, 1, True, 0]), TypeError),
    ],
)
def test_type_and_boolean_inputs_fail_closed(call, error):
    with pytest.raises(error):
        call()


def test_boolean_permutation_parameters_fail_closed():
    distance = gm.normalized_l1_distance(np.eye(5))
    with pytest.raises(TypeError, match="reps"):
        gm.mgc_precomputed(distance, distance, reps=True, seed=1)
    with pytest.raises(TypeError, match="seed"):
        gm.mgc_precomputed(distance, distance, reps=1, seed=False)


def test_degenerate_inputs_fail_closed():
    zero = np.zeros((5, 5))
    with pytest.raises(ValueError, match="degenerate"):
        gm.mgc_precomputed(zero, np.eye(5), reps=1, seed=1)
    with pytest.raises(ValueError, match="at least two"):
        gm.collapse_pure_group_labels(
            [1, 1, 1, 1, 1], list("abcde"), name="endpoint", kind="categorical"
        )


def test_watched_fail_non_symmetric_distance_is_rejected():
    corrupted = gm.normalized_l1_distance(np.eye(5))
    corrupted[0, 1] = 0.25
    with pytest.raises(ValueError, match="symmetric"):
        gm.mgc_precomputed(corrupted, gm.normalized_l1_distance(np.eye(5)), reps=1, seed=1)


def test_watched_fail_scalar_burden_substitution_erases_opposing_feature_signal():
    profiles = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=float)
    correct = gm.normalized_l1_distance(profiles)
    corrupted = np.abs(profiles.mean(axis=1)[:, None] - profiles.mean(axis=1)[None, :])
    assert np.any(correct > 0)
    assert not np.any(corrupted > 0)
