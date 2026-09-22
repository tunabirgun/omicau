"""Development checks for the focused synthetic data-generating mechanisms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omicau.data.benchmark_data import (
    make_mock_dataset,
    validate_global_null_observed_masks,
    write_mock_dataset,
)


def test_scenario_generation_is_deterministic_for_a_development_seed():
    settings = dict(
        task="regression", n_samples=120, seed=17, complementary_strength=0.6,
        batch_outcome_confounding=0.4, group_dependence=0.5, group_mode="repeated",
    )
    first = make_mock_dataset(**settings)
    second = make_mock_dataset(**settings)

    assert first.truth == second.truth
    assert first.oracle_truth["target_coefficients"] == second.oracle_truth["target_coefficients"]
    pd.testing.assert_frame_equal(first.clinical, second.clinical)
    for name in first.modalities:
        pd.testing.assert_frame_equal(first.modalities[name], second.modalities[name])


def test_null_scenarios_define_their_independence_in_oracle_truth():
    global_null = make_mock_dataset(
        task="regression", scenario="global_null", n_samples=400, seed=3,
        include_complementary=True,
    )
    conditional = make_mock_dataset(
        task="regression", scenario="conditional_null", n_samples=400, seed=3,
    )

    assert global_null.oracle_truth["target_coefficients"] == {
        "shared": 0.0, "complementary": 0.0, "batch": 0.0,
    }
    assert global_null.oracle_truth["missingness"] == {
        "mcar_probability": 0.04, "mnar_probability": 0.0,
        "conditional_on_target": False,
    }
    assert conditional.oracle_truth["target_coefficients"]["shared"] > 0
    assert conditional.oracle_truth["target_coefficients"]["complementary"] == 0.0
    latent = conditional.oracle_truth["latent_factors"]
    assert abs(np.corrcoef(latent["shared"], latent["complementary"])[0, 1]) < 0.2


def test_global_null_mask_gate_checks_observed_matrices_and_rejects_contamination():
    bundle = make_mock_dataset(
        task="classification", scenario="global_null", n_samples=800, seed=31,
        group_mode="independent", feature_naming="generic",
    )
    validate_global_null_observed_masks(bundle)

    positive = bundle.clinical["label"] == "responder"
    contaminated = bundle.modalities["layer_a"].copy()
    contaminated.iloc[positive.to_numpy(), 0] = np.nan
    bundle.modalities["layer_a"] = contaminated
    with pytest.raises(ValueError, match="Observed masks differ by class"):
        validate_global_null_observed_masks(bundle)


def test_multidimensional_truth_and_matrices_are_finite_except_missing_values():
    bundle = make_mock_dataset(
        task="classification", n_samples=100, seed=5, complementary_strength=0.5,
        batch_outcome_confounding=0.5,
    )
    dimensions = bundle.oracle_truth["modality_dimensions"]

    assert dimensions["complementary"]["incremental_target_effect"] is True
    assert dimensions["confounded"]["batch_structured"] is True
    assert dimensions["confounded"]["target_linked"] is True
    assert dimensions["redundant"]["shared_information_factor"] is True
    assert dimensions["redundant"]["incremental_target_effect"] is None
    for name, frame in bundle.modalities.items():
        assert frame.shape[0] == len(bundle.clinical)
        assert set(frame.columns) == set(bundle.oracle_truth["feature_sets"][name])
        assert np.isfinite(frame.to_numpy(dtype=float)[~frame.isna().to_numpy()]).all()


def test_binary_development_sizes_are_feasible_for_five_folds():
    for seed in range(8):
        bundle = make_mock_dataset(task="classification", n_samples=60, seed=seed)
        counts = bundle.clinical["label"].value_counts()
        assert counts.min() >= 5


def test_regression_target_association_increases_with_declared_effects():
    shared_correlations = []
    complementary_correlations = []
    for strength in (0.25, 0.75, 1.5):
        shared = make_mock_dataset(
            task="regression", n_samples=400, seed=11, signal_strength=strength,
            include_complementary=True, complementary_strength=0.0,
        )
        complementary = make_mock_dataset(
            task="regression", n_samples=400, seed=11, signal_strength=1.0,
            include_complementary=True, complementary_strength=strength,
        )
        shared_correlations.append(abs(np.corrcoef(
            shared.clinical["label"], shared.oracle_truth["latent_factors"]["shared"]
        )[0, 1]))
        complementary_correlations.append(abs(np.corrcoef(
            complementary.clinical["label"],
            complementary.oracle_truth["latent_factors"]["complementary"],
        )[0, 1]))

    assert shared_correlations == sorted(shared_correlations)
    assert complementary_correlations == sorted(complementary_correlations)


def test_group_dependence_is_shared_within_repeated_patients():
    bundle = make_mock_dataset(
        task="regression", n_samples=120, seed=9, group_dependence=0.8,
        group_mode="repeated",
    )
    group_component = bundle.oracle_truth["latent_factors"]["group_component"]
    groups = bundle.clinical.groupby("patient_id", sort=False).indices

    assert bundle.oracle_truth["groups"]["dependence"] == 0.8
    assert any(len(indices) > 1 for indices in groups.values())
    for indices in groups.values():
        assert np.unique(group_component[np.asarray(indices)]).size == 1
    assert np.std(group_component) > 0


def test_oracle_truth_is_not_written_as_a_model_input(tmp_path):
    dataset_dir = write_mock_dataset(
        tmp_path / "dataset", task="regression", seed=19,
        complementary_strength=0.5,
    )
    written = {path.name for path in dataset_dir.iterdir()}
    config = (dataset_dir / "config.json").read_text(encoding="utf-8")

    assert not any("oracle" in name.lower() or "truth" in name.lower() for name in written)
    assert "oracle_truth" not in config
    assert "complementary" in config


def test_generic_study_mode_uses_independent_units_and_nonsemantic_input_names(tmp_path):
    bundle = make_mock_dataset(
        task="regression", n_samples=80, seed=23, complementary_strength=0.5,
        group_mode="independent", feature_naming="generic",
    )
    dataset_dir = write_mock_dataset(
        tmp_path / "generic", task="regression", seed=23, complementary_strength=0.5,
        group_mode="independent", feature_naming="generic",
    )
    config = (dataset_dir / "config.json").read_text(encoding="utf-8")
    from omicau.config import OmicauConfig
    from omicau.data.alignment import load_and_align

    aligned = load_and_align(OmicauConfig.from_file(dataset_dir / "config.json"))

    assert bundle.clinical["patient_id"].is_unique
    assert set(bundle.modalities) == {"layer_a", "layer_b", "layer_c", "layer_d", "layer_e"}
    assert set(aligned.modalities) == set(bundle.modalities)
    assert "signal" not in config and "redundant" not in config and "confounded" not in config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_samples": 1}, "at least 2"),
        ({"scenario": "unknown"}, "Unknown scenario"),
        ({"complementary_strength": -0.1}, "non-negative"),
        ({"scenario": "conditional_null", "include_complementary": False}, "requires"),
        ({"scenario": "global_null", "batch_outcome_confounding": 0.2}, "fixes"),
        ({"scenario": "global_null", "missing_mnar": 0.1}, "target-independent"),
        ({"group_dependence": 0.2}, "requires group_mode"),
    ],
)
def test_malformed_scenario_settings_fail_loudly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_mock_dataset(**kwargs)
