from __future__ import annotations

import pandas as pd
import pytest

from omicau.config import OmicauConfig
from omicau.data.alignment import align_modalities
from omicau.data.benchmark_data import make_mock_dataset, mock_config


def _two_modality_inputs():
    bundle = make_mock_dataset(task="classification", n_samples=48, seed=17)
    modalities = {name: bundle.modalities[name] for name in ("signal", "noise")}
    config = mock_config(task="classification")
    config.modalities = [spec for spec in config.modalities if spec.name in modalities]
    return bundle, modalities, config


def test_config_round_trip_preserves_batch_mapping():
    mapping = {"signal": "site_signal", "noise": "site_noise"}
    config = OmicauConfig.from_dict({"clinical": {"batch_by_modality": mapping}})
    assert config.clinical.batch_by_modality == mapping
    assert config.to_dict()["clinical"]["batch_by_modality"] == mapping


def test_modality_specific_batches_are_aligned_and_preserve_missing_labels():
    bundle, modalities, config = _two_modality_inputs()
    clinical = bundle.clinical.copy()
    clinical["site_signal"] = ["S1", "S2"] * (len(clinical) // 2)
    clinical["site_noise"] = ["N1", pd.NA, "N2"] * (len(clinical) // 3)
    dropped_sample = clinical.loc[clinical.index[0], "sample_id"]
    clinical.loc[clinical.index[0], "label"] = pd.NA
    config.clinical.batch_by_modality = {
        "signal": "site_signal",
        "noise": "site_noise",
    }

    aligned = align_modalities(modalities, clinical, config)

    assert set(aligned.batch_by_modality) == set(aligned.modalities)
    assert dropped_sample not in aligned.sample_ids
    assert list(aligned.batch_by_modality["signal"].index) == aligned.sample_ids
    expected = clinical.set_index("sample_id").loc[aligned.sample_ids, "site_noise"]
    assert aligned.batch_by_modality["noise"].isna().equals(expected.isna())
    assert "NA" not in aligned.batch_by_modality["noise"].dropna().tolist()


def test_shared_batch_is_expanded_without_changing_legacy_batch_behavior():
    bundle, modalities, config = _two_modality_inputs()
    clinical = bundle.clinical.copy()
    clinical.loc[clinical.index[0], "batch"] = pd.NA
    missing_sample = clinical.loc[clinical.index[0], "sample_id"]

    aligned = align_modalities(modalities, clinical, config)

    assert set(aligned.batch_by_modality) == set(aligned.modalities)
    assert aligned.batch.loc[missing_sample] == "NA"
    assert all(values.isna().loc[missing_sample] for values in aligned.batch_by_modality.values())


@pytest.mark.parametrize(
    "mapping",
    [
        {"signal": "batch"},
        {"signal": "batch", "noise": "batch", "extra": "batch"},
        {"signal": "batch", "noise": 3},
    ],
)
def test_explicit_mapping_requires_exact_modality_coverage(mapping):
    bundle, modalities, config = _two_modality_inputs()
    config.clinical.batch_by_modality = mapping

    with pytest.raises(
        ValueError,
        match="clinical.batch_by_modality must cover every aligned modality exactly",
    ):
        align_modalities(modalities, bundle.clinical, config)


def test_explicit_mapping_rejects_missing_column_without_reflection():
    bundle, modalities, config = _two_modality_inputs()
    marker = "C:/private/credential-marker"
    config.clinical.batch_by_modality = {"signal": "batch", "noise": marker}

    with pytest.raises(ValueError) as caught:
        align_modalities(modalities, bundle.clinical, config)

    assert str(caught.value) == "clinical.batch_by_modality refers to an unavailable clinical column"
    assert marker not in str(caught.value)


def test_batch_mapping_rejects_non_mapping_value():
    bundle, modalities, config = _two_modality_inputs()
    config.clinical.batch_by_modality = ["batch"]

    with pytest.raises(
        ValueError,
        match="clinical.batch_by_modality must be an exact modality-to-column mapping",
    ):
        align_modalities(modalities, bundle.clinical, config)
