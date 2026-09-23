from __future__ import annotations

import numpy as np

from omicau.interpretation.utility import _linear_cka, _prep_cka


def feature_gram_reference(first: np.ndarray, second: np.ndarray) -> float:
    a, b = _prep_cka(first), _prep_cka(second)
    numerator = np.linalg.norm(a.T @ b) ** 2
    return float(numerator / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def test_small_feature_path_preserves_previous_value() -> None:
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(60, 20)), rng.normal(size=(60, 30))
    assert _linear_cka(a, b) == feature_gram_reference(a, b)


def test_sample_gram_path_matches_feature_reference_with_missingness() -> None:
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=(24, 180)), rng.normal(size=(24, 220))
    a[0, 0] = np.nan
    b[1, 1] = np.nan
    assert np.isclose(_linear_cka(a, b), feature_gram_reference(a, b), atol=1e-12, rtol=1e-12)


def test_wide_data_never_norms_feature_by_feature_matrices(monkeypatch) -> None:
    rng = np.random.default_rng(19)
    a, b = rng.normal(size=(20, 300)), rng.normal(size=(20, 400))
    original = np.linalg.norm
    shapes = []

    def observed(value, *args, **kwargs):
        shapes.append(np.asarray(value).shape)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", observed)
    value = _linear_cka(a, b)
    assert np.isfinite(value)
    assert shapes == [(20, 20), (20, 20)]
