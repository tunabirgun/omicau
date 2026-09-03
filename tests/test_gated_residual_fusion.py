"""Focused behavioural checks for the gated residual neural fusion model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from omicau.config import NeuralSpec, OmicauConfig
from omicau.models import neural
from omicau.models.neural import GatedResidualFusion, MaskedGlobalPoolingFusion


FEATURE_DIMS = {"rna": 3, "protein": 2}


def _batch() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        "rna": (
            torch.tensor([[0.2, 1.0, -0.4], [0.8, -0.2, 0.3], [0.1, 0.5, 0.7]], dtype=torch.float32),
            torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        ),
        "protein": (
            torch.tensor([[1.2, -0.6], [0.4, 0.9], [0.3, 0.1]], dtype=torch.float32),
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
        ),
    }


def _model() -> GatedResidualFusion:
    torch.manual_seed(37)
    return GatedResidualFusion(FEATURE_DIMS, embed_dim=5, hidden_dim=7, out_dim=2, dropout=0.0)


def _assert_gate_invariants(gates: torch.Tensor, availability: torch.Tensor) -> None:
    available = availability.to(dtype=torch.bool)
    assert torch.isfinite(gates).all()
    assert torch.all(gates >= 0)
    assert torch.equal(gates.masked_select(~available), torch.zeros_like(gates.masked_select(~available)))
    has_available = available.any(dim=1)
    torch.testing.assert_close(
        gates[has_available].sum(dim=1),
        torch.ones_like(gates[has_available, 0]),
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.equal(gates[~has_available], torch.zeros_like(gates[~has_available]))


def test_masked_values_cannot_change_gated_output():
    model = _model().eval()
    original = _batch()
    altered = {name: (x.clone(), mask.clone()) for name, (x, mask) in original.items()}
    altered["rna"][0][0, 2] = 1e6
    altered["rna"][0][1, :] = -1e6
    altered["protein"][0][2, :] = 1e6

    with torch.no_grad():
        before = model(original)
        after = model(altered)
    torch.testing.assert_close(before, after, rtol=0.0, atol=1e-6)


def test_unavailable_modalities_have_zero_gates_and_finite_outputs():
    model = _model().eval()
    batch = _batch()
    batch["protein"] = (batch["protein"][0], torch.zeros_like(batch["protein"][1]))

    with torch.no_grad():
        final, parts = model.forward_components(batch)

    assert torch.isfinite(final).all()
    assert torch.isfinite(parts["base_logits"]).all()
    assert torch.equal(parts["availability"], torch.tensor([[True, False], [False, False], [True, False]]))
    _assert_gate_invariants(parts["gates"], parts["availability"])
    assert torch.equal(parts["gates"][:, 1], torch.zeros(3))
    assert torch.equal(parts["gates"][1], torch.zeros(2))


def test_gate_normalization_and_unavailable_contribution_fault_detection():
    model = _model().eval()
    with torch.no_grad():
        _final, parts = model.forward_components(_batch())
    _assert_gate_invariants(parts["gates"], parts["availability"])

    faulty = parts["gates"].clone()
    faulty[0, 1] = 1.0
    with pytest.raises(AssertionError):
        _assert_gate_invariants(faulty, parts["availability"])


def test_zero_initialized_residual_matches_gated_unimodal_base_exactly():
    model = _model().eval()
    with torch.no_grad():
        final, parts = model.forward_components(_batch())
    assert torch.equal(parts["residual"], torch.zeros_like(parts["residual"]))
    assert torch.equal(final, parts["base_logits"])
    assert parts["unimodal_logits"].shape == (3, 2, 2)


def test_residual_branch_receives_finite_then_nonzero_gradients_after_update():
    model = _model().train()
    batch = _batch()
    target = torch.tensor([0, 1, 0], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    optimizer.zero_grad()
    first_loss = F.cross_entropy(model(batch), target)
    first_loss.backward()
    first_grads = [p.grad for name, p in model.named_parameters() if "residual" in name and p.grad is not None]
    assert first_grads and all(torch.isfinite(grad).all() for grad in first_grads)
    optimizer.step()

    optimizer.zero_grad()
    second_loss = F.cross_entropy(model(batch), target)
    second_loss.backward()
    residual_grads = [p.grad for name, p in model.named_parameters() if "residual" in name and p.grad is not None]
    assert residual_grads and any(torch.count_nonzero(grad).item() for grad in residual_grads)
    assert all(torch.isfinite(grad).all() for grad in residual_grads)


def test_modality_dropout_is_seeded_training_only_and_preserves_one_available_modality():
    batch = _batch()
    model = _model().train()
    first_gen = torch.Generator().manual_seed(91)
    second_gen = torch.Generator().manual_seed(91)
    first_final, first = model.forward_components(batch, modality_dropout=1.0, generator=first_gen)
    second_final, second = model.forward_components(batch, modality_dropout=1.0, generator=second_gen)

    torch.testing.assert_close(first_final, second_final, rtol=0.0, atol=0.0)
    assert torch.equal(first["availability"], second["availability"])
    initially_available = torch.stack([mask.any(dim=1) for _x, mask in batch.values()], dim=1)
    assert torch.all(first["availability"][initially_available.any(dim=1)].any(dim=1))

    model.eval()
    with torch.no_grad():
        baseline_final, baseline = model.forward_components(batch)
        eval_final, eval_dropout = model.forward_components(
            batch, modality_dropout=1.0, generator=torch.Generator().manual_seed(91)
        )
    torch.testing.assert_close(baseline_final, eval_final, rtol=0.0, atol=0.0)
    assert torch.equal(baseline["availability"], eval_dropout["availability"])


def test_legacy_fusion_contract_is_unchanged():
    torch.manual_seed(17)
    legacy = MaskedGlobalPoolingFusion(FEATURE_DIMS, embed_dim=5, hidden_dim=7, out_dim=2, dropout=0.0)
    output = legacy(_batch())
    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()
    assert set(legacy.state_dict()) == {
        "encoders.rna.embed", "encoders.rna.bias", "encoders.rna.norm.weight", "encoders.rna.norm.bias",
        "encoders.protein.embed", "encoders.protein.bias", "encoders.protein.norm.weight", "encoders.protein.norm.bias",
        "head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias",
    }


def test_cpu_initialization_and_output_are_deterministic_with_legacy_config_compatibility():
    torch.manual_seed(71)
    first = GatedResidualFusion(FEATURE_DIMS, embed_dim=5, hidden_dim=7, out_dim=2, dropout=0.0).eval()
    torch.manual_seed(71)
    second = GatedResidualFusion(FEATURE_DIMS, embed_dim=5, hidden_dim=7, out_dim=2, dropout=0.0).eval()
    with torch.no_grad():
        first_output = first(_batch())
        second_output = second(_batch())
    torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)
    assert NeuralSpec().architecture == "gated_residual"
    assert OmicauConfig.from_dict({"neural": {"architecture": "legacy"}}).neural.architecture == "legacy"


def test_candidate_defaults_match_controlled_regimen():
    spec = NeuralSpec()
    assert spec.embed_dim == 8
    assert spec.batch_size == 16
    assert spec.max_features_per_modality == 256


def test_feature_cap_is_fit_only_deterministic_and_tie_stable():
    matrix = np.asarray([
        [0.0, 0.0, 0.0, 1.0, 3.0],
        [1.0, 1.0, 2.0, 1.0, 3.0],
        [2.0, 2.0, 4.0, 1.0, 3.0],
        [3.0, 3.0, 6.0, 1.0, 3.0],
        [9.0, -9.0, 0.0, 50.0, -50.0],
        [8.0, -8.0, 0.0, 60.0, -60.0],
    ], dtype=np.float32)
    fit = np.arange(4)
    first = neural._selected_columns(matrix, fit, 2)
    poisoned = matrix.copy()
    poisoned[4:] = 1e8
    second = neural._selected_columns(poisoned, fit, 2)
    assert np.array_equal(first, np.asarray([2, 0]))
    assert np.array_equal(first, second)
    arrays_a, dims_a, columns_a, scaling_a = neural._prepare_modalities({"m": matrix}, fit, 2)
    arrays_b, dims_b, columns_b, scaling_b = neural._prepare_modalities({"m": poisoned}, fit, 2)
    assert dims_a == dims_b == {"m": 2}
    assert np.array_equal(columns_a["m"], columns_b["m"])
    assert np.array_equal(scaling_a["m"][0], scaling_b["m"][0])
    assert np.array_equal(scaling_a["m"][1], scaling_b["m"][1])
    assert arrays_a["m"]["Xs"].shape[1] <= 2


def test_fixed_epoch_refit_uses_all_and_only_outer_training_rows(monkeypatch):
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(10, 4)).astype(np.float32)
    arrays, dimensions, _, _ = neural._prepare_modalities({"m": matrix}, np.arange(7), None)
    target = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    seen = []
    original = neural._to_batch

    def recording_batch(values, indices, device):
        seen.extend(np.asarray(indices).tolist())
        return original(values, indices, device)

    monkeypatch.setattr(neural, "_to_batch", recording_batch)
    cfg = SimpleNamespace(
        architecture="legacy", embed_dim=3, hidden_dim=4, dropout=0.0,
        pooling="mean", lr=1e-3, weight_decay=0.0, epochs=9, patience=2,
    )
    model = neural._train_fold(
        arrays, target, np.arange(7), np.arange(7), dimensions, "classification", 2,
        cfg, torch.device("cpu"), seed=13, batch_size=3, fixed_epochs=3,
    )
    assert model._omicau_selected_epoch == 3
    assert sorted(seen) == sorted(np.tile(np.arange(7), 3).tolist())
    assert not set(seen) & {7, 8, 9}


def test_assessment_label_poisoning_cannot_change_refit_state():
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(12, 5)).astype(np.float32)
    train = np.arange(8)
    arrays, dimensions, columns, scaling = neural._prepare_modalities({"m": matrix}, train, 3)
    poisoned_matrix = matrix.copy()
    poisoned_matrix[8:] = 1e8
    poisoned_arrays, poisoned_dimensions, poisoned_columns, poisoned_scaling = (
        neural._prepare_modalities({"m": poisoned_matrix}, train, 3)
    )
    assert dimensions == poisoned_dimensions
    assert np.array_equal(columns["m"], poisoned_columns["m"])
    assert np.array_equal(scaling["m"][0], poisoned_scaling["m"][0])
    assert np.array_equal(scaling["m"][1], poisoned_scaling["m"][1])
    target = np.asarray([0, 1] * 6, dtype=np.int64)
    poisoned = target.copy()
    poisoned[8:] = 1 - poisoned[8:]
    cfg = SimpleNamespace(
        architecture="gated_residual", embed_dim=8, hidden_dim=16, dropout=0.2,
        gated_dropout=0.1, encoder_hidden_dim=16, residual_hidden_dim=16,
        pooling="mean", lr=1e-3, weight_decay=1e-4, epochs=4, patience=2,
        auxiliary_loss_weight=0.25, modality_dropout=0.1,
    )
    selected_first = neural._train_fold(
        arrays, target, np.arange(6), np.arange(6, 8), dimensions,
        "classification", 2, cfg, torch.device("cpu"), seed=29, batch_size=4,
    )
    selected_second = neural._train_fold(
        poisoned_arrays, poisoned, np.arange(6), np.arange(6, 8), dimensions,
        "classification", 2, cfg, torch.device("cpu"), seed=29, batch_size=4,
    )
    assert selected_first._omicau_selected_epoch == selected_second._omicau_selected_epoch
    for name, value in selected_first.state_dict().items():
        assert torch.equal(value, selected_second.state_dict()[name])
    first = neural._train_fold(
        arrays, target, train, train, dimensions, "classification", 2, cfg,
        torch.device("cpu"), seed=29, batch_size=4, fixed_epochs=2,
    )
    second = neural._train_fold(
        poisoned_arrays, poisoned, train, train, dimensions, "classification", 2, cfg,
        torch.device("cpu"), seed=29, batch_size=4, fixed_epochs=2,
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_capped_refit_cv_runs_and_importance_maps_to_original_names(task):
    rng = np.random.default_rng(31)
    n = 24
    target = (
        np.asarray([0, 1] * (n // 2), dtype=np.int64)
        if task == "classification"
        else rng.normal(size=n)
    )
    matrix = rng.normal(size=(n, 6)).astype(np.float32)
    aligned = SimpleNamespace(
        task=task,
        y=pd.Series(target),
        groups=pd.Series(np.arange(n)),
        modalities={"m": SimpleNamespace(X=matrix, feature_names=[f"f{i}" for i in range(6)])},
    )
    cfg = OmicauConfig()
    cfg.cv.n_splits = 2
    cfg.neural.epochs = 1
    cfg.neural.patience = 1
    cfg.neural.max_features_per_modality = 2
    result = neural._neural_cv(
        "neural::m", aligned, ["m"], cfg, torch.device("cpu"), True
    )
    assert np.isfinite(result.primary)
    assert set(result.feature_importance) == {f"m::f{i}" for i in range(6)}


def test_explicit_legacy_architecture_runs_with_capped_refit():
    cfg = NeuralSpec(architecture="legacy", epochs=1, patience=1, max_features_per_modality=2)
    arrays, dimensions, _, _ = neural._prepare_modalities(
        {"m": np.arange(30, dtype=np.float32).reshape(10, 3)}, np.arange(8), 2
    )
    model = neural._train_fold(
        arrays, np.asarray([0, 1] * 5), np.arange(8), np.arange(8), dimensions,
        "classification", 2, cfg, torch.device("cpu"), seed=3, batch_size=4,
        fixed_epochs=1,
    )
    assert isinstance(model, MaskedGlobalPoolingFusion)
