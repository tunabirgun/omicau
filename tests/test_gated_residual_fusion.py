"""Focused behavioural checks for the gated residual neural fusion model."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from omicau.config import NeuralSpec, OmicauConfig
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
