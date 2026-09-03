"""PyTorch Masked Global Pooling Fusion network and its CV benchmark.

Each modality is encoded by a learned per-feature embedding table. A sample's
feature values scale their embeddings into tokens, which are pooled *only over
observed features* using the missingness mask -- so missing entries are ignored
rather than imputed (no artificial variance is injected). The pooled per-modality
embeddings are concatenated and passed to an MLP head. The design is agnostic to
feature counts and to which features are missing per sample.

Standardization statistics are computed inside each training fold (masked, from
observed training entries only) and applied to the validation fold, keeping the
procedure leakage-safe. Training is wrapped in an out-of-memory self-repair loop
that halves the batch size, clears the device cache, and retries; if it still
fails it falls back to CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from omicau.models.base import (
    CVResult,
    PRIMARY_METRIC,
    attach_cis,
    make_cv_splitter,
    safe_n_splits,
    score_predictions,
)
from omicau.models.split_plan import ValidatedSplitPlan


_VALIDATED_SPLIT_STATUS = "validated_c06_exact_splits"
_LEGACY_SPLIT_STATUS = "legacy_nonbenchmark_dynamic_splits"


def _check_partition(
    train: np.ndarray,
    assessment: np.ndarray,
    parent: np.ndarray,
    groups: np.ndarray,
    level: str,
) -> None:
    train_set = set(np.asarray(train, dtype=np.int64).tolist())
    assessment_set = set(np.asarray(assessment, dtype=np.int64).tolist())
    parent_set = set(np.asarray(parent, dtype=np.int64).tolist())
    if not train_set or not assessment_set:
        raise RuntimeError(f"c06_{level}_partition_empty")
    if train_set & assessment_set:
        raise RuntimeError(f"c06_{level}_partition_overlap")
    if train_set | assessment_set != parent_set:
        raise RuntimeError(f"c06_{level}_partition_universe_mismatch")
    if {groups[index] for index in train_set} & {groups[index] for index in assessment_set}:
        raise RuntimeError(f"c06_{level}_group_overlap")


def _validated_partitions(
    validated_plan, n: int, groups: np.ndarray, task: str, y: np.ndarray
):
    if type(validated_plan) is not ValidatedSplitPlan:
        raise TypeError("c06_validated_split_plan_required")
    if groups is None or len(groups) != n:
        raise RuntimeError("c06_runtime_groups_missing_or_misaligned")
    validated_plan._private_validate_runtime_universe(
        groups=groups,
        task=task,
        y=y,
    )
    receipt = validated_plan.receipt()
    outer = tuple(validated_plan._private_outer_splits())
    if len(outer) != receipt["outer_fold_count"]:
        raise RuntimeError("c06_outer_fold_count_mismatch")
    universe = np.arange(n, dtype=np.int64)
    selected_inner = []
    for outer_fold, (train, assessment) in enumerate(outer):
        _check_partition(train, assessment, universe, groups, "outer")
        inner = tuple(validated_plan._private_inner_splits(outer_fold))
        if len(inner) != receipt["inner_fold_count"]:
            raise RuntimeError("c06_inner_fold_count_mismatch")
        if not inner:
            raise RuntimeError("c06_neural_inner_partition_missing")
        fit, early_stop = inner[0]
        _check_partition(fit, early_stop, train, groups, "neural_inner")
        selected_inner.append((fit, early_stop))
    return outer, tuple(selected_inner), receipt


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(preference: str = "auto") -> torch.device:
    """Select MPS > CUDA > CPU (or honor an explicit preference)."""
    pref = (preference or "auto").lower()
    if pref in {"cpu", "cuda", "mps"}:
        if pref == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return torch.device("cpu")
        if pref == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(pref)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class ModalityEncoder(nn.Module):
    """Per-feature embedding + masked global pooling for one modality."""

    def __init__(self, n_features: int, embed_dim: int, pooling: str = "mean"):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(n_features, embed_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        if pooling not in ("mean", "max"):
            raise ValueError(f"Unsupported pooling '{pooling}'; expected 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x, mask: [B, P]; tokens: [B, P, d]
        tokens = x.unsqueeze(-1) * self.embed.unsqueeze(0)
        m = mask.unsqueeze(-1)
        if self.pooling == "max":
            # true -inf sentinel so a row with every feature masked pools to -inf and
            # the guard below zero-fills it (finfo.min stays finite and never trips isinf).
            masked = tokens.masked_fill(m == 0, float("-inf"))
            pooled = masked.max(dim=1).values
            pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        else:  # masked mean
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = (tokens * m).sum(dim=1) / denom
        return self.norm(pooled + self.bias)

    def feature_norms(self) -> np.ndarray:
        return self.embed.detach().cpu().norm(dim=1).numpy()


class MaskedGlobalPoolingFusion(nn.Module):
    """Multi-modal masked-pooling fusion network."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        embed_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.modalities = list(feature_dims.keys())
        self.encoders = nn.ModuleDict(
            {name: ModalityEncoder(p, embed_dim, pooling) for name, p in feature_dims.items()}
        )
        fused_dim = embed_dim * len(feature_dims)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        embs = [self.encoders[name](x, mask) for name, (x, mask) in batch.items()]
        return self.head(torch.cat(embs, dim=1))


class NonlinearModalityEncoder(nn.Module):
    """Masked pooled encoder with a compact residual nonlinear projection."""

    def __init__(
        self,
        n_features: int,
        embed_dim: int,
        pooling: str = "mean",
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pooled = ModalityEncoder(n_features, embed_dim, pooling)
        projection_dim = hidden_dim if hidden_dim is not None else 2 * embed_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pooled(x, mask)
        return self.norm(pooled + self.projection(pooled))

    def feature_norms(self) -> np.ndarray:
        return self.pooled.feature_norms()


class GatedResidualFusion(nn.Module):
    """Availability-aware fusion with unimodal heads and a zero-initialized residual."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        embed_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
        encoder_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.modalities = list(feature_dims.keys())
        self.encoders = nn.ModuleDict(
            {
                name: NonlinearModalityEncoder(
                    n_features, embed_dim, pooling, encoder_hidden_dim, dropout
                )
                for name, n_features in feature_dims.items()
            }
        )
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
            for name in self.modalities
        })
        self.unimodal_heads = nn.ModuleDict(
            {name: nn.Linear(embed_dim, out_dim) for name in self.modalities}
        )
        residual_dim = embed_dim * (len(self.modalities) + 1)
        self.residual = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def _drop_availability(
        availability: torch.Tensor,
        probability: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if probability <= 0.0:
            return availability
        if probability >= 1.0:
            probability = 1.0
        sampled = torch.rand(
            availability.shape, generator=generator, device="cpu", dtype=torch.float32
        ).to(availability.device)
        dropped = availability & (sampled < probability)
        all_dropped = dropped.sum(dim=1) == availability.sum(dim=1)
        if torch.any(all_dropped):
            keep_index = availability[all_dropped].to(torch.int64).argmax(dim=1)
            rows = torch.nonzero(all_dropped, as_tuple=False).view(-1)
            dropped[rows, keep_index] = False
        return availability & ~dropped

    def forward_components(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor]],
        modality_dropout: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings: list[torch.Tensor] = []
        observed_fractions: list[torch.Tensor] = []
        unimodal_logits: list[torch.Tensor] = []
        for name in self.modalities:
            x, mask = batch[name]
            embedding = self.encoders[name](x, mask)
            embeddings.append(embedding)
            observed_fractions.append(mask.float().mean(dim=1, keepdim=True))
            unimodal_logits.append(self.unimodal_heads[name](embedding))

        stacked_embeddings = torch.stack(embeddings, dim=1)
        fractions = torch.cat(observed_fractions, dim=1)
        availability = fractions > 0.0
        if self.training:
            availability = self._drop_availability(availability, modality_dropout, generator)
        gate_inputs = [
            torch.cat((embeddings[index], observed_fractions[index]), dim=1)
            for index in range(len(self.modalities))
        ]
        gate_logits = torch.cat(
            [self.gates[name](gate_inputs[index]) for index, name in enumerate(self.modalities)],
            dim=1,
        )
        gate_logits = gate_logits.masked_fill(~availability, float("-inf"))
        no_available = ~availability.any(dim=1)
        if torch.any(no_available):
            gate_logits = gate_logits.clone()
            gate_logits[no_available] = 0.0
        gates = torch.softmax(gate_logits, dim=1)
        gates = torch.where(availability, gates, torch.zeros_like(gates))

        stacked_unimodal = torch.stack(unimodal_logits, dim=1)
        base_logits = (gates.unsqueeze(-1) * stacked_unimodal).sum(dim=1)
        fused = (gates.unsqueeze(-1) * stacked_embeddings).sum(dim=1)
        residual_input = torch.cat(
            (fused, (gates.unsqueeze(-1) * stacked_embeddings).flatten(start_dim=1)), dim=1
        )
        residual = self.residual(residual_input)
        final = base_logits + residual
        return final, {
            "base_logits": base_logits,
            "unimodal_logits": stacked_unimodal,
            "gates": gates,
            "availability": availability,
            "residual": residual,
        }

    def forward(self, batch: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.forward_components(batch)[0]


# --------------------------------------------------------------------------- #
# Fold-internal standardization (masked, leakage-safe)
# --------------------------------------------------------------------------- #
def _masked_stats(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr = X[train_idx]
    obs = ~np.isnan(tr)
    cnt = obs.sum(axis=0)
    filled = np.where(obs, tr, 0.0)
    mean = np.where(cnt > 0, filled.sum(axis=0) / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 0, np.where(obs, (tr - mean) ** 2, 0.0).sum(axis=0) / np.maximum(cnt, 1), 1.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    return mean, std


def _selected_columns(
    X: np.ndarray, fit_idx: np.ndarray, max_features: int | None
) -> np.ndarray:
    """Select fold-local features by descending observed variance with stable ties."""
    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1:
        if max_features is not None:
            raise ValueError("max_features_per_modality must be a positive integer or None")
    if max_features is None or X.shape[1] <= max_features:
        return np.arange(X.shape[1], dtype=np.int64)
    training = X[fit_idx]
    observed = ~np.isnan(training)
    counts = observed.sum(axis=0)
    means = np.where(
        counts > 0,
        np.where(observed, training, 0.0).sum(axis=0) / np.maximum(counts, 1),
        0.0,
    )
    centered = np.where(observed, training - means, 0.0)
    variance = np.where(
        counts > 0,
        (centered * centered).sum(axis=0) / np.maximum(counts, 1),
        0.0,
    )
    order = np.lexsort((np.arange(X.shape[1]), -variance))
    return order[:max_features].astype(np.int64, copy=False)


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (~np.isnan(X)).astype(np.float32)
    Xs = (np.where(np.isnan(X), 0.0, X) - mean) / std
    Xs = np.where(mask > 0, Xs, 0.0).astype(np.float32)
    return Xs, mask


def _prepare_modalities(
    raw: dict[str, np.ndarray],
    fit_idx: np.ndarray,
    max_features: int | None,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, int],
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    dimensions: dict[str, int] = {}
    columns: dict[str, np.ndarray] = {}
    scaling: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, matrix in raw.items():
        selected = _selected_columns(matrix, fit_idx, max_features)
        values = matrix[:, selected]
        mean, std = _masked_stats(values, fit_idx)
        Xs, mask = _standardize(values, mean, std)
        arrays[name] = {"Xs": Xs, "mask": mask}
        dimensions[name] = int(selected.size)
        columns[name] = selected
        scaling[name] = (mean, std)
    return arrays, dimensions, columns, scaling


# --------------------------------------------------------------------------- #
# Training with OOM self-repair
# --------------------------------------------------------------------------- #
def _to_batch(mod_arrays, idx, device):
    return {
        name: (
            torch.from_numpy(d["Xs"][idx]).to(device),
            torch.from_numpy(d["mask"][idx]).to(device),
        )
        for name, d in mod_arrays.items()
    }


def _build_neural_model(
    feature_dims: dict[str, int], out_dim: int, cfg
) -> tuple[nn.Module, str]:
    """Construct the requested neural architecture without changing legacy defaults."""
    architecture = getattr(cfg, "architecture", "legacy")
    if architecture == "legacy":
        return (
            MaskedGlobalPoolingFusion(
                feature_dims,
                cfg.embed_dim,
                cfg.hidden_dim,
                out_dim,
                cfg.dropout,
                cfg.pooling,
            ),
            architecture,
        )
    if architecture == "gated_residual":
        return (
            GatedResidualFusion(
                feature_dims,
                cfg.embed_dim,
                getattr(cfg, "residual_hidden_dim", cfg.hidden_dim),
                out_dim,
                getattr(cfg, "gated_dropout", cfg.dropout),
                cfg.pooling,
                getattr(cfg, "encoder_hidden_dim", None),
            ),
            architecture,
        )
    raise ValueError(
        f"Unsupported neural architecture '{architecture}'; expected 'legacy' or 'gated_residual'."
    )


def _unimodal_loss(
    logits: torch.Tensor,
    availability: torch.Tensor,
    target: torch.Tensor,
    task: str,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """Average task loss over observed unimodal heads only."""
    if not torch.any(availability):
        return logits.sum() * 0.0
    if task == "classification":
        repeated_target = target.unsqueeze(1).expand(-1, logits.shape[1])
        return loss_fn(logits[availability], repeated_target[availability])
    repeated_target = target.view(-1, 1).expand(-1, logits.shape[1])
    return loss_fn(logits.squeeze(-1)[availability], repeated_target[availability])


def _train_fold(
    mod_arrays: dict[str, dict[str, np.ndarray]],
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_dims: dict[str, int],
    task: str,
    out_dim: int,
    cfg,
    device: torch.device,
    seed: int,
    batch_size: int,
    fixed_epochs: int | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, architecture = _build_neural_model(feature_dims, out_dim, cfg)
    model = model.to(device)
    if architecture == "legacy":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if task == "classification":
        loss_fn = nn.CrossEntropyLoss()
        y_t = torch.from_numpy(y.astype(np.int64))
    else:
        loss_fn = nn.MSELoss()
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)
    y_t = y_t.to(device)

    def _y(idx):
        return y_t[torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device)]

    best_state, best_val, patience = None, float("inf"), 0
    rng = np.random.default_rng(seed)
    dropout_generator = torch.Generator(device="cpu")
    dropout_generator.manual_seed(seed)
    auxiliary_loss_weight = getattr(cfg, "auxiliary_loss_weight", 0.25)
    modality_dropout = getattr(cfg, "modality_dropout", 0.10)
    epochs = fixed_epochs if fixed_epochs is not None else cfg.epochs
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(fit_idx)
        for start in range(0, len(perm), batch_size):
            bidx = perm[start : start + batch_size]
            opt.zero_grad()
            batch = _to_batch(mod_arrays, bidx, device)
            if architecture == "legacy":
                out = model(batch)
                loss = loss_fn(out, _y(bidx))
            else:
                out, components = model.forward_components(
                    batch,
                    modality_dropout=modality_dropout,
                    generator=dropout_generator,
                )
                task_loss = loss_fn(out, _y(bidx))
                auxiliary_loss = _unimodal_loss(
                    components["unimodal_logits"],
                    components["availability"],
                    _y(bidx),
                    task,
                    loss_fn,
                )
                loss = task_loss + auxiliary_loss_weight * auxiliary_loss
            loss.backward()
            opt.step()
        if fixed_epochs is not None:
            continue
        # internal validation for early stopping (subset of the training fold)
        model.eval()
        with torch.no_grad():
            out_v = model(_to_batch(mod_arrays, val_idx, device))
            vloss = float(loss_fn(out_v, _y(val_idx)).item())
        if vloss < best_val - 1e-4:
            best_val, best_state, best_epoch, patience = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._omicau_selected_epoch = fixed_epochs if fixed_epochs is not None else best_epoch
    return model


def _train_fold_resilient(*args, device: torch.device, batch_size: int, **kwargs) -> tuple[nn.Module, torch.device, int]:
    """Train with OOM self-repair: halve batch, clear cache, retry, then CPU."""
    bs = batch_size
    dev = device
    attempts = 0
    while True:
        try:
            model = _train_fold(*args, device=dev, batch_size=max(1, bs), **kwargs)
            return model, dev, max(1, bs)
        except RuntimeError as exc:  # noqa: PERF203
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or "can't allocate" in msg or "mps backend out of memory" in msg
            attempts += 1
            _empty_cache(dev)
            if is_oom and bs > 1:
                bs = max(1, bs // 2)
                continue
            if is_oom and dev.type != "cpu":
                dev = torch.device("cpu")
                bs = batch_size
                continue
            raise


# --------------------------------------------------------------------------- #
# CV runner
# --------------------------------------------------------------------------- #
def _neural_cv(
    name: str,
    aligned,
    modalities: list[str],
    config,
    device: torch.device,
    compute_importance: bool,
    validated_plan=None,
) -> CVResult:
    task = aligned.task
    seed = config.seed
    y = aligned.y.to_numpy()
    groups = aligned.groups.to_numpy() if aligned.groups is not None else None
    n = len(y)

    raw = {m: aligned.modalities[m].X for m in modalities}
    feature_dims = {m: raw[m].shape[1] for m in modalities}
    feature_names = {m: aligned.modalities[m].feature_names for m in modalities}
    out_dim = int(len(np.unique(y))) if task == "classification" else 1

    if validated_plan is not None:
        outer_splits, inner_splits, split_receipt = _validated_partitions(
            validated_plan, n, groups, task, y
        )
        k = len(outer_splits)
        split_status = _VALIDATED_SPLIT_STATUS
    else:
        k = safe_n_splits(task, y, groups, config.cv.n_splits)
        splitter = make_cv_splitter(task, k, seed, config.cv.shuffle, groups)
        outer_splits = tuple(splitter.split(np.zeros(n), y, groups))
        inner_splits = None
        split_receipt = None
        split_status = _LEGACY_SPLIT_STATUS

    if task == "classification":
        n_classes = out_dim
        oof_score = np.full((n, n_classes), np.nan)
    oof_pred = np.full(n, np.nan)
    per_fold: list[dict[str, float]] = []
    fold_primary: list[float] = []
    imp_acc = {m: np.zeros(feature_dims[m]) for m in modalities} if compute_importance else None
    imp_folds = 0
    max_features = getattr(config.neural, "max_features_per_modality", None)

    fold_id = 0
    for train_idx, val_idx in outer_splits:
        if inner_splits is not None:
            fit_idx, val_internal = inner_splits[fold_id]
        else:
            rng = np.random.default_rng(seed + fold_id)
            tr = np.array(train_idx)
            rng.shuffle(tr)
            cut = max(1, int(0.15 * len(tr)))
            val_internal, fit_idx = tr[:cut], tr[cut:]
            if len(fit_idx) == 0:
                fit_idx, val_internal = tr, tr

        inner_arrays, inner_dims, _, _ = _prepare_modalities(
            raw, np.asarray(fit_idx), max_features
        )
        selection_model, device, _bs = _train_fold_resilient(
            inner_arrays, y, fit_idx, val_internal, inner_dims, task, out_dim, config.neural,
            seed=seed + fold_id, device=device, batch_size=config.neural.batch_size,
        )
        selected_epoch = int(getattr(selection_model, "_omicau_selected_epoch", 1))
        if selected_epoch < 1:
            raise RuntimeError("neural_selected_epoch_invalid")

        outer_arrays, outer_dims, outer_columns, _ = _prepare_modalities(
            raw, np.asarray(train_idx), max_features
        )
        model, device, _bs = _train_fold_resilient(
            outer_arrays,
            y,
            np.asarray(train_idx),
            np.asarray(train_idx),
            outer_dims,
            task,
            out_dim,
            config.neural,
            seed=seed + fold_id,
            device=device,
            batch_size=config.neural.batch_size,
            fixed_epochs=selected_epoch,
        )

        model.eval()
        with torch.no_grad():
            logits = model(_to_batch(outer_arrays, np.array(val_idx), device)).cpu()
        if task == "classification":
            proba = torch.softmax(logits, dim=1).numpy()
            oof_score[val_idx] = proba
            preds = proba.argmax(axis=1)
            oof_pred[val_idx] = preds
            fm = score_predictions(
                y[val_idx], proba[:, 1] if n_classes == 2 else proba, preds, task
            )
        else:
            preds = logits.view(-1).numpy()
            oof_pred[val_idx] = preds
            fm = score_predictions(y[val_idx], preds, preds, task)
        per_fold.append(fm)
        fold_primary.append(fm.get(PRIMARY_METRIC[task], np.nan))

        if compute_importance:
            for m in modalities:
                columns = outer_columns[m]
                observed_std = np.nan_to_num(
                    np.nanstd(raw[m][np.asarray(train_idx)][:, columns], axis=0), nan=0.0
                )
                imp_acc[m][columns] += model.encoders[m].feature_norms() * (
                    observed_std + 1e-6
                )
            imp_folds += 1
        fold_id += 1

    if task == "classification":
        pooled = oof_score[:, 1] if n_classes == 2 else oof_score
        metrics = score_predictions(y, pooled, oof_pred.astype(int), task)
    else:
        pooled = oof_pred
        metrics = score_predictions(y, oof_pred, oof_pred, task)

    importance: dict[str, float] = {}
    if compute_importance and imp_folds:
        for m in modalities:
            score = imp_acc[m] / imp_folds
            for j, fname in enumerate(feature_names[m]):
                importance[f"{m}::{fname}"] = float(score[j])

    return CVResult(
        name=name, task=task, metrics=metrics, per_fold=per_fold,
        fold_primary=[float(v) for v in fold_primary], feature_importance=importance,
        n_features=int(sum(feature_dims.values())), modalities=list(modalities),
        extra={
            "n_splits": int(k),
            "device": device.type,
            "split_plan_status": split_status,
            "split_plan_receipt": split_receipt,
        },
        oof_true=y, oof_score=pooled, oof_pred=oof_pred,
        oof_groups=(np.asarray(groups) if groups is not None else None),
    )


def run_neural_benchmark(aligned, config, validated_plan=None) -> dict[str, Any]:
    """Run the masked-pooling fusion benchmark (single, fusion, leave-one-out)."""
    if not config.neural.enabled:
        return {"enabled": False, "results": []}

    device = resolve_device(config.compute.device)
    torch.manual_seed(config.seed)
    mods = aligned.modality_names
    results: list[CVResult] = []

    # single-modality
    for m in mods:
        results.append(_neural_cv(
            f"neural::{m}", aligned, [m], config, device, compute_importance=False,
            validated_plan=validated_plan,
        ))
    # full fusion (with native attribution)
    results.append(
        _neural_cv(
            "neural::FUSION", aligned, mods, config, device,
            compute_importance=config.xai.enabled, validated_plan=validated_plan,
        )
    )
    # leave-one-out
    if len(mods) > 1:
        for m in mods:
            subset = [x for x in mods if x != m]
            results.append(
                _neural_cv(
                    f"neural::FUSION-minus-{m}", aligned, subset, config, device,
                    compute_importance=False, validated_plan=validated_plan,
                )
            )

    attach_cis(results, n_boot=config.cv.n_bootstrap, seed=config.seed)
    return {
        "enabled": True,
        "device": device.type,
        "primary_metric": PRIMARY_METRIC[aligned.task],
        "results": results,
        "split_execution_status": (
            _VALIDATED_SPLIT_STATUS if validated_plan is not None else _LEGACY_SPLIT_STATUS
        ),
        "split_plan_receipt": validated_plan.receipt() if validated_plan is not None else None,
    }
