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


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (~np.isnan(X)).astype(np.float32)
    Xs = (np.where(np.isnan(X), 0.0, X) - mean) / std
    Xs = np.where(mask > 0, Xs, 0.0).astype(np.float32)
    return Xs, mask


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
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MaskedGlobalPoolingFusion(
        feature_dims, cfg.embed_dim, cfg.hidden_dim, out_dim, cfg.dropout, cfg.pooling
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
    for _ in range(cfg.epochs):
        model.train()
        perm = rng.permutation(fit_idx)
        for start in range(0, len(perm), batch_size):
            bidx = perm[start : start + batch_size]
            opt.zero_grad()
            out = model(_to_batch(mod_arrays, bidx, device))
            loss = loss_fn(out, _y(bidx))
            loss.backward()
            opt.step()
        # internal validation for early stopping (subset of the training fold)
        model.eval()
        with torch.no_grad():
            out_v = model(_to_batch(mod_arrays, val_idx, device))
            vloss = float(loss_fn(out_v, _y(val_idx)).item())
        if vloss < best_val - 1e-4:
            best_val, best_state, patience = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
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

    k = safe_n_splits(task, y, groups, config.cv.n_splits)
    splitter = make_cv_splitter(task, k, seed, config.cv.shuffle, groups)

    if task == "classification":
        n_classes = out_dim
        oof_score = np.full((n, n_classes), np.nan)
    oof_pred = np.full(n, np.nan)
    per_fold: list[dict[str, float]] = []
    fold_primary: list[float] = []
    imp_acc = {m: np.zeros(feature_dims[m]) for m in modalities} if compute_importance else None
    imp_folds = 0

    fold_id = 0
    for train_idx, val_idx in splitter.split(np.zeros(n), y, groups):
        # internal early-stopping split from the training fold (stratified-ish).
        rng = np.random.default_rng(seed + fold_id)
        tr = np.array(train_idx)
        rng.shuffle(tr)
        cut = max(1, int(0.15 * len(tr)))
        val_internal, fit_idx = tr[:cut], tr[cut:]
        if len(fit_idx) == 0:
            fit_idx, val_internal = tr, tr

        # masked standardization from the FIT split only.
        mod_arrays: dict[str, dict[str, np.ndarray]] = {}
        for m in modalities:
            mean, std = _masked_stats(raw[m], fit_idx)
            Xs, mask = _standardize(raw[m], mean, std)
            mod_arrays[m] = {"Xs": Xs, "mask": mask}

        model, device, _bs = _train_fold_resilient(
            mod_arrays, y, fit_idx, val_internal, feature_dims, task, out_dim, config.neural,
            seed=seed + fold_id, device=device, batch_size=config.neural.batch_size,
        )

        model.eval()
        with torch.no_grad():
            logits = model(_to_batch(mod_arrays, np.array(val_idx), device)).cpu()
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
                imp_acc[m] += model.encoders[m].feature_norms()
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
            obs_std = np.nan_to_num(np.nanstd(raw[m], axis=0), nan=0.0)
            score = (imp_acc[m] / imp_folds) * (obs_std + 1e-6)
            for j, fname in enumerate(feature_names[m]):
                importance[f"{m}::{fname}"] = float(score[j])

    return CVResult(
        name=name, task=task, metrics=metrics, per_fold=per_fold,
        fold_primary=[float(v) for v in fold_primary], feature_importance=importance,
        n_features=int(sum(feature_dims.values())), modalities=list(modalities),
        extra={"n_splits": int(k), "device": device.type},
        oof_true=y, oof_score=pooled, oof_pred=oof_pred,
        oof_groups=(np.asarray(groups) if groups is not None else None),
    )


def run_neural_benchmark(aligned, config) -> dict[str, Any]:
    """Run the masked-pooling fusion benchmark (single, fusion, leave-one-out)."""
    if not config.neural.enabled:
        return {"enabled": False, "results": []}

    device = resolve_device(config.compute.device)
    torch.manual_seed(config.seed)
    mods = aligned.modality_names
    results: list[CVResult] = []

    # single-modality
    for m in mods:
        results.append(_neural_cv(f"neural::{m}", aligned, [m], config, device, compute_importance=False))
    # full fusion (with native attribution)
    results.append(
        _neural_cv("neural::FUSION", aligned, mods, config, device,
                   compute_importance=config.xai.enabled)
    )
    # leave-one-out
    if len(mods) > 1:
        for m in mods:
            subset = [x for x in mods if x != m]
            results.append(
                _neural_cv(f"neural::FUSION-minus-{m}", aligned, subset, config, device, compute_importance=False)
            )

    attach_cis(results, n_boot=config.cv.n_bootstrap, seed=config.seed)
    return {
        "enabled": True,
        "device": device.type,
        "primary_metric": PRIMARY_METRIC[aligned.task],
        "results": results,
    }
