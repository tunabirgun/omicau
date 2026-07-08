"""In-fold batch-adjustment primitives for the opt-in sensitivity probe.

The probe answers one question: *does the fusion signal survive removing batch
variance?* Per-batch location offsets are estimated on TRAIN-fold rows only and
applied to that fold's validation rows (mirroring the masked-standardization
precedent), so no batch statistic crosses the train/test boundary. It produces
no corrected dataset -- "diagnose, don't correct" holds: omicau tells you what
correction would do to your signal without handing you a corrected matrix.

Location-only centering (no outcome covariate, no scale term) is the conservative,
most-defensible probe; empirical-Bayes ComBat is a possible future upgrade that
must obey the same in-fold + confounding-gate contract.
"""

from __future__ import annotations

import numpy as np


def fit_batch_centering(X_train: np.ndarray, batch_train: np.ndarray):
    """Per-feature per-batch mean offset (vs the global mean) from TRAIN rows only.
    NaN-aware so omicau's true-missing masks are preserved."""
    global_mean = np.nanmean(X_train, axis=0)
    global_mean = np.where(np.isfinite(global_mean), global_mean, 0.0)
    offsets: dict = {}
    for b in np.unique(batch_train):
        m = batch_train == b
        bm = np.nanmean(X_train[m], axis=0)
        offsets[b] = np.where(np.isfinite(bm), bm - global_mean, 0.0)
    return global_mean, offsets


def apply_batch_centering(X: np.ndarray, batch: np.ndarray, offsets: dict) -> np.ndarray:
    """Subtract each row's batch offset; rows of an unseen batch are unchanged
    (NaN entries stay NaN: NaN - finite = NaN)."""
    Xc = np.array(X, copy=True)
    for b, off in offsets.items():
        m = batch == b
        if m.any():
            Xc[m] -= off
    return Xc


def can_correct_in_fold(splitter, X, y, groups, batch_codes, min_per_batch: int):
    """Straddle + min-count guard. Correcting a validation batch needs that batch
    present (>= min_per_batch rows) on the TRAIN side of the *same* fold. Returns
    (ok: bool, reason: str). Iterates the exact splits the probe will use."""
    split_args = (X, y, groups) if groups is not None else (X, y)
    bad_straddle: set = set()
    bad_count: set = set()
    for tr, va in splitter.split(*split_args):
        tr_batches = set(np.unique(batch_codes[tr]).tolist())
        for b in np.unique(batch_codes[va]):
            if b not in tr_batches:
                bad_straddle.add(int(b))
            elif int((batch_codes[tr] == b).sum()) < min_per_batch:
                bad_count.add(int(b))
    if bad_straddle:
        return False, (f"batch(es) {sorted(bad_straddle)} appear only in validation folds "
                       "(cannot fit a train-side offset); likely batch is nearly one-per-group.")
    if bad_count:
        return False, (f"batch(es) {sorted(bad_count)} have fewer than {min_per_batch} training "
                       "rows in some fold (offset would be noise).")
    return True, "ok"
