"""Low-rank donor mixture and cross-validated few-shot ridge adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from .config import FewShotParentAdapterConfig


@dataclass(frozen=True)
class FewShotRidgeFit:
    donor_weights: np.ndarray
    adapter_weight: np.ndarray
    selected_lambda: float
    base_cv_mse: float
    adapted_cv_mse: float
    relative_cv_gain: float
    cv_eligible: bool
    num_support: int
    num_donors: int


@dataclass(frozen=True)
class FewShotCandidate:
    base_delta: np.ndarray
    delta: np.ndarray
    source_available: np.ndarray
    parent_calibration_scale: np.ndarray
    parent_calibration_lambda: np.ndarray
    fit: FewShotRidgeFit


def build_leakage_safe_equal_line_prior(
    target_delta: np.ndarray,
    train_mask: np.ndarray,
    donor_line_ids: np.ndarray,
    donor_line_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild raw equal-line priors from the artifact-approved donors only."""

    target = np.asarray(target_delta, dtype=np.float64)
    mask = np.asarray(train_mask, dtype=bool)
    donor_ids = np.asarray(donor_line_ids, dtype=np.int64)
    donor_active = np.asarray(donor_line_mask, dtype=bool)
    if target.ndim != 3 or mask.shape != target.shape[:2]:
        raise ValueError("target_delta/train_mask must have shapes [L,P,G]/[L,P]")
    if donor_ids.shape != donor_active.shape or donor_ids.shape[0] != len(target):
        raise ValueError("donor IDs/mask must share shape [L,D]")
    prior = np.zeros_like(target, dtype=np.float64)
    available = np.zeros(mask.shape, dtype=bool)
    for line in range(len(target)):
        donors = donor_ids[line, donor_active[line]]
        if not len(donors):
            continue
        donor_mask = mask[donors]
        counts = donor_mask.sum(axis=0)
        valid = counts > 0
        numerator = (
            target[donors] * donor_mask[:, :, None]
        ).sum(axis=0, dtype=np.float64)
        prior[line, valid] = numerator[valid] / counts[valid, None]
        available[line] = valid
    return prior.astype(np.float32), available


def _fit_no_intercept_parent_scale(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lambdas: tuple[float, ...],
    default_lambda: float,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical per-gene y=scale*x ridge, clipped to [0,2]."""

    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if x64.ndim != 2 or y64.shape != x64.shape or len(x64) < 1:
        raise ValueError("Parent calibration needs matching nonempty [N,G] arrays")
    candidates = np.asarray(lambdas, dtype=np.float64)
    total_x2 = np.square(x64, dtype=np.float64).sum(axis=0)
    total_xy = (x64 * y64).sum(axis=0, dtype=np.float64)
    if len(x64) < 2 or int(folds) < 2:
        selected = np.full(x64.shape[1], float(default_lambda), dtype=np.float64)
        scale = total_xy / np.maximum(total_x2 + selected, 1e-12)
        return np.clip(scale, 0.0, 2.0), selected
    errors = np.zeros((len(candidates), x64.shape[1]), dtype=np.float64)
    splitter = KFold(
        n_splits=min(int(folds), len(x64)),
        shuffle=True,
        random_state=int(seed),
    )
    for _, validation_ids in splitter.split(np.arange(len(x64))):
        x_validation = x64[validation_ids]
        y_validation = y64[validation_ids]
        validation_x2 = np.square(
            x_validation, dtype=np.float64
        ).sum(axis=0)
        validation_xy = (
            x_validation * y_validation
        ).sum(axis=0, dtype=np.float64)
        train_x2 = total_x2 - validation_x2
        train_xy = total_xy - validation_xy
        for index, ridge_lambda in enumerate(candidates):
            scale = train_xy / np.maximum(
                train_x2 + float(ridge_lambda), 1e-12
            )
            scale = np.clip(scale, 0.0, 2.0)
            errors[index] += np.square(
                y_validation - x_validation * scale[None, :],
                dtype=np.float64,
            ).sum(axis=0)
    best = np.argmin(errors, axis=0)
    selected = candidates[best]
    scale = total_xy / np.maximum(total_x2 + selected, 1e-12)
    return np.clip(scale, 0.0, 2.0), selected


def _validate_basis(basis: np.ndarray, gene_dim: int) -> np.ndarray:
    value = np.asarray(basis, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != int(gene_dim):
        raise ValueError("basis must have shape [R,G]")
    if not np.isfinite(value).all():
        raise ValueError("basis must be finite")
    gram = value @ value.T
    if not np.allclose(gram, np.eye(len(value)), atol=2e-4, rtol=2e-4):
        raise ValueError("active response basis rows must be orthonormal")
    return value


def project_simplex(value: np.ndarray) -> np.ndarray:
    """Euclidean projection of a short vector onto the probability simplex."""

    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("simplex input must be one finite non-empty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(vector) + 1, dtype=np.float64)
    positive = ordered - cumulative / indices > 0
    rho = int(np.flatnonzero(positive)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    return projected / projected.sum()


def _fit_donor_weights(
    donor_latent: np.ndarray,
    donor_mask: np.ndarray,
    target_latent: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit one three-line convex donor mixture on support conditions only."""

    samples, donors, rank = donor_latent.shape
    if donors < 1 or donors > 3:
        raise ValueError("donor_latent must contain one to three source lines")
    uniform = np.full(donors, 1.0 / donors, dtype=np.float64)
    complete = np.asarray(donor_mask, dtype=bool).all(axis=1)
    if complete.sum() < 2:
        return uniform
    design = donor_latent[complete].transpose(0, 2, 1).reshape(-1, donors)
    target = target_latent[complete].reshape(-1)
    system = design.T @ design + float(ridge) * np.eye(donors)
    right = design.T @ target + float(ridge) * uniform
    try:
        weights = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, right, rcond=None)[0]
    return project_simplex(weights)


def _mix_donors(
    donor_latent: np.ndarray,
    donor_mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(donor_mask, dtype=bool)
    effective = mask.astype(np.float64) * np.asarray(weights, dtype=np.float64)[None]
    mass = effective.sum(axis=1)
    available = mass > 0
    mixed = np.zeros((len(mask), donor_latent.shape[2]), dtype=np.float64)
    if available.any():
        mixed[available] = (
            donor_latent[available] * effective[available, :, None]
        ).sum(axis=1) / mass[available, None]
    return mixed, available


def _fit_adapter_weight(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("ridge x/y must share shape [N,R]")
    if len(x) < 1:
        raise ValueError("ridge fit needs at least one sample")
    rank = x.shape[1]
    if len(x) <= rank:
        dual = x @ x.T + float(ridge) * np.eye(len(x))
        try:
            solved = np.linalg.solve(dual, y)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(dual, y, rcond=None)[0]
        return x.T @ solved
    system = x.T @ x + float(ridge) * np.eye(rank)
    right = x.T @ y
    try:
        return np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, right, rcond=None)[0]


def _candidate_from_latent(
    *,
    base: np.ndarray,
    mixed_latent: np.ndarray,
    source_available: np.ndarray,
    adapter_weight: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    base_latent = base @ basis.T
    predicted_latent = mixed_latent @ adapter_weight
    correction = (predicted_latent - base_latent) @ basis
    candidate = base + correction
    return np.where(source_available[:, None], candidate, base)


def fit_fewshot_ridge_candidate(
    *,
    base_delta: np.ndarray,
    target_delta: np.ndarray,
    donor_delta: np.ndarray,
    donor_mask: np.ndarray,
    support_perturbations: np.ndarray,
    basis: np.ndarray,
    config: FewShotParentAdapterConfig,
) -> FewShotCandidate:
    """Fit support-only donor/ridge maps and predict every perturbation.

    Cross-validation is also support-only.  If it does not beat the analytic
    Parent, ``cv_eligible`` is false; downstream composition must then return
    the base exactly.
    """

    config.validate()
    raw_base = np.asarray(base_delta, dtype=np.float64)
    target = np.asarray(target_delta, dtype=np.float64)
    donors = np.asarray(donor_delta, dtype=np.float64)
    masks = np.asarray(donor_mask, dtype=bool)
    support = np.asarray(support_perturbations, dtype=np.int64)
    if raw_base.ndim != 2 or target.shape != raw_base.shape:
        raise ValueError("base_delta and target_delta must share [P,G]")
    if donors.ndim != 3 or donors.shape[0] != len(raw_base) or donors.shape[2] != raw_base.shape[1]:
        raise ValueError("donor_delta must have shape [P,D,G]")
    if masks.shape != donors.shape[:2]:
        raise ValueError("donor_mask must have shape [P,D]")
    if len(support) != int(config.support_size):
        raise ValueError("support IDs must match config.support_size")
    if len(np.unique(support)) != len(support):
        raise ValueError("support perturbations must be unique")
    if len(support) and (support.min() < 0 or support.max() >= len(raw_base)):
        raise ValueError("support perturbation is out of range")
    finite = (raw_base, target, donors)
    if any(not np.isfinite(value).all() for value in finite):
        raise ValueError("few-shot deltas must be finite")
    basis64 = _validate_basis(basis, raw_base.shape[1])
    donor_latent = np.einsum("pdg,rg->pdr", donors, basis64, optimize=True)
    target_latent = target @ basis64.T

    n_splits = min(int(config.cv_folds), len(support))
    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(config.seed),
    )
    candidate_error = np.zeros(len(config.ridge_lambdas), dtype=np.float64)
    base_error = 0.0
    observations = 0
    for train_position, validation_position in splitter.split(support):
        train_ids = support[train_position]
        validation_ids = support[validation_position]
        weights = _fit_donor_weights(
            donor_latent[train_ids],
            masks[train_ids],
            target_latent[train_ids],
            config.donor_ridge,
        )
        mixed_train, available_train = _mix_donors(
            donor_latent[train_ids], masks[train_ids], weights
        )
        mixed_validation, available_validation = _mix_donors(
            donor_latent[validation_ids], masks[validation_ids], weights
        )
        fit_train = available_train
        if fit_train.sum() < 1:
            candidate_error[:] = np.inf
            continue
        fold_scale, _ = _fit_no_intercept_parent_scale(
            raw_base[train_ids],
            target[train_ids],
            lambdas=tuple(config.parent_calibration_lambdas),
            default_lambda=float(config.parent_calibration_default_lambda),
            folds=int(config.cv_folds),
            seed=int(config.seed) + 7919 + int(train_position[0]),
        )
        base_fold = raw_base[validation_ids] * fold_scale[None, :]
        truth_fold = target[validation_ids]
        base_error += np.square(base_fold - truth_fold).sum(dtype=np.float64)
        observations += int(np.prod(base_fold.shape))
        for lambda_index, ridge_lambda in enumerate(config.ridge_lambdas):
            weight = _fit_adapter_weight(
                mixed_train[fit_train],
                target_latent[train_ids][fit_train],
                float(ridge_lambda),
            )
            prediction = _candidate_from_latent(
                base=base_fold,
                mixed_latent=mixed_validation,
                source_available=available_validation,
                adapter_weight=weight,
                basis=basis64,
            )
            candidate_error[lambda_index] += np.square(
                prediction - truth_fold
            ).sum(dtype=np.float64)
    base_cv_mse = base_error / max(observations, 1)
    candidate_mse = candidate_error / max(observations, 1)
    best = int(np.argmin(candidate_mse))
    adapted_cv_mse = float(candidate_mse[best])
    relative_gain = (
        max(0.0, (base_cv_mse - adapted_cv_mse) / max(base_cv_mse, 1e-12))
        if np.isfinite(adapted_cv_mse)
        else 0.0
    )
    eligible = bool(
        np.isfinite(adapted_cv_mse)
        and adapted_cv_mse + float(config.cv_tolerance) < base_cv_mse
        and relative_gain + float(config.cv_tolerance)
        >= float(config.minimum_relative_cv_gain)
    )

    final_weights = _fit_donor_weights(
        donor_latent[support],
        masks[support],
        target_latent[support],
        config.donor_ridge,
    )
    mixed_support, available_support = _mix_donors(
        donor_latent[support], masks[support], final_weights
    )
    adapter_weight = _fit_adapter_weight(
        mixed_support[available_support],
        target_latent[support][available_support],
        float(config.ridge_lambdas[best]),
    )
    mixed_all, available_all = _mix_donors(donor_latent, masks, final_weights)
    parent_scale, parent_lambda = _fit_no_intercept_parent_scale(
        raw_base[support],
        target[support],
        lambdas=tuple(config.parent_calibration_lambdas),
        default_lambda=float(config.parent_calibration_default_lambda),
        folds=int(config.cv_folds),
        seed=int(config.seed) + 15401,
    )
    base = raw_base * parent_scale[None, :]
    candidate = _candidate_from_latent(
        base=base,
        mixed_latent=mixed_all,
        source_available=available_all,
        adapter_weight=adapter_weight,
        basis=basis64,
    ).astype(np.float32)
    fit = FewShotRidgeFit(
        donor_weights=final_weights.astype(np.float32),
        adapter_weight=adapter_weight.astype(np.float32),
        selected_lambda=float(config.ridge_lambdas[best]),
        base_cv_mse=float(base_cv_mse),
        adapted_cv_mse=adapted_cv_mse,
        relative_cv_gain=float(relative_gain),
        cv_eligible=eligible,
        num_support=len(support),
        num_donors=donors.shape[1],
    )
    return FewShotCandidate(
        base_delta=base.astype(np.float32),
        delta=candidate,
        source_available=available_all,
        parent_calibration_scale=parent_scale.astype(np.float32),
        parent_calibration_lambda=parent_lambda.astype(np.float32),
        fit=fit,
    )


__all__ = [
    "FewShotCandidate",
    "FewShotRidgeFit",
    "build_leakage_safe_equal_line_prior",
    "fit_fewshot_ridge_candidate",
    "project_simplex",
]
