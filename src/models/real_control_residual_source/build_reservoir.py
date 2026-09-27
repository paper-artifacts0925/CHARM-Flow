"""Build a safe real-control reservoir aligned to a fixed Child bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .reservoir import REAL_CONTROL_RESERVOIR_SCHEMA


def validate_control_assignment_partition(
    *,
    all_control_obs_indices: np.ndarray,
    retained_control_obs_indices: np.ndarray,
    rejected_control_obs_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a retained/rejected partition of stable control row IDs.

    Older context artifacts do not carry an explicit rejected vector. For
    those artifacts the historical exact-all-controls requirement remains in
    force. New filtered artifacts must persist both sides of the partition so
    a missing, duplicated, overlapping, or non-control row fails closed.
    """

    def _ids(value, name: str, *, allow_empty: bool) -> np.ndarray:
        result = np.asarray(value, dtype=np.int64)
        if result.ndim != 1 or (not allow_empty and len(result) < 1):
            qualifier = "a vector" if allow_empty else "a non-empty vector"
            raise ValueError(f"{name} must be {qualifier}")
        if (result < 0).any():
            raise ValueError(f"{name} must contain only non-negative IDs")
        if len(np.unique(result)) != len(result):
            raise ValueError(f"{name} must contain unique IDs")
        return result

    all_ids = _ids(
        all_control_obs_indices, "all_control_obs_indices", allow_empty=False
    )
    retained = _ids(
        retained_control_obs_indices,
        "retained_control_obs_indices",
        allow_empty=False,
    )
    if rejected_control_obs_indices is None:
        if not np.array_equal(np.sort(retained), np.sort(all_ids)):
            raise ValueError(
                "legacy control assignments must contain exactly all controls; "
                "filtered assignments require rejected_control_obs_indices"
            )
        return retained, np.empty((0,), dtype=np.int64)

    rejected = _ids(
        rejected_control_obs_indices,
        "rejected_control_obs_indices",
        allow_empty=True,
    )
    overlap = np.intersect1d(retained, rejected, assume_unique=True)
    if len(overlap):
        raise ValueError(
            "retained and rejected control assignments must be disjoint; "
            f"overlap={overlap[:8].tolist()}"
        )
    partition = np.concatenate([retained, rejected])
    if not np.array_equal(np.sort(partition), np.sort(all_ids)):
        missing = np.setdiff1d(all_ids, partition, assume_unique=True)
        extra = np.setdiff1d(partition, all_ids, assume_unique=True)
        raise ValueError(
            "retained plus rejected controls must equal all controls exactly; "
            f"missing={missing[:8].tolist()}, extra={extra[:8].tolist()}"
        )
    return retained, rejected


def normalized_control_means(
    *,
    control_expression: np.ndarray,
    control_line_ids: np.ndarray,
    cellline_names: Sequence[str],
    normalization_divisor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized means/counts from the complete control population."""

    expression = np.asarray(control_expression, dtype=np.float32)
    line_ids = np.asarray(control_line_ids, dtype=np.int64)
    names = [str(name) for name in cellline_names]
    divisor = float(normalization_divisor)
    if expression.ndim != 2 or expression.shape[0] < 1:
        raise ValueError("control_expression must have shape [N,G], N > 0")
    if line_ids.shape != (len(expression),):
        raise ValueError("control_line_ids must have shape [N]")
    if not np.isfinite(expression).all():
        raise ValueError("control expression must be finite")
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("normalization_divisor must be finite and positive")
    if len(names) < 1 or len(names) != len(set(names)):
        raise ValueError("cellline_names must be non-empty and unique")
    if ((line_ids < 0) | (line_ids >= len(names))).any():
        raise ValueError("control_line_ids contains an out-of-range line")
    means = np.zeros((len(names), expression.shape[1]), dtype=np.float32)
    counts = np.zeros((len(names),), dtype=np.int64)
    normalized = expression / np.float32(divisor)
    for line, name in enumerate(names):
        rows = line_ids == line
        if not rows.any():
            raise ValueError(f"cell line {name!r} has no controls")
        means[line] = normalized[rows].mean(axis=0, dtype=np.float64)
        counts[line] = int(rows.sum())
    return means, counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fixed_unicode(values) -> np.ndarray:
    flat = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    width = max((len(value) for value in flat), default=1)
    return np.asarray(flat, dtype=f"<U{width}").reshape(np.asarray(values).shape)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def pack_assigned_controls(
    *,
    control_expression: np.ndarray,
    control_line_ids: np.ndarray,
    control_child_ids: np.ndarray,
    cellline_names: Sequence[str],
    child_prototypes: np.ndarray,
    child_mask: np.ndarray | None = None,
    normalization_divisor: float = 10.0,
    max_members_per_child: int = 128,
    seed: int = 42,
    source_obs_indices: np.ndarray | None = None,
    control_mean_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pack already assigned *control-only* rows into a fixed safe tensor.

    ``child_prototypes`` must already be in the runtime Child order. Input
    expression and prototypes are on the pre-divisor scale. When supplied,
    ``control_mean_override`` is already on the normalized scale and may come
    from the complete control population while ``control_expression`` contains
    only retained Child members. The output stores normalized residuals and
    prototypes.
    """

    expression = np.asarray(control_expression, dtype=np.float32)
    line_ids = np.asarray(control_line_ids, dtype=np.int64)
    child_ids = np.asarray(control_child_ids, dtype=np.int64)
    prototypes = np.asarray(child_prototypes, dtype=np.float32)
    names = [str(name) for name in cellline_names]
    divisor = float(normalization_divisor)
    capacity = int(max_members_per_child)
    if expression.ndim != 2 or expression.shape[0] < 1:
        raise ValueError("control_expression must have shape [N,G], N > 0")
    if line_ids.shape != (len(expression),) or child_ids.shape != (len(expression),):
        raise ValueError("control line/Child IDs must have shape [N]")
    if not np.isfinite(expression).all():
        raise ValueError("control expression must be finite")
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("normalization_divisor must be finite and positive")
    if capacity < 1:
        raise ValueError("max_members_per_child must be positive")
    if len(names) < 1 or len(names) != len(set(names)):
        raise ValueError("cellline_names must be non-empty and unique")
    if prototypes.ndim != 3 or prototypes.shape[0] != len(names):
        raise ValueError("child_prototypes must have shape [L,K,G]")
    lines, children, genes = prototypes.shape
    if genes != expression.shape[1] or children < 1:
        raise ValueError("prototype and expression dimensions disagree")
    if child_mask is None:
        active = np.ones((lines, children), dtype=bool)
    else:
        active = np.asarray(child_mask, dtype=bool)
        if active.shape != (lines, children):
            raise ValueError("child_mask must have shape [L,K]")
    if ((line_ids < 0) | (line_ids >= lines)).any():
        raise ValueError("control_line_ids contains an out-of-range line")
    if ((child_ids < 0) | (child_ids >= children)).any():
        raise ValueError("control_child_ids contains an out-of-range Child")
    if not active[line_ids, child_ids].all():
        raise ValueError("a control row is assigned to an inactive Child")
    if source_obs_indices is None:
        obs_indices = np.arange(len(expression), dtype=np.int64)
    else:
        obs_indices = np.asarray(source_obs_indices, dtype=np.int64)
        if obs_indices.shape != (len(expression),):
            raise ValueError("source_obs_indices must have shape [N]")
        if (obs_indices < 0).any() or len(np.unique(obs_indices)) != len(obs_indices):
            raise ValueError("source_obs_indices must be unique non-negative rows")

    normalized = expression / np.float32(divisor)
    control_mean = np.zeros((lines, genes), dtype=np.float32)
    source_counts = np.zeros((lines,), dtype=np.int64)
    reference_mean = None
    if control_mean_override is not None:
        reference_mean = np.asarray(control_mean_override, dtype=np.float32)
        if reference_mean.shape != (lines, genes):
            raise ValueError("control_mean_override must have shape [L,G]")
        if not np.isfinite(reference_mean).all():
            raise ValueError("control_mean_override must be finite")
    for line in range(lines):
        rows = line_ids == line
        if not rows.any():
            raise ValueError(f"cell line {names[line]!r} has no controls")
        if reference_mean is None:
            control_mean[line] = normalized[rows].mean(axis=0, dtype=np.float64)
        else:
            control_mean[line] = reference_mean[line]
        source_counts[line] = int(rows.sum())

    residuals = np.zeros(
        (lines, children, capacity, genes), dtype=np.float32
    )
    member_mask = np.zeros((lines, children, capacity), dtype=bool)
    member_obs = np.full((lines, children, capacity), -1, dtype=np.int64)
    full_counts = np.zeros((lines, children), dtype=np.int64)
    stored_counts = np.zeros((lines, children), dtype=np.int32)
    for line in range(lines):
        for child in range(children):
            rows = np.flatnonzero((line_ids == line) & (child_ids == child))
            full_counts[line, child] = len(rows)
            if not active[line, child]:
                continue
            if len(rows) == 0:
                raise ValueError(
                    f"active Child ({names[line]}, {child}) has no controls"
                )
            local_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(line), int(child)])
            )
            if len(rows) > capacity:
                rows = np.sort(
                    local_rng.choice(rows, size=capacity, replace=False)
                )
            count = len(rows)
            residuals[line, child, :count] = (
                normalized[rows] - control_mean[line]
            )
            member_mask[line, child, :count] = True
            member_obs[line, child, :count] = obs_indices[rows]
            stored_counts[line, child] = count

    arrays = {
        "schema_version": np.asarray(REAL_CONTROL_RESERVOIR_SCHEMA),
        "control_only": np.asarray(True, dtype=bool),
        "treated_expression_used": np.asarray(False, dtype=bool),
        "cellline_names": _fixed_unicode(names),
        "cellline_ids": np.arange(lines, dtype=np.int16),
        "normalization_divisor": np.asarray(divisor, dtype=np.float32),
        "member_residuals": residuals,
        "member_mask": member_mask,
        "source_obs_indices": member_obs,
        "source_control_count": source_counts,
        "full_child_counts": full_counts,
        "stored_child_counts": stored_counts,
        "child_mask": active,
        "child_prototypes": prototypes / np.float32(divisor),
        "control_mean": control_mean,
        "build_seed": np.asarray(int(seed), dtype=np.int64),
    }
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"builder produced unsafe object array {key!r}")
    return arrays


def pack_control_only_from_full_arrays(
    *,
    expression: np.ndarray,
    perturbation_labels: Sequence[str],
    cell_line_ids: np.ndarray,
    child_ids: np.ndarray,
    control_label: str,
    **pack_kwargs,
) -> dict[str, np.ndarray]:
    """Audit-friendly helper proving treated expression is filtered first."""

    values = np.asarray(expression)
    perturbations = np.asarray([str(value) for value in perturbation_labels])
    line_ids = np.asarray(cell_line_ids)
    assigned = np.asarray(child_ids)
    if values.ndim != 2 or perturbations.shape != (len(values),):
        raise ValueError("full expression/perturbation rows do not align")
    if line_ids.shape != (len(values),) or assigned.shape != (len(values),):
        raise ValueError("full line/Child rows do not align")
    control = perturbations == str(control_label)
    if not control.any():
        raise ValueError("no control rows match control_label")
    source_indices = np.flatnonzero(control).astype(np.int64)
    return pack_assigned_controls(
        control_expression=values[control],
        control_line_ids=line_ids[control],
        control_child_ids=assigned[control],
        source_obs_indices=source_indices,
        **pack_kwargs,
    )


def _reconstruct_balanced_labels(
    raw_values: np.ndarray,
    batches: np.ndarray,
    *,
    num_tokens: int,
    pca_dim: int,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-run the exact balanced builder because the legacy bank lost labels."""

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    from src.models.hungarian_flow.build_anchor_bank import batch_center
    from src.models.hungarian_flow.build_balanced_anchor_bank import (
        balanced_kmeans,
    )

    discovery = batch_center(raw_values, batches)
    components = min(int(pca_dim), discovery.shape[0] - 1, discovery.shape[1])
    latent = PCA(
        n_components=max(2, components),
        svd_solver="randomized",
        random_state=int(seed),
    ).fit_transform(discovery)
    initializer = MiniBatchKMeans(
        n_clusters=int(num_tokens),
        batch_size=min(2048, max(256, len(latent))),
        n_init=10,
        random_state=int(seed),
        reassignment_ratio=0.0,
    ).fit(latent)
    labels, _, _ = balanced_kmeans(
        latent, initializer.cluster_centers_, max_iterations=int(iterations)
    )
    means = np.stack(
        [raw_values[labels == child].mean(axis=0) for child in range(num_tokens)]
    ).astype(np.float32)
    counts = np.bincount(labels, minlength=int(num_tokens)).astype(np.int64)
    return labels.astype(np.int64), means, counts


def build_from_h5ad_and_balanced_context(
    *,
    h5ad_path: Path,
    context_bank_path: Path,
    expression_key: str,
    perturbation_key: str,
    control_label: str,
    cell_line_key: str,
    batch_key: str,
    normalization_divisor: float,
    max_members_per_child: int,
    seed: int | None = None,
    prototype_atol: float = 2e-5,
    prototype_rtol: float = 2e-4,
) -> dict[str, np.ndarray]:
    """Read only control expression and reconstruct legacy balanced labels."""

    import anndata as ad

    with Path(context_bank_path).open("rb") as handle:
        context_bank = pickle.load(handle)
    if context_bank.get("schema") != "hungarian_flow_balanced_fixed_anchor_bank_v1":
        raise ValueError("context bank must use the balanced fixed-anchor schema")
    contexts = context_bank.get("context_by_cell_line", {})
    metadata = context_bank.get("metadata", {})
    if not contexts:
        raise ValueError("context bank contains no cell lines")
    num_tokens = int(metadata.get("anchor_capacity", 128))
    pca_dim = int(metadata.get("pca_dim", 64))
    iterations = int(metadata.get("balanced_iterations", 20))
    base_seed = int(metadata.get("seed", 42) if seed is None else seed)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    all_values = []
    all_lines = []
    all_children = []
    all_obs_indices = []
    runtime_prototypes = []
    runtime_child_masks = []
    cellline_names = sorted(str(name) for name in contexts)
    try:
        obs = adata.obs
        perturbations = obs[perturbation_key].astype(str)
        cell_lines = obs[cell_line_key].astype(str)
        control_global = perturbations.eq(str(control_label))
        for line_index, cell_line in enumerate(cellline_names):
            # Critical leakage boundary: positions are filtered before any
            # expression matrix access. Treated rows are never read.
            mask = control_global & cell_lines.eq(cell_line)
            positions = np.flatnonzero(mask.to_numpy()).astype(np.int64)
            if len(positions) < num_tokens:
                raise ValueError(
                    f"cell line {cell_line!r} has fewer controls than Children"
                )
            raw = np.asarray(adata.obsm[expression_key][positions], dtype=np.float32)
            batches = obs.iloc[positions][batch_key].astype(str).to_numpy()
            labels, rebuilt_means, rebuilt_counts = _reconstruct_balanced_labels(
                raw,
                batches,
                num_tokens=num_tokens,
                pca_dim=pca_dim,
                seed=base_seed + line_index,
                iterations=iterations,
            )
            context = contexts[cell_line]
            reference = np.asarray(context["prototype_means"], dtype=np.float32)
            reference_counts = np.asarray(
                context["cluster_cell_counts"], dtype=np.int64
            )
            if reference.shape != rebuilt_means.shape:
                raise ValueError("rebuilt/reference Child prototype shapes differ")
            if not np.allclose(
                rebuilt_means,
                reference,
                atol=float(prototype_atol),
                rtol=float(prototype_rtol),
            ):
                maximum = float(np.max(np.abs(rebuilt_means - reference)))
                raise ValueError(
                    "could not reconstruct exact legacy Child assignments; "
                    f"maximum prototype difference={maximum:.7g}"
                )
            if not np.array_equal(rebuilt_counts, reference_counts):
                raise ValueError("rebuilt/reference Child counts differ")
            weights = np.asarray(context["weights"], dtype=np.float32)
            # Match build_unique_cell_line_control_bank exactly. This sorting
            # defines the Child IDs seen by Hungarian and Router.
            order = np.argsort(-weights)
            inverse_order = np.empty(num_tokens, dtype=np.int64)
            inverse_order[order] = np.arange(num_tokens, dtype=np.int64)
            runtime_labels = inverse_order[labels]
            all_values.append(raw)
            all_lines.append(np.full(len(raw), line_index, dtype=np.int64))
            all_children.append(runtime_labels)
            all_obs_indices.append(positions)
            runtime_prototypes.append(reference[order])
            runtime_child_masks.append(np.ones(num_tokens, dtype=bool))
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    arrays = pack_assigned_controls(
        control_expression=np.concatenate(all_values, axis=0),
        control_line_ids=np.concatenate(all_lines, axis=0),
        control_child_ids=np.concatenate(all_children, axis=0),
        cellline_names=cellline_names,
        child_prototypes=np.stack(runtime_prototypes),
        child_mask=np.stack(runtime_child_masks),
        normalization_divisor=normalization_divisor,
        max_members_per_child=max_members_per_child,
        seed=base_seed,
        source_obs_indices=np.concatenate(all_obs_indices, axis=0),
    )
    arrays.update(
        {
            "context_bank_sha256": np.asarray(_sha256(Path(context_bank_path))),
            "control_label": np.asarray(str(control_label)),
            "expression_key": np.asarray(str(expression_key)),
            "perturbation_key": np.asarray(str(perturbation_key)),
            "cell_line_key": np.asarray(str(cell_line_key)),
            "batch_key": np.asarray(str(batch_key)),
            "assignment_method": np.asarray(
                "exact balanced-kmeans reconstruction with prototype audit"
            ),
        }
    )
    return arrays


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a real-control Child reservoir without treated expression"
    )
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--context-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expression-key", default="X_hvg")
    parser.add_argument("--perturbation-key", default="gene")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--cell-line-key", default="cell_line")
    parser.add_argument("--batch-key", default="gem_group")
    parser.add_argument("--normalization-divisor", type=float, default=10.0)
    parser.add_argument("--max-members-per-child", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prototype-atol", type=float, default=2e-5)
    parser.add_argument("--prototype-rtol", type=float, default=2e-4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    arrays = build_from_h5ad_and_balanced_context(
        h5ad_path=args.h5ad.expanduser().resolve(),
        context_bank_path=args.context_bank.expanduser().resolve(),
        expression_key=args.expression_key,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        cell_line_key=args.cell_line_key,
        batch_key=args.batch_key,
        normalization_divisor=args.normalization_divisor,
        max_members_per_child=args.max_members_per_child,
        seed=args.seed,
        prototype_atol=args.prototype_atol,
        prototype_rtol=args.prototype_rtol,
    )
    output = args.output.expanduser().resolve()
    _atomic_savez(output, arrays)
    summary = {
        "schema_version": REAL_CONTROL_RESERVOIR_SCHEMA,
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "control_only": True,
        "treated_expression_used": False,
        "shape": list(arrays["member_residuals"].shape),
        "source_control_count": arrays["source_control_count"].tolist(),
        "full_child_count_range": [
            int(arrays["full_child_counts"].min()),
            int(arrays["full_child_counts"].max()),
        ],
        "stored_child_count_range": [
            int(arrays["stored_child_counts"].min()),
            int(arrays["stored_child_counts"].max()),
        ],
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_from_h5ad_and_balanced_context",
    "normalized_control_means",
    "pack_assigned_controls",
    "pack_control_only_from_full_arrays",
    "validate_control_assignment_partition",
]
