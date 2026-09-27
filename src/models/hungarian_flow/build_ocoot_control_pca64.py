#!/usr/bin/env python
"""Build a dataset-specific control-only PCA64 basis for OCOO-T."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.models.hungarian_flow.build_ocoot_recursive_artifacts import (
    OCOOT_DATASET_SPECS,
    OCOOT_GENE_DIM,
    _atomic_savez,
    _atomic_write_json,
    _fixed_unicode,
    resolve_ocoot_source_paths,
)
from src.models.hungarian_flow.build_ocoot_recursive_artifacts_v2 import (
    OCOOT_SCRATCH_NORMALIZATION_DIVISOR,
    load_ocoot_control_rows_v2,
)
from src.models.hungarian_flow.context_artifact import sha256_file


OCOOT_PCA_COMPONENTS = 64


def build_ocoot_control_pca64(
    *,
    dataset: str,
    input_path: str | Path,
    output: str | Path,
    selected_gene_file: str | Path | None = None,
    feature_key: str = "X_hvg",
    expected_source_files: int | None = None,
    normalize_counts: float = OCOOT_SCRATCH_NORMALIZATION_DIVISOR,
    seed: int = 42,
) -> dict:
    """Fit and persist the existing Parent-residual PCA artifact schema."""

    sources = resolve_ocoot_source_paths(
        dataset,
        input_path,
        expected_source_files=expected_source_files,
    )
    controls = load_ocoot_control_rows_v2(
        sources,
        dataset=dataset,
        feature_key=feature_key,
        selected_gene_file=selected_gene_file,
    )
    divisor = float(normalize_counts)
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("normalize_counts must be finite and positive")
    if len(controls.expression) <= OCOOT_PCA_COMPONENTS:
        raise ValueError(
            f"PCA64 requires more than {OCOOT_PCA_COMPONENTS} controls; "
            f"found {len(controls.expression)}"
        )
    values = controls.expression.astype(np.float32, copy=True)
    values /= np.float32(divisor)
    fitted = PCA(
        n_components=OCOOT_PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=int(seed),
    ).fit(values)
    components = fitted.components_.astype(np.float32)
    mean = fitted.mean_.astype(np.float32)
    variance = fitted.explained_variance_.astype(np.float32)
    ratio = fitted.explained_variance_ratio_.astype(np.float32)
    if components.shape != (OCOOT_PCA_COMPONENTS, OCOOT_GENE_DIM):
        raise RuntimeError("PCA components do not satisfy the [64,2000] contract")
    if mean.shape != (OCOOT_GENE_DIM,):
        raise RuntimeError("PCA mean does not satisfy the [2000] contract")
    if not (
        np.isfinite(components).all()
        and np.isfinite(mean).all()
        and np.isfinite(variance).all()
        and np.isfinite(ratio).all()
    ):
        raise ValueError("PCA fit produced non-finite values")
    if (variance <= 0.0).any():
        raise ValueError("PCA64 explained variance must be strictly positive")

    arrays = {
        # This exactly matches replogle_control_pca64.npz.
        "components": components,
        "mean": mean,
        "explained_variance": variance,
        "explained_variance_ratio": ratio,
        "genes": _fixed_unicode(controls.gene_names),
        "n_control": np.asarray([len(values)], dtype=np.int64),
        "normalize_counts": np.asarray([divisor], dtype=np.float32),
    }
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"PCA builder produced unsafe object array {key!r}")
    destination = Path(output).expanduser().resolve()
    _atomic_savez(destination, arrays)
    report = {
        "artifact": str(destination),
        "artifact_sha256": sha256_file(destination),
        "schema": "replogle_control_pca64_compatible",
        "builder": "ocoot_control_pca64_v2",
        "dataset": str(dataset).lower(),
        "source_h5ads": [str(path) for path in sources],
        "source_file_count": len(sources),
        "selected_gene_file": (
            None
            if selected_gene_file is None
            else str(Path(selected_gene_file).expanduser().resolve())
        ),
        "feature_key": feature_key,
        "control_only": True,
        "treated_expression_used": False,
        "n_control": int(len(values)),
        "genes": OCOOT_GENE_DIM,
        "components": OCOOT_PCA_COMPONENTS,
        "normalize_counts": divisor,
        "seed": int(seed),
        "explained_variance_ratio_sum": float(ratio.sum()),
    }
    _atomic_write_json(
        destination.with_suffix(destination.suffix + ".json"), report
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build PBMC/Tahoe control-only PCA64 in scratch /10 space"
    )
    parser.add_argument("--dataset", choices=sorted(OCOOT_DATASET_SPECS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-gene-file", type=Path, default=None)
    parser.add_argument("--feature-key", choices=("X", "X_hvg"), default="X_hvg")
    parser.add_argument(
        "--normalize-counts",
        type=float,
        default=OCOOT_SCRATCH_NORMALIZATION_DIVISOR,
        help="must match active scratch data.normalize_counts (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_ocoot_control_pca64(
        dataset=args.dataset,
        input_path=args.input,
        output=args.output,
        selected_gene_file=args.selected_gene_file,
        feature_key=args.feature_key,
        normalize_counts=args.normalize_counts,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OCOOT_PCA_COMPONENTS", "build_ocoot_control_pca64"]
