#!/usr/bin/env python
"""Bounded-memory OCOO-T v2 builder for dense ``X_hvg`` H5AD files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from src.models.hungarian_flow import build_ocoot_recursive_artifacts_v2 as v2
from src.models.hungarian_flow.build_ocoot_recursive_artifacts import (
    OCOOT_GENE_DIM,
    ControlRows,
    get_ocoot_dataset_spec,
    load_selected_genes,
)


READ_CHUNK_ROWS = 4096


def _strings(values) -> np.ndarray:
    array = np.asarray(values)
    flat = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in array.reshape(-1).tolist()
    ]
    width = max((len(value) for value in flat), default=1)
    return np.asarray(flat, dtype=f"<U{width}").reshape(array.shape)


def _index_dataset(group: h5py.Group) -> h5py.Dataset:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, (bytes, np.bytes_)):
        key = key.decode("utf-8")
    node = group.get(str(key))
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"H5AD dataframe is missing index dataset {key!r}")
    return node


def _categorical(node, positions: np.ndarray | None = None) -> np.ndarray:
    if isinstance(node, h5py.Group):
        if "codes" not in node or "categories" not in node:
            raise ValueError("unsupported H5AD categorical encoding")
        codes = np.asarray(node["codes"][...])
        if positions is not None:
            codes = codes[positions]
        categories = _strings(node["categories"][...]).reshape(-1)
        if ((codes < 0) | (codes >= len(categories))).any():
            raise ValueError("missing or invalid H5AD categorical code")
        return categories[codes]
    if not isinstance(node, h5py.Dataset):
        raise ValueError("unsupported H5AD obs-column encoding")
    values = node[...] if positions is None else node[positions]
    return _strings(values)


def _matching_positions(node, label: str) -> np.ndarray:
    if isinstance(node, h5py.Group):
        categories = _strings(node["categories"][...]).reshape(-1)
        match = np.flatnonzero(categories == str(label))
        if len(match) != 1:
            return np.empty(0, dtype=np.int64)
        codes = np.asarray(node["codes"][...])
        return np.flatnonzero(codes == int(match[0])).astype(np.int64)
    return np.flatnonzero(_categorical(node) == str(label)).astype(np.int64)


def _gene_names(
    handle: h5py.File,
    *,
    source: Path,
    feature_key: str,
    selected_genes: Sequence[str] | None,
) -> np.ndarray:
    var = handle["var"]
    names = _strings(_index_dataset(var)[...]).reshape(-1)
    hvg_names = None
    if "highly_variable" in var:
        mask = np.asarray(var["highly_variable"][...], dtype=bool)
        if mask.shape != names.shape:
            raise ValueError(f"{source}: var.highly_variable does not align with var")
        if int(mask.sum()) == OCOOT_GENE_DIM:
            hvg_names = names[mask]
    if feature_key == "X":
        genes = names
    elif selected_genes is not None:
        genes = np.asarray(selected_genes, dtype=str)
        if hvg_names is not None and not np.array_equal(genes, hvg_names):
            raise ValueError(
                f"{source}: selected gene order differs from var/highly_variable"
            )
    elif hvg_names is not None:
        genes = hvg_names
    elif len(names) == OCOOT_GENE_DIM:
        genes = names
    else:
        raise ValueError(
            f"{source}: cannot prove {feature_key} gene order; provide selected genes"
        )
    if genes.shape != (OCOOT_GENE_DIM,) or len(np.unique(genes)) != len(genes):
        raise ValueError(f"{source}: expected {OCOOT_GENE_DIM} unique feature genes")
    return genes.astype(str, copy=False)


def _stream_dense_rows(
    dataset: h5py.Dataset,
    positions: np.ndarray,
    output: np.ndarray,
    offset: int,
) -> None:
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("selected row positions must be one-dimensional")
    if len(positions) and (
        positions[0] < 0
        or positions[-1] >= int(dataset.shape[0])
        or (np.diff(positions) <= 0).any()
    ):
        raise ValueError("selected row positions must be sorted, unique, and in range")
    cursor = int(offset)
    run_starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(positions) != 1) + 1)
    ) if len(positions) else np.empty(0, dtype=np.int64)
    run_stops = np.concatenate(
        (run_starts[1:], np.asarray([len(positions)], dtype=np.int64))
    ) if len(positions) else np.empty(0, dtype=np.int64)
    for left, right in zip(run_starts.tolist(), run_stops.tolist()):
        source_start = int(positions[left])
        source_stop = int(positions[right - 1]) + 1
        selected = np.asarray(dataset[source_start:source_stop], dtype=np.float32)
        expected_rows = int(right - left)
        if selected.shape[0] != expected_rows:
            raise RuntimeError("continuous control-row read returned the wrong length")
        if not np.isfinite(selected).all():
            raise ValueError("selected control expression contains non-finite values")
        output[cursor : cursor + expected_rows] = selected
        cursor += expected_rows
    if cursor != int(offset) + len(positions):
        raise RuntimeError("did not stream every selected control row")


def load_ocoot_control_rows_v2_lowmem(
    source_paths: Sequence[str | Path],
    *,
    dataset: str,
    feature_key: str = "X_hvg",
    selected_gene_file: str | Path | None = None,
) -> ControlRows:
    """Load only controls without asking AnnData to open enormous ``obsm``."""

    spec = get_ocoot_dataset_spec(dataset)
    paths = tuple(Path(path).expanduser().resolve() for path in source_paths)
    if not paths:
        raise ValueError("at least one source H5AD is required")
    selected = (
        None
        if selected_gene_file in (None, "", "null")
        else load_selected_genes(selected_gene_file)
    )
    positions_parts, lines, batches = [], [], []
    logical, source_ids, source_rows, source_sizes = [], [], [], []
    reference_genes = None
    logical_offset = 0
    matrix_path = "X" if feature_key == "X" else f"obsm/{feature_key}"
    for source_id, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            obs = handle["obs"]
            missing = [
                key
                for key in (spec.perturbation_key, spec.cell_line_key, spec.batch_key)
                if key not in obs
            ]
            if missing:
                raise ValueError(f"{path}: missing obs columns {missing}")
            matrix = handle.get(matrix_path)
            if not isinstance(matrix, h5py.Dataset):
                raise ValueError(f"{path}: {matrix_path} must be a dense HDF5 dataset")
            n_obs = int(_index_dataset(obs).shape[0])
            if matrix.shape != (n_obs, OCOOT_GENE_DIM):
                raise ValueError(
                    f"{path}: {matrix_path} shape {matrix.shape} does not match "
                    f"[n_obs,{OCOOT_GENE_DIM}]"
                )
            genes = _gene_names(
                handle,
                source=path,
                feature_key=feature_key,
                selected_genes=selected,
            )
            if reference_genes is None:
                reference_genes = genes.copy()
            elif not np.array_equal(reference_genes, genes):
                raise ValueError(f"{path}: feature gene order differs from first shard")
            positions = _matching_positions(obs[spec.perturbation_key], spec.control_label)
            if not len(positions):
                raise ValueError(f"{path}: no rows match control {spec.control_label!r}")
            positions_parts.append(positions)
            lines.append(_categorical(obs[spec.cell_line_key], positions))
            batches.append(_categorical(obs[spec.batch_key], positions))
            logical.append(positions + logical_offset)
            source_ids.append(np.full(len(positions), source_id, dtype=np.int16))
            source_rows.append(positions)
            source_sizes.append(n_obs)
            logical_offset += n_obs

    expression = np.empty(
        (sum(len(value) for value in positions_parts), OCOOT_GENE_DIM),
        dtype=np.float32,
    )
    offset = 0
    for path, positions in zip(paths, positions_parts):
        with h5py.File(path, "r") as handle:
            _stream_dense_rows(handle[matrix_path], positions, expression, offset)
        offset += len(positions)
    return ControlRows(
        source_paths=paths,
        expression=expression,
        cell_lines=np.concatenate(lines).astype(str, copy=False),
        batches=np.concatenate(batches).astype(str, copy=False),
        logical_obs_indices=np.concatenate(logical).astype(np.int64, copy=False),
        source_file_ids=np.concatenate(source_ids).astype(np.int16, copy=False),
        source_obs_indices=np.concatenate(source_rows).astype(np.int64, copy=False),
        source_n_obs=np.asarray(source_sizes, dtype=np.int64),
        gene_names=np.asarray(reference_genes, dtype=str),
    )


def main(argv=None) -> int:
    args = v2.parse_args(argv)
    original = v2.load_ocoot_control_rows_v2
    v2.load_ocoot_control_rows_v2 = load_ocoot_control_rows_v2_lowmem
    try:
        report = v2.build_ocoot_recursive_artifacts_v2(
            dataset=args.dataset,
            input_path=args.input,
            selected_gene_file=args.selected_gene_file,
            feature_key=args.feature_key,
            context_output=args.context_output,
            reservoir_output=args.reservoir_output,
            anchor_capacity=args.anchor_capacity,
            target_child_occupancy=args.target_child_occupancy,
            minimum_support=args.minimum_support,
            pca_dim=args.pca_dim,
            kmeans_n_init=args.kmeans_n_init,
            silhouette_sample_size=args.silhouette_sample_size,
            seed=args.seed,
            insufficient_group_policy=args.insufficient_group_policy,
            control_grouping=args.control_grouping,
            clustering_mode=args.clustering_mode,
            normalization_divisor=args.normalization_divisor,
            max_members_per_child=args.max_members_per_child,
        )
    finally:
        v2.load_ocoot_control_rows_v2 = original
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_ocoot_control_rows_v2_lowmem"]
