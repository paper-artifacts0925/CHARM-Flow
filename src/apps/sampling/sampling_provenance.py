"""Fail-closed physical-row provenance for sampling outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from src.evaluation.pbmc_physical_identity import (
    canonical_source_namespace,
    read_source_obs_names,
)


SCHEMA_VERSION = "sampling-physical-row-provenance/v1"
IDENTITY_SEMANTICS = "physical_source_h5_row"
IDENTITY_COLUMNS = (
    "source_namespace",
    "source_dataset_name",
    "source_physical_row",
    "source_obs_name",
    "source_role",
)
SOURCE_FILE_COLUMN = "source_file"

_CONTROL_ROW = "_sampling_source_physical_row"
_CONTROL_OBS_NAME = "_sampling_source_obs_name"
_CONTROL_SOURCE_FILE = "_sampling_source_file"
_CONTROL_DATASET_NAME = "_sampling_source_dataset_name"
_CONTROL_NAMESPACE = "_sampling_source_namespace"


def _canonical_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=True))


def _namespace(kind: str, dataset_name: str, source_file: str) -> str:
    return canonical_source_namespace(
        kind=kind, dataset_name=dataset_name, source_file=source_file
    )


def annotate_control_source(
    adata,
    *,
    source_path: str | Path,
    dataset_name: str,
) -> None:
    """Attach control identities *before* any row filtering is performed."""

    source_file = _canonical_path(source_path)
    physical_rows = np.arange(adata.n_obs, dtype=np.int64)
    source_obs_names = np.asarray(adata.obs_names.astype(str), dtype=object)
    if source_obs_names.shape != physical_rows.shape:
        raise AssertionError("control obs names do not align to physical rows")
    adata.obs[_CONTROL_ROW] = physical_rows
    adata.obs[_CONTROL_OBS_NAME] = source_obs_names
    adata.uns[_CONTROL_SOURCE_FILE] = source_file
    adata.uns[_CONTROL_DATASET_NAME] = str(dataset_name)
    adata.uns[_CONTROL_NAMESPACE] = _namespace(
        "control-h5ad", str(dataset_name), source_file
    )


def collect_treated_provenance(batch_data, dataset, valid_mask) -> pd.DataFrame:
    """Collect treated identities in the exact row-major order used by masking."""

    for key in ("cell_index", "dataset_id"):
        if key not in batch_data:
            raise ValueError(
                f"sampling batch lacks {key!r}; refusing to fabricate source identity"
            )
    mask = torch.as_tensor(valid_mask).detach().cpu().numpy().astype(bool, copy=False)
    cell_index = batch_data["cell_index"].detach().cpu().numpy()
    dataset_id = batch_data["dataset_id"].detach().cpu().numpy()
    if mask.shape != cell_index.shape or mask.shape != dataset_id.shape:
        raise ValueError("sampling provenance tensors do not share the padding-mask shape")
    rows = np.asarray(cell_index[mask], dtype=np.int64)
    ids = np.asarray(dataset_id[mask], dtype=np.int64)
    if np.any(rows < 0) or np.any(ids < 0):
        raise ValueError("valid sampling rows contain missing physical identities")

    id_to_name = {
        int(value): str(name)
        for name, value in getattr(dataset, "datasets_to_num", {}).items()
    }
    if not id_to_name:
        raise ValueError("sampling dataset has no stable datasets_to_num mapping")

    names = np.empty(len(rows), dtype=object)
    files = np.empty(len(rows), dtype=object)
    namespaces = np.empty(len(rows), dtype=object)
    obs_names = np.empty(len(rows), dtype=object)
    for source_id in np.unique(ids):
        if int(source_id) not in id_to_name:
            raise ValueError(f"unknown sampling dataset_id={int(source_id)}")
        name = id_to_name[int(source_id)]
        if name not in dataset.dataset_path_map:
            raise ValueError(f"dataset_id={int(source_id)} has no source-file mapping")
        source_file = _canonical_path(dataset.dataset_path_map[name])
        selected = ids == source_id
        selected_rows = rows[selected]
        source_handle = dataset.store.dataset_file(name)
        names[selected] = name
        files[selected] = source_file
        namespaces[selected] = _namespace("dataset-h5", name, source_file)
        obs_names[selected] = read_source_obs_names(source_handle, selected_rows)

    return pd.DataFrame(
        {
            "source_namespace": namespaces,
            "source_dataset_name": names,
            "source_physical_row": rows,
            "source_obs_name": obs_names,
            "source_role": np.repeat("treated_target", len(rows)),
            SOURCE_FILE_COLUMN: files,
        }
    )


def control_provenance(ctrl_adata) -> pd.DataFrame:
    """Return control lineage retained from the unfiltered control H5AD."""

    required_obs = (_CONTROL_ROW, _CONTROL_OBS_NAME)
    missing = [key for key in required_obs if key not in ctrl_adata.obs]
    required_uns = (
        _CONTROL_SOURCE_FILE,
        _CONTROL_DATASET_NAME,
        _CONTROL_NAMESPACE,
    )
    missing_uns = [key for key in required_uns if key not in ctrl_adata.uns]
    if missing or missing_uns:
        raise ValueError(
            "control H5AD lacks pre-filter physical provenance; refusing to infer "
            f"it from output order (obs={missing}, uns={missing_uns})"
        )
    rows = pd.to_numeric(ctrl_adata.obs[_CONTROL_ROW], errors="coerce")
    if rows.isna().any() or np.any(rows.to_numpy(dtype=np.int64) < 0):
        raise ValueError("control source physical rows are invalid")
    n_obs = int(ctrl_adata.n_obs)
    return pd.DataFrame(
        {
            "source_namespace": np.repeat(
                str(ctrl_adata.uns[_CONTROL_NAMESPACE]), n_obs
            ),
            "source_dataset_name": np.repeat(
                str(ctrl_adata.uns[_CONTROL_DATASET_NAME]), n_obs
            ),
            "source_physical_row": rows.to_numpy(dtype=np.int64),
            "source_obs_name": ctrl_adata.obs[_CONTROL_OBS_NAME]
            .astype(str)
            .to_numpy(),
            "source_role": np.repeat("reference_control", n_obs),
            SOURCE_FILE_COLUMN: np.repeat(
                str(ctrl_adata.uns[_CONTROL_SOURCE_FILE]), n_obs
            ),
        }
    )


def attach_paired_output_provenance(
    obs: pd.DataFrame,
    treated_batches: Iterable[pd.DataFrame],
    ctrl_adata,
) -> pd.DataFrame:
    """Attach one authoritative lineage frame shared by real and prediction."""

    treated = list(treated_batches)
    if not treated:
        raise ValueError("sampling produced no treated physical provenance")
    lineage = pd.concat([*treated, control_provenance(ctrl_adata)], ignore_index=True)
    if len(lineage) != len(obs):
        raise AssertionError(
            f"sampling obs/provenance row mismatch: {len(obs)} != {len(lineage)}"
        )
    for column in (*IDENTITY_COLUMNS, SOURCE_FILE_COLUMN):
        if column not in lineage or lineage[column].isna().any():
            raise AssertionError(f"sampling provenance column {column!r} is incomplete")
        obs[column] = lineage[column].to_numpy(copy=True)
    return obs


def add_provenance_metadata(adata) -> None:
    """Describe source bindings; SHA256 sealing is intentionally downstream."""

    missing = [column for column in (*IDENTITY_COLUMNS, SOURCE_FILE_COLUMN) if column not in adata.obs]
    if missing:
        raise ValueError(f"sampling H5AD lacks physical provenance columns {missing}")
    bindings = (
        adata.obs[["source_namespace", "source_dataset_name", SOURCE_FILE_COLUMN, "source_role"]]
        .drop_duplicates()
        .sort_values(["source_namespace", "source_role"], kind="stable")
        .to_dict(orient="records")
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity_semantics": IDENTITY_SEMANTICS,
        "identity_columns": list(IDENTITY_COLUMNS),
        "source_file_column": SOURCE_FILE_COLUMN,
        "source_bindings": bindings,
        "source_sha256_policy": "required_in_evaluation_manifest",
        "historical_output_order_inference_allowed": False,
    }
    # A JSON scalar is stable across AnnData versions, unlike nested lists of
    # mappings whose HDF5 serialization support varies.
    adata.uns["sampling_physical_row_provenance_json"] = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def assert_paired_provenance_equal(real_adata, pred_adata) -> None:
    """Fail before writing if real/pred lineage differs in value or row order."""

    if real_adata.n_obs != pred_adata.n_obs:
        raise AssertionError("real/pred output row counts differ")
    columns = [*IDENTITY_COLUMNS, SOURCE_FILE_COLUMN]
    for column in columns:
        if column not in real_adata.obs or column not in pred_adata.obs:
            raise AssertionError(f"real/pred output lacks provenance column {column!r}")
        left = real_adata.obs[column].astype(str).to_numpy()
        right = pred_adata.obs[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise AssertionError(f"real/pred provenance differs in {column!r}")

