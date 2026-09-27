import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from pdex import parallel_differential_expression


def _decode(values) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.asarray([value.decode() for value in values], dtype=str)
    return values.astype(str)


def _categorical(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Dataset):
        return _decode(node[:])
    categories = _decode(node["categories"][:])
    return categories[np.asarray(node["codes"][:], dtype=np.int64)]


def _condition(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "condition must be CELL_LINE=PERTURBATION"
        )
    line, perturbation = value.split("=", 1)
    if not line or not perturbation:
        raise argparse.ArgumentTypeError(
            "condition must be CELL_LINE=PERTURBATION"
        )
    return line, perturbation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--control", default="non-targeting")
    parser.add_argument(
        "--cell-line",
        action="append",
        default=None,
        help="optional repeatable cell-line subset for safe parallel precompute",
    )
    parser.add_argument(
        "--parent-calibration",
        choices=("none", "no_intercept", "affine"),
        default="no_intercept",
        help="must match the runtime Parent calibration mode",
    )
    parser.add_argument(
        "--audit-condition",
        action="append",
        type=_condition,
        default=[],
        help=(
            "repeat CELL_LINE=PERTURBATION; audit treated rows are excluded "
            "before expression is read"
        ),
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.parent_artifact, allow_pickle=False) as parent:
        genes = parent["gene_names"].astype(str)
        lines = parent["cellline_names"].astype(str)
        perturbations = parent["perturbation_names"].astype(str)
        train_mask = parent["train_condition_mask"].astype(bool)
        validation_mask = parent["validation_condition_mask"].astype(bool)
        test_mask = parent["test_condition_mask"].astype(bool)
        divisor = float(parent["normalization_divisor"])
        parent_ridge_support = (
            parent["ridge_support_mask"].astype(bool)
            if "ridge_support_mask" in parent.files
            else None
        )
    if np.any(train_mask & (validation_mask | test_mask)):
        raise ValueError("Parent artifact train/held-out masks overlap")
    selected_lines = set(args.cell_line or lines.tolist())
    unknown_lines = selected_lines - set(lines.tolist())
    if unknown_lines:
        raise ValueError(f"unknown --cell-line values: {sorted(unknown_lines)}")
    line_index = {str(name): index for index, name in enumerate(lines)}
    perturbation_index = {
        str(name): index for index, name in enumerate(perturbations)
    }
    audit_pairs = [(str(line), str(pert)) for line, pert in args.audit_condition]
    if len(set(audit_pairs)) != len(audit_pairs):
        raise ValueError("duplicate audit condition")
    audit_mask = np.zeros_like(train_mask, dtype=bool)
    for line_name, perturbation_name in audit_pairs:
        if line_name not in line_index:
            raise ValueError(f"unknown audit cell line {line_name!r}")
        if perturbation_name not in perturbation_index:
            raise ValueError(f"unknown audit perturbation {perturbation_name!r}")
        line_id = line_index[line_name]
        perturbation_id = perturbation_index[perturbation_name]
        if validation_mask[line_id, perturbation_id] or test_mask[line_id, perturbation_id]:
            raise ValueError(
                f"audit condition {(line_name, perturbation_name)!r} overlaps validation/test"
            )
        if not train_mask[line_id, perturbation_id]:
            raise ValueError(
                f"audit condition {(line_name, perturbation_name)!r} is not Parent train"
            )
        audit_mask[line_id, perturbation_id] = True
    if args.parent_calibration == "none":
        calibration_support = np.zeros_like(train_mask, dtype=bool)
    else:
        if parent_ridge_support is None:
            raise ValueError(
                "calibrated Parent artifact is missing ridge_support_mask provenance"
            )
        if parent_ridge_support.shape != train_mask.shape:
            raise ValueError("Parent ridge_support_mask has the wrong [L,P] shape")
        calibration_support = parent_ridge_support
    overlap = calibration_support & audit_mask
    if overlap.any():
        examples = [
            (str(lines[line_id]), str(perturbations[perturbation_id]))
            for line_id, perturbation_id in np.argwhere(overlap)[:5]
        ]
        raise ValueError(
            "Parent calibration leakage: ridge_support_mask contains audit "
            f"conditions (count={int(overlap.sum())}, examples={examples}); "
            "rebuild/derive calibration without audit or use --parent-calibration none"
        )
    if np.any(calibration_support & (validation_mask | test_mask)):
        raise ValueError(
            "Parent calibration support contains validation/test conditions"
        )
    train_mask = train_mask & ~audit_mask

    # The Parent calibration/audit gate above runs before the H5AD is opened,
    # so a leaky configuration cannot spend hours computing unusable DE tables.
    with h5py.File(args.h5ad, "r") as handle:
        obs_lines = _categorical(handle["obs"], "cell_line")
        obs_perturbations = _categorical(handle["obs"], "gene")
        expression = handle["obsm"]["X_hvg"]
        if expression.shape[1] != len(genes):
            raise ValueError("X_hvg/Parent artifact gene dimension mismatch")
        manifest = {}
        for line_id, line_name in enumerate(lines):
            if str(line_name) not in selected_lines:
                continue
            allowed_targets = set(perturbations[train_mask[line_id]].tolist())
            row_mask = (obs_lines == line_name) & (
                np.isin(obs_perturbations, list(allowed_targets))
                | (obs_perturbations == args.control)
            )
            rows = np.flatnonzero(row_mask)
            selected_labels = obs_perturbations[rows]
            forbidden = set(selected_labels) - allowed_targets - {args.control}
            if forbidden:
                raise AssertionError(
                    f"audit/held-out rows survived filtering: {sorted(forbidden)}"
                )
            audit_targets = set(perturbations[audit_mask[line_id]].tolist())
            if audit_targets and np.isin(selected_labels, list(audit_targets)).any():
                raise AssertionError("audit treated rows survived pre-expression filter")
            # This is the first point where expression is read. The row set is
            # already proven to contain controls and train_condition_mask only.
            values = np.asarray(expression[rows], dtype=np.float32) / divisor
            frame = ad.AnnData(
                X=values,
                obs=pd.DataFrame(
                    {"gene": selected_labels},
                    index=[f"{line_name}_{index}" for index in range(len(rows))],
                ),
            )
            frame.var_names = genes
            de = parallel_differential_expression(
                adata=frame,
                groups=sorted(allowed_targets),
                reference=args.control,
                groupby_key="gene",
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                metric="wilcoxon",
                is_log1p=True,
                as_polars=True,
            )
            observed_targets = set(de["target"].unique().to_list())
            if observed_targets != allowed_targets:
                raise ValueError(
                    f"{line_name}: DE target coverage mismatch; "
                    f"missing={sorted(allowed_targets-observed_targets)[:10]}"
                )
            path = args.outdir / f"{line_name}_train_only_de.csv"
            de.write_csv(path)
            manifest[str(line_name)] = {
                "path": str(path.resolve()),
                "rows": int(de.height),
                "cells": int(len(rows)),
                "train_conditions": int(len(allowed_targets)),
                "audit_conditions": int(audit_mask[line_id].sum()),
                "audit_treated_rows_read": 0,
                "validation_test_treated_rows_read": 0,
                "parent_calibration_mode": args.parent_calibration,
                "parent_calibration_support_overlaps_audit": False,
            }
            del frame, values, de
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
