import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    ROOT
    / "outputs/pbmc_ctxw256_14k_full_test/seed42/by_cell_type"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/pbmc_ctxw256_14k_table4_eval/seed42/bulk"
)
CONTROL = "PBS"
PERT_COL = "cytokine"
EXPECTED = {
    "stages": 18,
    "perturbations": 62,
    "genes": 2_000,
    "treated_cells": 2_260_453,
    "control_cells": 235_478,
    "total_cells": 2_495_931,
    "donors": 4,
}
METRIC_COLUMNS = ("PDCorr", "MSE", "MAE", "PDS_L1", "PDS_L2", "PDS_cos")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, block_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def csv_dump(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def stage_directories(input_root: Path) -> list[Path]:
    stages = sorted(
        path.parent
        for path in input_root.glob("*/stage_manifest.json")
        if path.parent.is_dir()
    )
    require(
        len(stages) == EXPECTED["stages"],
        f"expected {EXPECTED['stages']} stages, found {len(stages)} in {input_root}",
    )
    return stages


def label_codes(values: pd.Series, labels: list[str]) -> np.ndarray:
    categorical = pd.Categorical(values.astype(str), categories=labels)
    codes = categorical.codes.astype(np.int32, copy=False)
    require(not np.any(codes < 0), "encountered a cytokine outside the frozen label set")
    return codes


def aggregate_h5ad(
    path: Path,
    labels: list[str],
    expected_genes: list[str],
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream one AnnData matrix into condition sums and counts."""

    data = ad.read_h5ad(path, backed="r")
    try:
        require(PERT_COL in data.obs, f"{path}: missing obs[{PERT_COL!r}]")
        genes = data.var_names.astype(str).tolist()
        require(genes == expected_genes, f"{path}: gene names/order differ")
        codes = label_codes(data.obs[PERT_COL], labels)
        n_labels = len(labels)
        n_genes = len(genes)
        sums = np.zeros((n_labels, n_genes), dtype=np.float64)
        counts = np.bincount(codes, minlength=n_labels).astype(np.int64)

        observed_rows = 0
        nonfinite_values = 0
        minimum = np.inf
        maximum = -np.inf
        for start in range(0, data.n_obs, chunk_rows):
            stop = min(start + chunk_rows, data.n_obs)
            block = data.X[start:stop]
            block_codes = codes[start:stop]

            if sparse.issparse(block):
                block = block.tocsr()
                values = block.data
                if values.size:
                    nonfinite_values += int((~np.isfinite(values)).sum())
                    minimum = min(minimum, float(values.min()))
                    maximum = max(maximum, float(values.max()))
                membership = sparse.csr_matrix(
                    (
                        np.ones(stop - start, dtype=np.float64),
                        (np.arange(stop - start), block_codes),
                    ),
                    shape=(stop - start, n_labels),
                )
                grouped = membership.T @ block
                grouped = grouped.toarray() if sparse.issparse(grouped) else grouped
                sums += np.asarray(grouped, dtype=np.float64)
            else:
                dense = np.asarray(block)
                nonfinite_values += int((~np.isfinite(dense)).sum())
                if dense.size:
                    minimum = min(minimum, float(dense.min()))
                    maximum = max(maximum, float(dense.max()))
                np.add.at(sums, block_codes, dense)
            observed_rows += stop - start

        require(observed_rows == data.n_obs, f"{path}: row accounting failed")
        require(nonfinite_values == 0, f"{path}: found {nonfinite_values} non-finite values")
        stats = {
            "path": str(path.resolve()),
            "rows": int(data.n_obs),
            "genes": int(data.n_vars),
            "matrix_backend": type(data.X).__name__,
            "matrix_dtype": str(data.X.dtype),
            "minimum": float(minimum),
            "maximum": float(maximum),
            "cell_types": sorted(data.obs["cell_type"].astype(str).unique().tolist()),
            "donors": sorted(data.obs["donor"].astype(str).unique().tolist()),
        }
        return sums, counts, stats
    finally:
        data.file.close()


def compute_metrics(
    labels: list[str],
    genes: list[str],
    real_means: np.ndarray,
    pred_means: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Invoke CellEval's exact non-DE metric implementations on pseudobulks."""

    from cell_eval import PerturbationAnndataPair
    from cell_eval.metrics._anndata import (
        discrimination_score,
        mae,
        mse,
        pearson_delta,
    )

    obs = pd.DataFrame({PERT_COL: labels}, index=[f"condition_{i}" for i in range(len(labels))])
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    real = ad.AnnData(X=real_means, obs=obs.copy(), var=var.copy())
    pred = ad.AnnData(X=pred_means, obs=obs.copy(), var=var.copy())
    pair = PerturbationAnndataPair(
        real=real,
        pred=pred,
        pert_col=PERT_COL,
        control_pert=CONTROL,
    )

    per_metric: dict[str, dict[str, float]] = {
        "PDCorr": pearson_delta(pair),
        "MSE": mse(pair),
        "MAE": mae(pair),
        "PDS_L1": discrimination_score(pair, metric="l1"),
        "PDS_L2": discrimination_score(pair, metric="l2"),
        "PDS_cos": discrimination_score(pair, metric="cosine"),
    }
    perts = pair.perts.astype(str).tolist()
    rows = [{"cytokine": pert, **{name: values[pert] for name, values in per_metric.items()}} for pert in perts]
    frame = pd.DataFrame(rows, columns=("cytokine", *METRIC_COLUMNS))
    macro = {name: float(np.mean(frame[name].to_numpy(dtype=np.float64))) for name in METRIC_COLUMNS}
    require(
        all(np.isfinite(value) for value in macro.values()),
        f"non-finite aggregate metric(s): {macro}",
    )
    return frame, macro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=8_192)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace outputs from a previous complete/partial invocation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    require(input_root.is_dir(), f"input root does not exist: {input_root}")
    require(args.chunk_rows > 0, "--chunk-rows must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    terminal_outputs = [
        output_dir / "metrics.json",
        output_dir / "metrics.csv",
        output_dir / "per_cytokine_metrics.csv",
        output_dir / "pseudobulk_arrays.npz",
        output_dir / "audit_manifest.json",
        output_dir / "PASS",
    ]
    if not args.overwrite:
        collisions = [str(path) for path in terminal_outputs if path.exists()]
        require(not collisions, f"refusing to overwrite existing output(s): {collisions}")

    stages = stage_directories(input_root)
    first_real = ad.read_h5ad(stages[0] / "real.h5ad", backed="r")
    try:
        genes = first_real.var_names.astype(str).tolist()
        labels = np.unique(first_real.obs[PERT_COL].to_numpy(dtype=str)).astype(str).tolist()
    finally:
        first_real.file.close()
    require(CONTROL in labels, f"control {CONTROL!r} is absent")
    require(len(labels) - 1 == EXPECTED["perturbations"], f"expected 62 perturbations, found {len(labels) - 1}")
    require(len(genes) == EXPECTED["genes"], f"expected 2000 genes, found {len(genes)}")

    real_sums = np.zeros((len(labels), len(genes)), dtype=np.float64)
    pred_sums = np.zeros_like(real_sums)
    real_counts = np.zeros(len(labels), dtype=np.int64)
    pred_counts = np.zeros_like(real_counts)
    stage_audits: list[dict[str, Any]] = []
    declared_gene_hashes: set[str] = set()
    observed_cell_types: set[str] = set()
    observed_donors: set[str] = set()

    for stage_index, stage in enumerate(stages, start=1):
        manifest_path = stage / "stage_manifest.json"
        pred_path = stage / "pred.h5ad"
        real_path = stage / "real.h5ad"
        require(pred_path.is_file(), f"missing prediction: {pred_path}")
        require(real_path.is_file(), f"missing truth: {real_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            stage_manifest = json.load(handle)
        require(stage_manifest.get("passed") is True, f"{manifest_path}: stage did not pass")
        require(stage_manifest.get("sampling_seed") == 42, f"{manifest_path}: sampling seed is not 42")
        require(stage_manifest.get("source_prediction_values_unchanged") is True, f"{manifest_path}: source predictions were changed")
        require(stage_manifest.get("canonical_controls_appended_once") is True, f"{manifest_path}: control contract failed")
        declared_gene_hashes.add(str(stage_manifest["gene_axis_sha256"]))

        print(f"[{stage_index:02d}/{len(stages)}] aggregating real {stage.name}", flush=True)
        stage_real_sums, stage_real_counts, real_stats = aggregate_h5ad(
            real_path, labels, genes, args.chunk_rows
        )
        print(f"[{stage_index:02d}/{len(stages)}] aggregating pred {stage.name}", flush=True)
        stage_pred_sums, stage_pred_counts, pred_stats = aggregate_h5ad(
            pred_path, labels, genes, args.chunk_rows
        )
        require(
            np.array_equal(stage_real_counts, stage_pred_counts),
            f"{stage}: predicted and real cytokine counts differ",
        )
        require(real_stats["cell_types"] == pred_stats["cell_types"], f"{stage}: cell-type mismatch")
        require(real_stats["donors"] == pred_stats["donors"], f"{stage}: donor mismatch")
        require(len(real_stats["cell_types"]) == 1, f"{stage}: stage is not one cell type")

        real_sums += stage_real_sums
        pred_sums += stage_pred_sums
        real_counts += stage_real_counts
        pred_counts += stage_pred_counts
        observed_cell_types.update(real_stats["cell_types"])
        observed_donors.update(real_stats["donors"])
        stage_audits.append(
            {
                "slug": stage.name,
                "stage_manifest": str(manifest_path.resolve()),
                "stage_manifest_sha256": sha256_file(manifest_path),
                "declared_pred_sha256": stage_manifest.get("pred_sha256"),
                "declared_real_sha256": stage_manifest.get("real_sha256"),
                "pred_file_bytes": pred_path.stat().st_size,
                "real_file_bytes": real_path.stat().st_size,
                "real": real_stats,
                "pred": pred_stats,
            }
        )

    require(len(declared_gene_hashes) == 1, "stage manifests disagree on gene-axis hash")
    require(len(observed_cell_types) == EXPECTED["stages"], "cell-type stages are not unique")
    require(len(observed_donors) == EXPECTED["donors"], f"expected four donors, found {observed_donors}")
    require(np.array_equal(real_counts, pred_counts), "global predicted and real counts differ")
    control_position = labels.index(CONTROL)
    treated_rows = int(real_counts.sum() - real_counts[control_position])
    control_rows = int(real_counts[control_position])
    require(treated_rows == EXPECTED["treated_cells"], f"treated-cell contract: {treated_rows}")
    require(control_rows == EXPECTED["control_cells"], f"control-cell contract: {control_rows}")
    require(int(real_counts.sum()) == EXPECTED["total_cells"], "total-cell contract failed")

    real_means = real_sums / real_counts[:, None]
    pred_means_from_inputs = pred_sums / pred_counts[:, None]
    pred_means = pred_means_from_inputs.copy()
    pred_input_control_max_abs_diff = float(
        np.max(np.abs(pred_means[control_position] - real_means[control_position]))
    )
    pred_means[control_position] = real_means[control_position]

    per_cytokine, macro = compute_metrics(labels, genes, real_means, pred_means)
    metric_frame = pd.DataFrame(
        [{"metric": name, "value": macro[name]} for name in METRIC_COLUMNS]
    )
    counts_frame = pd.DataFrame(
        {
            "cytokine": labels,
            "real_cells": real_counts,
            "pred_cells": pred_counts,
            "is_control": [label == CONTROL for label in labels],
        }
    )

    csv_dump(output_dir / "per_cytokine_metrics.csv", per_cytokine)
    csv_dump(output_dir / "metrics.csv", metric_frame)
    csv_dump(output_dir / "condition_counts.csv", counts_frame)
    npz_temporary = output_dir / "pseudobulk_arrays.npz.tmp.npz"
    np.savez_compressed(
        npz_temporary,
        conditions=np.asarray(labels, dtype=str),
        genes=np.asarray(genes, dtype=str),
        real_means=real_means,
        pred_means=pred_means,
        real_counts=real_counts,
        pred_counts=pred_counts,
    )
    os.replace(npz_temporary, output_dir / "pseudobulk_arrays.npz")

    metrics_payload = {
        "PDCorr": macro["PDCorr"],
        "PDS_L1": macro["PDS_L1"],
        "PDS_L2": macro["PDS_L2"],
        "PDS_cos": macro["PDS_cos"],
        "MSE": macro["MSE"],
        "MAE": macro["MAE"],
    }
    json_dump(output_dir / "metrics.json", metrics_payload)

    audit = {
        "schema_version": 1,
        "kind": "pbmc_ocoot_table4_perturbation_only_pooled_bulk_evaluation",
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_protocol": {
            "reference": "OCOO-T arXiv:2606.12838v2, Appendix B / Table 4",
            "grouping": "cytokine_only_across_all_cell_types_and_donors",
            "pseudobulk": "cell_count_weighted_arithmetic_mean",
            "perturbations": 62,
            "control": CONTROL,
            "predicted_control": "explicit_copy_of_real_PBS_pseudobulk",
            "mse_mae_space": "perturbed_pseudobulk_expression_not_delta",
            "pdcorr_space": "perturbation_minus_control_effect",
            "pds_space": "perturbation_minus_control_effect",
            "pds_score": "1_minus_zero_based_rank_divided_by_62",
            "pds_target_gene_exclusion": True,
            "pds_ties": "numpy_default_argsort_position_as_in_CellEval_0.6.6",
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "anndata": importlib.metadata.version("anndata"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "cell_eval": importlib.metadata.version("cell-eval"),
            "pdex": importlib.metadata.version("pdex"),
            "chunk_rows": args.chunk_rows,
            "sum_accumulator_dtype": "float64",
            "metric_functions": "direct imports from cell_eval.metrics._anndata",
        },
        "inputs": {
            "root": str(input_root),
            "stages": stage_audits,
            "declared_gene_axis_sha256": next(iter(declared_gene_hashes)),
            "local_gene_list_sha256": sha256_strings(genes),
        },
        "observed": {
            "cell_types": sorted(observed_cell_types),
            "donors": sorted(observed_donors),
            "conditions": labels,
            "treated_cells": treated_rows,
            "control_cells": control_rows,
            "total_cells": int(real_counts.sum()),
            "genes": len(genes),
            "input_pred_vs_real_PBS_pseudobulk_max_abs_diff": pred_input_control_max_abs_diff,
            "scored_pred_vs_real_PBS_pseudobulk_max_abs_diff": float(
                np.max(np.abs(pred_means[control_position] - real_means[control_position]))
            ),
        },
        "metrics": metrics_payload,
        "artifacts": {
            "metrics_json": str((output_dir / "metrics.json").resolve()),
            "metrics_csv": str((output_dir / "metrics.csv").resolve()),
            "per_cytokine_metrics_csv": str((output_dir / "per_cytokine_metrics.csv").resolve()),
            "condition_counts_csv": str((output_dir / "condition_counts.csv").resolve()),
            "pseudobulk_arrays_npz": str((output_dir / "pseudobulk_arrays.npz").resolve()),
        },
    }
    json_dump(output_dir / "audit_manifest.json", audit)
    (output_dir / "PASS").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(metrics_payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
