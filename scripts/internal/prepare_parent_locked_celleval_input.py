from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import anndata
import numpy as np

from scripts.parent_residual_gene_dit.postprocess_parent_locked_celleval_bound import (
    ALGORITHM,
    _dense,
    postprocess,
)
from scripts.parent_residual_gene_dit.verify_parent_locked_celleval_bound import (
    verify,
)
from src.models.parent_locked_residual.saved_audit import (
    GROUP_OBS_KEY,
    audit_saved_parent_locked_h5ad,
)


IDENTITY_ALGORITHM = "identity_strict_celleval_range"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_source_range(
    path: str | Path,
    *,
    threshold: float,
    chunk_size: int = 2048,
) -> dict:
    """Reopen a Parent-locked source and classify strict range violations."""

    path = Path(path).expanduser().resolve()
    threshold = float(threshold)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Cell-Eval threshold must be finite and positive")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    adata = anndata.read_h5ad(path, backed="r")
    try:
        if GROUP_OBS_KEY not in adata.obs:
            raise AssertionError("input has no Parent projection group identities")
        groups = np.asarray(adata.obs[GROUP_OBS_KEY], dtype=np.int64)
        if groups.shape != (adata.n_obs,):
            raise AssertionError("Parent projection group identities are malformed")

        minimum = math.inf
        maximum = -math.inf
        nonfinite = 0
        negative = 0
        at_or_above_threshold = 0
        generated_at_or_above_threshold = 0
        control_at_or_above_threshold = 0
        for start in range(0, adata.n_obs, int(chunk_size)):
            stop = min(start + int(chunk_size), adata.n_obs)
            block = _dense(adata.X[start:stop])
            finite = np.isfinite(block)
            nonfinite += int((~finite).sum())
            if finite.any():
                minimum = min(minimum, float(block[finite].min()))
                maximum = max(maximum, float(block[finite].max()))
            negative += int((block < 0).sum())
            violations = block >= threshold
            at_or_above_threshold += int(violations.sum())
            generated_rows = groups[start:stop] >= 0
            generated_at_or_above_threshold += int(
                violations[generated_rows].sum()
            )
            control_at_or_above_threshold += int(
                violations[~generated_rows].sum()
            )
        if nonfinite or negative:
            raise AssertionError(
                "Parent-locked source has invalid values: "
                f"nonfinite={nonfinite}, negative={negative}"
            )
        return {
            "source_h5ad": str(path),
            "cell_eval_threshold": threshold,
            "minimum_expression": float(minimum),
            "maximum_expression": float(maximum),
            "nonfinite_values": int(nonfinite),
            "negative_values": int(negative),
            "values_at_or_above_threshold": int(at_or_above_threshold),
            "generated_values_at_or_above_threshold": int(
                generated_at_or_above_threshold
            ),
            "control_values_at_or_above_threshold": int(
                control_at_or_above_threshold
            ),
            "strict_celleval_range_passed": at_or_above_threshold == 0,
        }
    finally:
        adata.file.close()


def prepare(
    *,
    input_path: Path,
    output_path: Path,
    cell_eval_threshold: float,
    max_singleton_fraction: float,
    max_endpoint_mean_drift: float,
) -> dict:
    """Return a deterministic, independently audited evaluation input record."""

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("bounded output must differ from the immutable source")
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise FileNotFoundError(input_path)
    threshold = float(cell_eval_threshold)
    source_sha256 = _sha256(input_path)
    source_parent_audit = audit_saved_parent_locked_h5ad(
        input_path,
        max_singleton_fraction=max_singleton_fraction,
        max_endpoint_mean_drift=max_endpoint_mean_drift,
    )
    source_range_audit = audit_source_range(input_path, threshold=threshold)
    if source_range_audit["control_values_at_or_above_threshold"]:
        raise ValueError(
            "control rows violate Cell-Eval's threshold; refusing to alter controls"
        )

    violations = int(source_range_audit["values_at_or_above_threshold"])
    sidecar = Path(str(output_path) + ".projection.json")
    if violations == 0:
        if output_path.exists() or sidecar.exists():
            raise FileExistsError(
                "identity input unexpectedly has a bounded artifact; refusing an "
                f"ambiguous route: {output_path}"
            )
        if _sha256(input_path) != source_sha256:
            raise AssertionError("source H5AD SHA changed during identity audit")
        return {
            "schema_version": 1,
            "passed": True,
            "kind": "identity",
            "algorithm": IDENTITY_ALGORITHM,
            "source_h5ad": str(input_path),
            "source_sha256": source_sha256,
            "evaluation_h5ad": str(input_path),
            "evaluation_sha256": source_sha256,
            "source_overwritten": False,
            "projection_sidecar": None,
            "projection_sidecar_sha256": None,
            "cell_eval_threshold": threshold,
            "source_range_audit": source_range_audit,
            "evaluation_range_audit": source_range_audit,
            "source_parent_audit": source_parent_audit,
            "evaluation_parent_audit": source_parent_audit,
        }

    if not source_range_audit["generated_values_at_or_above_threshold"]:
        raise AssertionError("threshold violations were not assigned to generated rows")
    output_exists = output_path.exists()
    sidecar_exists = sidecar.exists()
    if output_exists != sidecar_exists:
        raise FileExistsError(
            "partial bounded artifact set requires manual quarantine: "
            f"output={output_exists}, sidecar={sidecar_exists}"
        )
    if not output_exists:
        postprocess(
            input_path=input_path,
            output_path=output_path,
            cell_eval_threshold=threshold,
            max_singleton_fraction=max_singleton_fraction,
            max_endpoint_mean_drift=max_endpoint_mean_drift,
        )

    verified = verify(
        input_path=input_path,
        output_path=output_path,
        max_singleton_fraction=max_singleton_fraction,
        max_endpoint_mean_drift=max_endpoint_mean_drift,
    )
    if verified.get("algorithm") != ALGORITHM:
        raise AssertionError("bounded evaluation input used an unexpected algorithm")
    if _sha256(input_path) != source_sha256:
        raise AssertionError("source H5AD SHA changed during bounded preparation")
    if verified.get("input_sha256") != source_sha256:
        raise AssertionError("bounded verification source SHA disagrees")
    return {
        "schema_version": 1,
        "passed": True,
        "kind": "bounded_projection",
        "algorithm": ALGORITHM,
        "source_h5ad": str(input_path),
        "source_sha256": source_sha256,
        "evaluation_h5ad": str(output_path),
        "evaluation_sha256": str(verified["output_sha256"]),
        "source_overwritten": False,
        "projection_sidecar": str(sidecar),
        "projection_sidecar_sha256": _sha256(sidecar),
        "cell_eval_threshold": threshold,
        "source_range_audit": source_range_audit,
        "evaluation_range_audit": verified["bounded_range_audit"],
        "source_parent_audit": verified["source_parent_audit"],
        "evaluation_parent_audit": verified["bounded_parent_audit"],
        "changed_values": int(verified["changed_values_verified"]),
        "changes_outside_declared_pairs": int(
            verified["changes_outside_declared_pairs"]
        ),
        "changed_controls": int(verified["changed_controls"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-eval-threshold", type=float, default=15.0)
    parser.add_argument("--max-singleton-fraction", type=float, default=0.01)
    parser.add_argument("--max-endpoint-mean-drift", type=float, default=1.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                input_path=args.input,
                output_path=args.output,
                cell_eval_threshold=args.cell_eval_threshold,
                max_singleton_fraction=args.max_singleton_fraction,
                max_endpoint_mean_drift=args.max_endpoint_mean_drift,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
