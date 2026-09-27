import argparse
import hashlib
import json
from pathlib import Path

import anndata
import numpy as np

from scripts.parent_residual_gene_dit.postprocess_parent_locked_celleval_bound import (
    POSTPROCESS_KEY,
    _dense,
    audit_celleval_bounded_h5ad,
)
from src.models.parent_locked_residual.saved_audit import (
    GROUP_OBS_KEY,
    REFERENCE_KEY,
    audit_saved_parent_locked_h5ad,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_pair_set(record) -> set[tuple[int, int]]:
    groups = np.asarray(record["affected_projection_group_ids"], dtype=np.int64)
    genes = np.asarray(record["affected_gene_indices"], dtype=np.int64)
    if groups.ndim != 1 or genes.ndim != 1 or len(groups) != len(genes):
        raise AssertionError("serialized affected group/gene pairs are malformed")
    pairs = {(int(group), int(gene)) for group, gene in zip(groups, genes)}
    if len(pairs) != len(groups):
        raise AssertionError("serialized affected group/gene pairs are duplicated")
    return pairs


def verify(
    *,
    input_path: Path,
    output_path: Path,
    max_singleton_fraction: float,
    max_endpoint_mean_drift: float,
) -> dict:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    sidecar = Path(str(output_path) + ".projection.json")
    if input_path == output_path:
        raise ValueError("bounded output must differ from its immutable source")
    if not input_path.is_file() or not output_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("input, bounded output, and projection sidecar are required")
    sidecar_record = json.loads(sidecar.read_text())
    if Path(sidecar_record["input_h5ad"]).resolve() != input_path:
        raise AssertionError("projection sidecar input path does not match")
    if Path(sidecar_record["output_h5ad"]).resolve() != output_path:
        raise AssertionError("projection sidecar output path does not match")
    input_sha256 = _sha256(input_path)
    output_sha256 = _sha256(output_path)
    if sidecar_record.get("input_sha256") != input_sha256:
        raise AssertionError("source H5AD SHA changed after bounded projection")
    if sidecar_record.get("output_sha256") != output_sha256:
        raise AssertionError("bounded H5AD SHA disagrees with its sidecar")
    if bool(sidecar_record.get("source_overwritten", True)):
        raise AssertionError("bounded provenance does not prove no-overwrite")

    source = anndata.read_h5ad(input_path, backed="r")
    bounded = anndata.read_h5ad(output_path, backed="r")
    try:
        if source.shape != bounded.shape:
            raise AssertionError("bounded H5AD shape changed")
        if not np.array_equal(source.obs_names, bounded.obs_names):
            raise AssertionError("bounded H5AD observation identities changed")
        if not np.array_equal(source.var_names, bounded.var_names):
            raise AssertionError("bounded H5AD gene identities changed")
        if POSTPROCESS_KEY not in bounded.uns:
            raise AssertionError("bounded H5AD has no serialized provenance")
        postprocess_record = bounded.uns[POSTPROCESS_KEY]
        pairs = _load_pair_set(postprocess_record)
        if int(postprocess_record["affected_group_gene_pairs"]) != len(pairs):
            raise AssertionError("serialized affected-pair count disagrees")
        source_groups = np.asarray(source.obs[GROUP_OBS_KEY], dtype=np.int64)
        bounded_groups = np.asarray(bounded.obs[GROUP_OBS_KEY], dtype=np.int64)
        if not np.array_equal(source_groups, bounded_groups):
            raise AssertionError("bounded H5AD Parent group identities changed")
        source_reference = np.asarray(
            source.uns[REFERENCE_KEY]["effective_parent_mean"]
        )
        bounded_reference = np.asarray(
            bounded.uns[REFERENCE_KEY]["effective_parent_mean"]
        )
        if not np.array_equal(source_reference, bounded_reference):
            raise AssertionError("bounded H5AD effective Parent reference changed")

        changed_values = 0
        changed_controls = 0
        changes_outside_declared_pairs = 0
        for start in range(0, source.n_obs, 2048):
            stop = min(start + 2048, source.n_obs)
            source_block = _dense(source.X[start:stop])
            bounded_block = _dense(bounded.X[start:stop])
            changed = source_block != bounded_block
            changed_values += int(changed.sum())
            local_groups = source_groups[start:stop]
            changed_controls += int(changed[local_groups < 0].sum())
            for local_row, gene_index in np.argwhere(changed):
                pair = (int(local_groups[int(local_row)]), int(gene_index))
                if pair not in pairs:
                    changes_outside_declared_pairs += 1
        if changed_controls:
            raise AssertionError("bounded H5AD changed immutable controls")
        if changes_outside_declared_pairs:
            raise AssertionError("bounded H5AD changed undeclared group/gene pairs")
        if changed_values != int(postprocess_record["changed_values"]):
            raise AssertionError("serialized changed-value count disagrees")
    finally:
        source.file.close()
        bounded.file.close()

    threshold = float(sidecar_record["cell_eval_threshold"])
    upper = float(sidecar_record["effective_upper_bound"])
    source_parent_audit = audit_saved_parent_locked_h5ad(
        input_path,
        max_singleton_fraction=max_singleton_fraction,
        max_endpoint_mean_drift=max_endpoint_mean_drift,
    )
    bounded_parent_audit = audit_saved_parent_locked_h5ad(
        output_path,
        max_singleton_fraction=max_singleton_fraction,
        max_endpoint_mean_drift=max_endpoint_mean_drift,
    )
    range_audit = audit_celleval_bounded_h5ad(
        output_path,
        threshold=threshold,
        expected_upper_bound=upper,
    )
    return {
        **sidecar_record,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "changed_values_verified": int(changed_values),
        "changed_controls": int(changed_controls),
        "changes_outside_declared_pairs": int(changes_outside_declared_pairs),
        "source_parent_audit": source_parent_audit,
        "bounded_parent_audit": bounded_parent_audit,
        "bounded_range_audit": range_audit,
        "sidecar": str(sidecar),
        "verified_existing": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-singleton-fraction", type=float, default=0.01)
    parser.add_argument("--max-endpoint-mean-drift", type=float, default=1.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                input_path=args.input,
                output_path=args.output,
                max_singleton_fraction=args.max_singleton_fraction,
                max_endpoint_mean_drift=args.max_endpoint_mean_drift,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
