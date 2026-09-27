import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from sklearn.decomposition import PCA


def _decode(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.asarray([value.decode() for value in array], dtype=str)
    return array.astype(str)


def _categorical(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Dataset):
        return _decode(node[:])
    categories = _decode(node["categories"][:])
    codes = np.asarray(node["codes"][:], dtype=np.int64)
    if (codes < 0).any() or (codes >= len(categories)).any():
        raise ValueError(f"invalid categorical codes in obs/{key}")
    return categories[codes]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-key", default="X_hvg")
    parser.add_argument("--perturbation-key", default="gene")
    parser.add_argument("--control-label", default="non-targeting")
    parser.add_argument("--normalize-counts", type=float, default=10.0)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.h5ad.expanduser().resolve()
    parent_path = args.parent_artifact.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file() or not parent_path.is_file():
        raise FileNotFoundError("Replogle H5AD or Parent artifact is absent")
    if output.exists():
        raise FileExistsError(output)
    divisor = float(args.normalize_counts)
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("normalize-counts must be finite and positive")

    with np.load(parent_path, allow_pickle=False) as parent:
        genes = np.asarray(parent["gene_names"]).astype(str)
    with h5py.File(source, "r") as handle:
        perturbations = _categorical(handle["obs"], args.perturbation_key)
        rows = np.flatnonzero(perturbations == args.control_label).astype(np.int64)
        key = args.feature_key.strip("/")
        if key in handle:
            matrix = handle[key]
        elif "obsm" in handle and key in handle["obsm"]:
            matrix = handle["obsm"][key]
        else:
            raise KeyError(f"source H5AD has no {args.feature_key!r}")
        if not isinstance(matrix, h5py.Dataset):
            raise TypeError("Replogle PCA requires a dense HDF5 feature dataset")
        if matrix.shape[1] != len(genes):
            raise ValueError("feature width and Parent gene order differ")
        controls = np.asarray(matrix[rows], dtype=np.float32)

    controls /= np.float32(divisor)
    fitted = PCA(
        n_components=int(args.components),
        svd_solver="randomized",
        copy=False,
        random_state=int(args.seed),
    ).fit(controls)
    arrays = {
        "components": fitted.components_.astype(np.float32),
        "mean": fitted.mean_.astype(np.float32),
        "explained_variance": fitted.explained_variance_.astype(np.float32),
        "explained_variance_ratio": fitted.explained_variance_ratio_.astype(np.float32),
        "genes": genes,
        "n_control": np.asarray([len(controls)], dtype=np.int64),
        "normalize_counts": np.asarray([divisor], dtype=np.float32),
    }
    if any(np.asarray(value).dtype == object for value in arrays.values()):
        raise TypeError("PCA output contains an unsafe object array")
    _atomic_savez(output, **arrays)
    report = {
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "components": int(args.components),
        "control_label": args.control_label,
        "control_only": True,
        "feature_key": args.feature_key,
        "n_control": int(len(controls)),
        "normalize_counts": divisor,
        "seed": int(args.seed),
        "source_h5ad": str(source),
        "treated_expression_used": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
