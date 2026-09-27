import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


STAGE_OUTPUTS = {
    "bulk": "pooled_bulk",
    "builder": "pooled_inputs",
    "de": "pooled_de",
    "celleval": "pooled_celleval",
    "celltype": "evaluation",
    "summary": "summary",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def records(paths) -> dict:
    return {
        str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted({Path(value).resolve(strict=True) for value in paths})
    }


def tree_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix != ".lock")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".contract.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def transition(record_path: Path, *, action: str, contract: dict, output: Path) -> dict:
    """Only sealed outputs may be reused; interrupted nonempty trees fail closed."""
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    if previous is not None:
        if previous.get("contract") != contract:
            raise RuntimeError(f"postprocess input/code/parameter contract changed: {record_path}")
        if previous.get("phase") not in {"started", "complete"}:
            raise RuntimeError(f"invalid postprocess contract phase: {record_path}")
        if previous.get("phase") == "complete":
            actual = records(tree_files(output))
            if actual != previous.get("outputs"):
                raise RuntimeError(f"postprocess output changed or missing: {output}")
            return previous
        if action == "start" and output.exists() and any(output.iterdir()):
            raise RuntimeError(f"refusing unsealed interrupted postprocess output: {output}")
    elif action != "start":
        raise RuntimeError(f"cannot seal an unstarted postprocess stage: {record_path}")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing unbound existing postprocess output: {output}")

    value = {"schema_version": 1, "contract": contract, "phase": "started"}
    if action == "complete":
        files = tree_files(output)
        if not files:
            raise RuntimeError(f"cannot seal empty postprocess output: {output}")
        value.update(phase="complete", outputs=records(files))
    atomic_json(record_path, value)
    return value


def stage_contract(run_root: Path, stage: str, threads: int) -> dict:
    merge_path = run_root / "by_cell_type/merge_manifest.json"
    freeze_path = run_root / "logs/audit/postprocess_protocol_freeze_v1.json"
    freeze = json.loads(freeze_path.read_text())
    # The freeze includes launcher + adapter, therefore literal command flags,
    # statistical parameters and implementation changes alter this contract.
    frozen_assets = records(Path(path) for path in freeze["assets"])
    if frozen_assets != freeze["assets"]:
        raise RuntimeError("postprocess frozen dependency changed before stage execution")
    merge = json.loads(merge_path.read_text())
    inputs = [merge_path, freeze_path, Path(merge["protocol"])]
    if stage in {"bulk", "builder", "celltype"}:
        for item in merge["stages"]:
            inputs.extend([Path(item["pred"]), Path(item["real"]), Path(item["pred"]).parent / "stage_manifest.json"])
    if stage in {"de", "celleval"}:
        inputs.extend(run_root / "pooled_inputs" / name for name in (
            "builder_manifest.json", "physical_identity_provenance.json",
            "source_row_mapping.csv", "real_pooled.h5ad", "pred_pooled.h5ad",
        ))
    if stage == "de":
        inputs.extend(Path(item["pred"]).parent / "stage_manifest.json" for item in merge["stages"])
    if stage == "celleval":
        inputs.extend(run_root / "pooled_de" / name for name in (
            "de_cache_manifest.json", "input_audit.json", "real_de.csv", "pred_de.csv",
        ))
    if stage == "summary":
        inputs.extend(tree_files(run_root / "evaluation"))
    input_records = records(inputs)
    value = {
        "stage": stage,
        "run_root": str(run_root),
        "inputs": input_records,
        "implementations": frozen_assets,
        "parameters": {"sampling_seed": 42, "eval_threads": threads},
    }
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "complete"))
    parser.add_argument("stage", choices=tuple(STAGE_OUTPUTS))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    result = transition(
        root / "logs/audit" / f"postprocess_{args.stage}_contract_v1.json",
        action=args.action,
        contract=stage_contract(root, args.stage, args.threads),
        output=root / STAGE_OUTPUTS[args.stage],
    )
    print(f"postprocess_contract stage={args.stage} phase={result['phase']}")


if __name__ == "__main__":
    main()
