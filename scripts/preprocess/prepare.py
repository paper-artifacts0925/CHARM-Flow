from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
LOCAL_RESOURCE_ROOT = ROOT / "resources"

PUBLISHED_HASHES = {
    "replogle": {
        "context": "785afb9beac28bd742e0316a2d0accfd9bc7302d1b618bf1026c835baad4fc4a",
        "parent": "00dc083d6fb71abf355a15b1cedd1c61302a90a3c4a58e54df5e305696780420",
        "reservoir": "3bf4d12e8abdb5a91cb5ec753c785299ebe92f26692fe7f9d3680c895afc37b4",
        "pca": "b89b2a84431905c2a1dd6ac1bc4d2ff74d188259f031790687db6513971b4236",
        "de": "c2187a6bee61c7413d4fc5e03d58f5fb2e16d0197230eb48662fdedd4e58d7e9",
        "hurdle": "9465ac75f075f31b3451b3ad71e9e9d8995ee6aa51d90874e0ee2a9cc8edfb5b",
    },
    "pbmc": {
        "context": "dd91bcb3aae06d70a4591ff7c0b4cc1f93038f5fa3b9199bd9515cd6408b0e6d",
        "parent": "c47bb7202b81f76c99125b9d5aeab145b5187fcc5c44393482f9fc0f23cb3d6b",
        "reservoir": "c7273173a5d2a9cfb7664b366f853b4ad431bfb60b9783921cd79c06cd5d88f5",
        "pca": "bf123768469db7d084339f039a268c1b41055f9d65089c354a6a45c02e58045e",
        "de": "741b56e71afc8bd96d9a98203ad05acfbd8f59f66a5c821542b98a817b5ad1c3",
        "hurdle": "b65255f4ba7233c1eb08d132ffc6009225b407c82f8c605f00eaac45635437b6",
    },
    "tahoe": {
        "context": "d87df65bf9a090d39f5c4c99d0e1ae64384002bf11af4f2664aad77aac0e8eab",
        "parent": "d12b79060c60f2c475b0bb1c1f041e28fb9ecc140f9b7cf2644bdad825fbb467",
        "reservoir": "48a5b5fe5706e8d38b60ef380703766fa478d064f5c9997dadcf3dcbb853d9bd",
        "pca": "989aa9bb8488567b653016d07da5cc388e1969eb75359f974715801b6ce61d63",
        "de": "16db24b264ea09db9dc250b16d7840ff2d25bda48615946eaf249fba6cf45d0a",
        "hurdle": "b2de5de30639f3254b7e18375f2d97c0c82914acf128d5771b547113c77738f7",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_resource_root() -> Path:
    configured = os.environ.get("CHARM_RESOURCE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return LOCAL_RESOURCE_ROOT


def portable_path(path: Path) -> str:
    """Use a project-relative path for resources stored inside this bundle."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def paths(dataset: str, data_root: Path, artifact_root: Path, supervision_root: Path) -> dict[str, Path]:
    common = data_root / "finetune_data"
    selected = data_root / "selected_genes"
    parent_root = artifact_root / "parent_residual_gene_dit"
    if dataset == "replogle":
        return {
            "source": common / "nadig_processed_data/replogle.h5ad",
            "selected": selected / "replogle_real_selected_genes.pkl",
            "parent": parent_root / "replogle_parent_residual_v1.npz",
            "parent_metadata": parent_root / "replogle_parent_residual_v1.metadata.json",
            "context_safe": artifact_root / "hungarian_flow/replogle_recursive_sse_child128_min20_v1.npz",
            "context": artifact_root / "hungarian_flow/replogle_recursive_sse_child128_min20_v1.runtime.pkl",
            "reservoir": artifact_root / "real_control_residual_source/replogle_recursive_sse_child128_min20_real64_v1.npz",
            "pca": artifact_root / "match_fm/replogle_control_pca64.npz",
            "de_tables": parent_root / "replogle_official_train_only_de_tables",
            "de": parent_root / "replogle_official_train_only_de_labels_v1.npz",
            "hurdle": parent_root / "replogle_train_hurdle_zero_rates_v1.npz",
        }
    if dataset == "pbmc":
        return {
            "source": common / "pbmc_new/Parse_10M_PBMC_cytokines_processed_Xselected.h5ad",
            "selected": selected / "pbmc_real_selected_genes.pkl",
            "parent": parent_root / "pbmc_donor_celltype_parent_residual_ocoot_v1.npz",
            "parent_metadata": parent_root / "pbmc_donor_celltype_parent_residual_ocoot_v1.metadata.json",
            "context": artifact_root / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_context_v1.npz",
            "reservoir": artifact_root / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_real8_v1.npz",
            "pca": artifact_root / "ocoot/pbmc_control_pca64_div10_v2.npz",
            "de": supervision_root / "pbmc_donor_celltype_train_de_labels_v1.npz",
            "hurdle": supervision_root / "pbmc_donor_celltype_train_hurdle_zero_rates_v1.npz",
            "supervision_manifest": supervision_root / "manifest.json",
        }
    return {
        "source": common / "tahoe100m_full_selected_processed_new",
        "selected": selected / "tahoe100m_real_selected_genes.pkl",
        "parent": parent_root / "tahoe100m_parent_residual_ocoot_v1.npz",
        "parent_metadata": parent_root / "tahoe100m_parent_residual_ocoot_v1.metadata.json",
        "context_safe": artifact_root / "ocoot/tahoe100m_recursive128_minleaf20_context_v2.npz",
        "context": artifact_root / "ocoot/tahoe100m_recursive128_minleaf20_context_v2.runtime.pkl",
        "reservoir": artifact_root / "ocoot/tahoe100m_recursive128_minleaf20_real_control_div10_v2.npz",
        "pca": artifact_root / "ocoot/tahoe100m_control_pca64_div10_v2.npz",
        "de": parent_root / "tahoe100m_train_supervision_v1/tahoe100m_parent_delta_top256_de_labels_v1.npz",
        "hurdle": parent_root / "tahoe100m_train_supervision_v1/tahoe100m_train_hurdle_zero_rates_v1.npz",
        "supervision_manifest": parent_root / "tahoe100m_train_supervision_v1/manifest.json",
    }


def validate_h5ad(path: Path, *, expected_rows: int | None = None) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        if "obsm/X_hvg" in handle:
            matrix = handle["obsm/X_hvg"]
            location = "obsm/X_hvg"
        elif "X" in handle:
            matrix = handle["X"]
            location = "X"
        else:
            raise ValueError(f"{path}: no X or obsm/X_hvg")
        if isinstance(matrix, h5py.Dataset):
            shape = tuple(int(value) for value in matrix.shape)
        else:
            shape = tuple(int(value) for value in matrix.attrs["shape"])
    if len(shape) != 2 or shape[1] != 2000 or shape[0] < 1:
        raise ValueError(f"{path}: invalid model feature shape {shape}")
    if expected_rows is not None and shape[0] != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {shape[0]}")
    return {"path": str(path), "feature": location, "shape": list(shape), "bytes": path.stat().st_size}


def check_inputs(dataset: str, items: dict[str, Path]) -> list[dict]:
    if not items["selected"].is_file():
        raise FileNotFoundError(items["selected"])
    source = items["source"]
    reports = []
    if dataset == "tahoe":
        if not source.is_dir():
            raise FileNotFoundError(source)
        shards = sorted(source.glob("plate*_*.h5ad"))
        if len(shards) != 14:
            raise ValueError(f"Tahoe requires 14 processed H5AD shards, found {len(shards)}")
        reports.extend(validate_h5ad(path) for path in shards)
    else:
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = 643_413 if dataset == "replogle" else 9_697_974
        reports.append(validate_h5ad(source, expected_rows=expected))
    return reports


def required_artifacts(items: dict[str, Path]) -> dict[str, Path]:
    return {key: items[key] for key in ("context", "parent", "reservoir", "pca", "de", "hurdle")}


def read_source_identity(hurdle: Path) -> str:
    import numpy as np

    with np.load(hurdle, allow_pickle=False) as loaded:
        return str(np.asarray(loaded["source_h5ad_sha256"]).reshape(-1)[0])


def create_manifest(
    dataset: str,
    data_root: Path,
    artifact_root: Path,
    supervision_root: Path,
    items: dict[str, Path],
    output: Path,
) -> dict:
    artifact_records = {}
    for name, path in required_artifacts(items).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        artifact_records[name] = {
            "path": portable_path(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "matches_published_bytes": digest == PUBLISHED_HASHES.get(dataset, {}).get(name),
        }
    source_identity = read_source_identity(items["hurdle"])
    overrides = {
        "data.control_distribution_context_path": portable_path(items["context"]),
        "model.parent_residual_artifact_path": portable_path(items["parent"]),
        "model.parent_residual_artifact_sha256": artifact_records["parent"]["sha256"],
        "model.parent_residual_real_control_reservoir_path": portable_path(items["reservoir"]),
        "model.parent_residual_real_control_reservoir_sha256": artifact_records["reservoir"]["sha256"],
        "model.parent_residual_set_pca_path": portable_path(items["pca"]),
        "model.parent_residual_set_pca_sha256": artifact_records["pca"]["sha256"],
        "model.parent_residual_de_aware_artifact_path": portable_path(items["de"]),
        "model.parent_residual_de_aware_artifact_sha256": artifact_records["de"]["sha256"],
        "model.parent_residual_hurdle_artifact_path": portable_path(items["hurdle"]),
        "model.parent_residual_hurdle_artifact_sha256": artifact_records["hurdle"]["sha256"],
        "model.parent_residual_hurdle_source_h5ad_sha256": source_identity,
    }
    if dataset == "pbmc":
        overrides["data.control_distribution_context_sha256"] = artifact_records["context"]["sha256"]
    if dataset == "replogle":
        overrides["model.parent_residual_hurdle_source_h5ad_path"] = portable_path(items["source"])
    payload = {
        "schema_version": "charm_flow.final_preprocessing_manifest.v1",
        "dataset": dataset,
        "data_root": portable_path(data_root),
        "artifact_root": portable_path(artifact_root),
        "supervision_root": portable_path(supervision_root),
        "source_identity_sha256": source_identity,
        "artifacts": artifact_records,
        "config_overrides": overrides,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


class Step:
    def __init__(
        self,
        name: str,
        outputs: Iterable[Path],
        command: Callable[[], list[str]] | list[str],
        *,
        skip_if_any: Iterable[Path] = (),
    ):
        self.name = name
        self.outputs = tuple(outputs)
        self.skip_if_any = tuple(skip_if_any)
        self._command = command

    def command(self) -> list[str]:
        return self._command() if callable(self._command) else list(self._command)


def build_steps(dataset: str, items: dict[str, Path], python: str, de_python: str) -> list[Step]:
    module = [python, "-m"]
    parent_script = ROOT / "scripts/parent_residual_gene_dit"
    if dataset == "replogle":
        config = ROOT / "configs/frozen/replogle_train.yaml"
        tables = items["de_tables"]
        table_files = [tables / f"{line}_train_only_de.csv" for line in ("hepg2", "jurkat", "k562", "rpe1")]
        return [
            Step("parent", [items["parent"], items["parent_metadata"]], [python, str(parent_script / "build_replogle_artifact.py"), "--resolved-config", str(config), "--h5ad", str(items["source"]), "--selected-gene-file", str(items["selected"]), "--output", str(items["parent"]), "--seed", "42"]),
            Step("control-context", [items["context_safe"]], module + ["src.models.hungarian_flow.build_recursive_anchor_bank", "--h5ad", str(items["source"]), "--output", str(items["context_safe"]), "--anchor-capacity", "128", "--min-leaf-size", "20", "--seed", "42"]),
            Step("context-runtime", [items["context"]], lambda: [python, str(parent_script / "export_recursive_context_legacy_pickle.py"), "--context-npz", str(items["context_safe"]), "--expected-context-sha256", sha256(items["context_safe"]) if items["context_safe"].is_file() else "<computed-after-context-build>", "--output", str(items["context"])]),
            Step("real-control-reservoir", [items["reservoir"]], module + ["src.models.real_control_residual_source.build_recursive_reservoir", "--h5ad", str(items["source"]), "--context-bank", str(items["context_safe"]), "--output", str(items["reservoir"]), "--normalization-divisor", "10", "--max-members-per-child", "64", "--seed", "42"]),
            Step("control-pca64", [items["pca"]], [python, str(ROOT / "scripts/preprocess/build_replogle_pca.py"), "--h5ad", str(items["source"]), "--parent-artifact", str(items["parent"]), "--output", str(items["pca"]), "--seed", "42"]),
            Step("train-only-de-tables", table_files + [tables / "manifest.json"], [de_python, str(parent_script / "compute_train_only_de_tables.py"), "--h5ad", str(items["source"]), "--parent-artifact", str(items["parent"]), "--outdir", str(tables)], skip_if_any=[items["de"]]),
            Step("de-labels", [items["de"]], [python, str(parent_script / "build_de_aware_label_artifact.py"), "--parent-artifact", str(items["parent"]), "--de-table", f"hepg2={table_files[0]}", "--de-table", f"jurkat={table_files[1]}", "--de-table", f"k562={table_files[2]}", "--de-table", f"rpe1={table_files[3]}", "--output", str(items["de"])]),
            Step("hurdle", [items["hurdle"]], module + ["src.models.train_hurdle_bank.build_artifact", "--parent-artifact", str(items["parent"]), "--source-h5ad", str(items["source"]), "--output", str(items["hurdle"])]),
        ]

    config = ROOT / f"configs/frozen/{dataset}_train.yaml"
    dataset_name = "pbmc" if dataset == "pbmc" else "tahoe100m"
    expected_shards = "1" if dataset == "pbmc" else "14"
    parent_args = [python, str(parent_script / "build_ocoot_artifact.py"), "--resolved-config", str(config), "--dataset-kind", dataset_name, "--dataset-path", str(items["source"]), "--selected-gene-file", str(items["selected"]), "--expected-shards", expected_shards, "--output", str(items["parent"]), "--seed", "42"]
    context_args = module + ["src.models.hungarian_flow.build_ocoot_recursive_artifacts_v2", "--dataset", dataset_name, "--input", str(items["source"]), "--selected-gene-file", str(items["selected"]), "--feature-key", "X_hvg", "--context-output", str(items.get("context_safe", items["context"])), "--reservoir-output", str(items["reservoir"]), "--anchor-capacity", "128", "--pca-dim", "64", "--seed", "42", "--insufficient-group-policy", "adaptive", "--normalization-divisor", "10"]
    pca_args = module + ["src.models.hungarian_flow.build_ocoot_control_pca64", "--dataset", dataset_name, "--input", str(items["source"]), "--output", str(items["pca"]), "--selected-gene-file", str(items["selected"]), "--feature-key", "X_hvg", "--normalize-counts", "10", "--seed", "42"]
    if dataset == "pbmc":
        parent_args += ["--parent-grouping", "donor_celltype", "--chunk-size", "8192"]
        context_args += ["--control-grouping", "donor_celltype", "--target-child-occupancy", "64", "--minimum-support", "20", "--max-members-per-child", "8"]
        supervision = [python, str(ROOT / "scripts/internal/build_pbmc_donor_celltype_train_supervision.py"), "--parent", str(items["parent"]), "--parent-metadata", str(items["parent_metadata"]), "--source-h5ad", str(items["source"]), "--de-output", str(items["de"]), "--hurdle-output", str(items["hurdle"]), "--manifest", str(items["supervision_manifest"]), "--mode", "all"]
    else:
        parent_args += ["--chunk-size", "4096"]
        context_args += ["--max-members-per-child", "64"]
        supervision = [python, str(parent_script / "build_tahoe_train_supervision.py"), "--parent", str(items["parent"]), "--source-dir", str(items["source"]), "--de-output", str(items["de"]), "--hurdle-output", str(items["hurdle"]), "--manifest", str(items["supervision_manifest"]), "--mode", "all"]
    result = [
        Step("parent", [items["parent"], items["parent_metadata"]], parent_args),
        Step("control-context-and-reservoir", [items.get("context_safe", items["context"]), items["reservoir"]], context_args),
    ]
    if dataset == "tahoe":
        result.append(Step("context-runtime", [items["context"]], lambda: [python, str(parent_script / "export_recursive_context_legacy_pickle.py"), "--context-npz", str(items["context_safe"]), "--expected-context-sha256", sha256(items["context_safe"]) if items["context_safe"].is_file() else "<computed-after-context-build>", "--output", str(items["context"])]))
    result.extend([
        Step("control-pca64", [items["pca"]], pca_args),
        Step("train-only-supervision", [items["de"], items["hurdle"], items["supervision_manifest"]], supervision),
    ])
    return result


def run_steps(steps: list[Step], *, execute: bool) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for step in steps:
        if not execute:
            command = step.command()
            print(f"STEP {step.name}\n  {shlex.join(command)}", flush=True)
            continue
        if any(path.exists() for path in step.skip_if_any):
            print(f"SKIP {step.name}: downstream artifact already exists")
            continue
        existing = [path.exists() for path in step.outputs]
        if all(existing):
            print(f"SKIP {step.name}: outputs already exist")
            continue
        if any(existing):
            present = [str(path) for path, found in zip(step.outputs, existing) if found]
            raise RuntimeError(f"{step.name}: partial outputs exist; move them aside first: {present}")
        command = step.command()
        print(f"STEP {step.name}\n  {shlex.join(command)}", flush=True)
        if execute:
            for output in step.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("replogle", "pbmc", "tahoe", "all"))
    parser.add_argument("action", choices=("plan", "check-inputs", "check", "build", "manifest"), nargs="?", default="check")
    parser.add_argument("--resource-root", type=Path, default=default_resource_root())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--pbmc-supervision-root", type=Path)
    parser.add_argument("--python", default=os.environ.get("CHARM_TRAIN_PYTHON", sys.executable))
    parser.add_argument("--de-python", default=os.environ.get("CHARM_DE_PYTHON", os.environ.get("CHARM_EVAL_PYTHON", sys.executable)))
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    resource = args.resource_root.expanduser().resolve()
    data_root = (args.data_root or resource / "data/PerturbDiff_data/perturb_data").expanduser().resolve()
    artifact_root = (args.artifact_root or resource / "artifacts").expanduser().resolve()
    datasets = ("replogle", "pbmc", "tahoe") if args.dataset == "all" else (args.dataset,)
    results = {}
    for dataset in datasets:
        configured_supervision = args.pbmc_supervision_root or os.environ.get(
            "CHARM_PBMC_SUPERVISION_ROOT"
        )
        if dataset == "pbmc" and configured_supervision:
            supervision_root = Path(configured_supervision).expanduser().resolve()
        else:
            supervision_root = artifact_root / "parent_residual_gene_dit/pbmc_donor_celltype_train_supervision"
        item_map = paths(dataset, data_root, artifact_root, supervision_root)
        print(f"DATASET {dataset}")
        if args.action != "plan":
            input_report = check_inputs(dataset, item_map)
            print(json.dumps({"inputs": input_report}, sort_keys=True))
        if args.action == "check-inputs":
            continue
        steps = build_steps(dataset, item_map, args.python, args.de_python)
        if args.action in {"plan", "build"}:
            run_steps(steps, execute=args.action == "build")
            if args.action == "plan":
                continue
        manifest_path = args.manifest_output if len(datasets) == 1 and args.manifest_output else artifact_root / "preprocessing_manifests" / f"{dataset}.json"
        payload = create_manifest(dataset, data_root, artifact_root, supervision_root, item_map, manifest_path)
        results[dataset] = {"manifest": str(manifest_path), "artifacts": payload["artifacts"]}
        print(f"READY dataset={dataset} manifest={manifest_path}")
    if results:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
