import argparse
import json
from pathlib import Path
from typing import Any

import yaml


PORTABLE_DATA_ROOT = Path("resources/data/PerturbDiff_data/perturb_data")
PORTABLE_ARTIFACT_ROOT = Path("resources/artifacts")
PORTABLE_PBMC_SUPERVISION_ROOT = Path(
    "resources/artifacts/parent_residual_gene_dit/"
    "pbmc_donor_celltype_train_supervision"
)


def _replace_prefix(value: str, old: Path, new: Path) -> str:
    old_text = str(old)
    if value == old_text:
        return str(new)
    prefix = old_text.rstrip("/") + "/"
    if value.startswith(prefix):
        return str(new / value[len(prefix) :])
    return value


def rebase(value: Any, replacements: tuple[tuple[Path, Path], ...]) -> Any:
    if isinstance(value, dict):
        return {key: rebase(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            updated = _replace_prefix(value, old, new)
            if updated != value:
                return updated
    return value


def apply_manifest(payload: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Apply path/hash overrides emitted by the final preprocessing pipeline."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "charm_flow.final_preprocessing_manifest.v1":
        raise ValueError(f"unsupported preprocessing manifest: {manifest_path}")
    overrides = manifest.get("config_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("preprocessing manifest has no config_overrides mapping")
    for dotted_key, value in overrides.items():
        keys = str(dotted_key).split(".")
        target: Any = payload
        for key in keys[:-1]:
            if not isinstance(target, dict) or key not in target:
                raise KeyError(f"manifest override does not exist in config: {dotted_key}")
            target = target[key]
        leaf = keys[-1]
        if not isinstance(target, dict) or leaf not in target:
            raise KeyError(f"manifest override does not exist in config: {dotted_key}")
        target[leaf] = value
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pbmc-supervision-root", type=Path, required=True)
    parser.add_argument("--preprocessing-manifest", type=Path)
    args = parser.parse_args()

    payload = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    replacements = (
        (PORTABLE_DATA_ROOT, args.data_root.expanduser().resolve()),
        (PORTABLE_ARTIFACT_ROOT, args.artifact_root.expanduser().resolve()),
        (
            PORTABLE_PBMC_SUPERVISION_ROOT,
            args.pbmc_supervision_root.expanduser().resolve(),
        ),
    )
    rebased = rebase(payload, replacements)
    if args.preprocessing_manifest is not None:
        rebased = apply_manifest(rebased, args.preprocessing_manifest.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(rebased, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
