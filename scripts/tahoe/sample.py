from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import omegaconf


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(os.environ.get("CHARM_RESOURCE_ROOT", str(ROOT / "resources")))
DATA_ROOT = Path(os.environ.get("CHARM_DATA_ROOT", str(RESOURCE_ROOT / "data/PerturbDiff_data/perturb_data")))
ARTIFACT_ROOT = Path(os.environ.get("CHARM_ARTIFACT_ROOT", str(RESOURCE_ROOT / "artifacts")))
UPSTREAM_PATH = (
    ROOT
    / "scripts/tahoe/full_test.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_tahoe_ctxw256_full_trainonly_hs", UPSTREAM_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import Tahoe ctxw256 full evaluator: {UPSTREAM_PATH}")
upstream = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upstream)
base = upstream.base

CURRENT_EXPORT = ROOT / "weights/tahoe.pt"
CURRENT_EXPORT_SHA256 = (
    "2c31ef6c936d7447252082406e2e0b64c3fafc91a2f059271d041f2927a2aee4"
)
DE_AWARE_ARTIFACT = (
    ARTIFACT_ROOT
    / "parent_residual_gene_dit/tahoe100m_train_supervision_v1"
    / "tahoe100m_parent_delta_top256_de_labels_v1.npz"
)
DE_AWARE_ARTIFACT_SHA256 = (
    "16db24b264ea09db9dc250b16d7840ff2d25bda48615946eaf249fba6cf45d0a"
)
HURDLE_ARTIFACT = (
    ARTIFACT_ROOT
    / "parent_residual_gene_dit/tahoe100m_train_supervision_v1"
    / "tahoe100m_train_hurdle_zero_rates_v1.npz"
)
HURDLE_ARTIFACT_SHA256 = (
    "b2de5de30639f3254b7e18375f2d97c0c82914acf128d5771b547113c77738f7"
)
HOLDOUT_LABEL_AUDIT: dict[str, dict[str, int]] | None = None

# The upstream evaluator is deliberately fail-closed.  Re-pin its original
# audit module before extending that audit with the H/S-specific contract.
base.FINAL_EXPORT = CURRENT_EXPORT
base.FINAL_EXPORT_SHA256 = CURRENT_EXPORT_SHA256
base.CONTROL = CURRENT_EXPORT
base.CELL_LINE_MAP = CURRENT_EXPORT
_upstream_audit_cfg = base.audit_cfg
_upstream_select_screen_groups = base.select_screen_groups


def _same_artifact(model_cfg: dict[str, Any], key: str, path: Path, digest: str) -> None:
    base.require(base.same_path(model_cfg.get(key), path), f"checkpoint {key} changed")
    base.require(
        str(model_cfg.get(key.replace("_path", "_sha256"))) == digest,
        f"checkpoint {key} SHA changed",
    )
    base.require(path.is_file() and base.sha256(path) == digest, f"artifact bytes changed: {path}")


def audit_cfg(cfg: omegaconf.DictConfig) -> dict[str, Any]:
    """Retain the 735-group protocol and assert the target-free H/S contract."""

    audit = _upstream_audit_cfg(cfg)
    base.allow_omegaconf_checkpoint_unpickling()
    checkpoint = base.torch.load(
        CURRENT_EXPORT, map_location="cpu", weights_only=True, mmap=True
    )
    model_cfg = checkpoint["hyper_parameters"]["model_cfg"]

    for key in (
        "parent_residual_de_aware_enabled",
        "parent_residual_hurdle_geometry_enabled",
        "parent_residual_sparse_support_enabled",
        "parent_residual_sparse_joint_context_enabled",
        "parent_residual_soft_gated_regression_enabled",
        "parent_residual_soft_gate_detach_regression",
        "parent_residual_gene_relation_enabled",
    ):
        base.require(bool(model_cfg.get(key)), f"checkpoint disabled required H/S field {key}")
    base.require(
        float(model_cfg.get("parent_residual_hurdle_alpha", -1.0)) == 1.0,
        "terminal hurdle alpha changed",
    )
    base.require(
        float(model_cfg.get("parent_residual_soft_gate_blend", -1.0)) == 0.1,
        "soft-gate blend changed",
    )
    base.require(
        int(model_cfg.get("parent_residual_gene_relation_rank", -1)) == 64,
        "gene-relation rank changed",
    )
    base.require(
        model_cfg.get("parent_residual_hurdle_source_h5ad_path") is None,
        "inference must not read a treated H5AD target",
    )
    _same_artifact(
        model_cfg,
        "parent_residual_de_aware_artifact_path",
        DE_AWARE_ARTIFACT,
        DE_AWARE_ARTIFACT_SHA256,
    )
    _same_artifact(
        model_cfg,
        "parent_residual_hurdle_artifact_path",
        HURDLE_ARTIFACT,
        HURDLE_ARTIFACT_SHA256,
    )
    audit.update(
        {
            "entrypoint": "sample.py",
            "model_variant": "ctxw256_trainonly_deaware_sparse_hurdle_14k",
            "terminal_hurdle_alpha": 1.0,
            "soft_gate_blend": 0.1,
            "supervision_scope": "training_split_only",
            "inference_target_access": False,
            "holdout_label_multiplicity": HOLDOUT_LABEL_AUDIT,
        }
    )
    return audit


base.audit_cfg = audit_cfg


def audit_holdout_labels(cfg: omegaconf.DictConfig) -> None:
    """Fail closed on the known Tahoe holdout-list multiplicity."""

    global HOLDOUT_LABEL_AUDIT
    report: dict[str, dict[str, int]] = {}
    for split in ("validation", "test"):
        original = base.strings(cfg.data.holdout_pert[split])
        unique = list(dict.fromkeys(original))
        report[split] = {
            "input_labels": len(original),
            "unique_labels": len(unique),
            "duplicate_entries": len(original) - len(unique),
        }
    base.require(
        report["validation"]
        == {
            "input_labels": 54,
            "unique_labels": 54,
            "duplicate_entries": 0,
        },
        "unexpected Tahoe validation holdout-label multiplicity",
    )
    base.require(
        report["test"]
        == {
            "input_labels": 777,
            "unique_labels": 735,
            "duplicate_entries": 42,
        },
        "unexpected Tahoe test holdout-label multiplicity",
    )
    HOLDOUT_LABEL_AUDIT = report


def select_screen_groups(datamodule: Any, cfg: omegaconf.DictConfig) -> dict[str, Any]:
    """Select the unchanged strata, then re-sample unique raw identities."""

    dataset = datamodule.test_dataset
    original_groups = {
        ds_name: dict(groups)
        for ds_name, groups in dataset.grouped_pert_data_indices.items()
    }
    selection = _upstream_select_screen_groups(datamodule, cfg)
    cells_per_group = int(cfg.sampling.tahoe_cells_per_group)
    seed = int(cfg.optimization.seed)
    removed_positions = 0
    affected_source_groups = 0
    minimum_unique_available: int | None = None
    for row in selection["groups"]:
        ds_name = str(row["dataset"])
        group_key = tuple(map(int, row["key_codes"]))
        perturb_split = np.asarray(
            dataset.data_indices[ds_name]["perturb"], dtype=np.int64
        )
        positions = np.asarray(
            original_groups[ds_name][group_key], dtype=np.int64
        )
        base.require(
            bool(
                positions.size
                and positions.min() >= 0
                and positions.max() < perturb_split.size
            ),
            f"selected source positions are invalid for {ds_name}/{group_key}",
        )
        raw_indices = perturb_split[positions]
        first = np.unique(raw_indices, return_index=True)[1]
        unique_positions = positions[np.sort(first)]
        unique_available = int(unique_positions.size)
        base.require(
            unique_available >= cells_per_group,
            f"only {unique_available} unique raw cells for {ds_name}/{group_key}",
        )
        removed = int(positions.size - unique_available)
        removed_positions += removed
        affected_source_groups += int(removed > 0)
        minimum_unique_available = (
            unique_available
            if minimum_unique_available is None
            else min(minimum_unique_available, unique_available)
        )
        rng = np.random.default_rng(
            base.hash_rank(seed, ds_name, group_key) % (2**32)
        )
        sampled = np.sort(
            rng.choice(unique_positions, size=cells_per_group, replace=False)
        )
        dataset.grouped_pert_data_indices[ds_name][group_key] = sampled
        dataset.grouped_pert_num_cell[ds_name][group_key] = cells_per_group
        row["available_cells"] = unique_available
        row["sampled_indices_sha256"] = base.hashlib.sha256(
            sampled.tobytes()
        ).hexdigest()

    base.require(affected_source_groups > 0, "Tahoe raw-cell repair did not engage")
    selection["source_group_identity_deduplication"] = {
        "passed": True,
        "method": "stable-first-occurrence-by-raw-cell-index",
        "selected_source_groups": int(selection["selected_groups"]),
        "affected_source_groups": affected_source_groups,
        "removed_duplicate_positions": removed_positions,
        "minimum_unique_available": minimum_unique_available,
        "uses_expression_values": False,
    }
    dataset_ids = getattr(dataset, "datasets_to_num", None)
    if dataset_ids is None:
        dataset_ids = {
            name: offset
            for offset, name in enumerate(dataset.dataset_path_map.keys())
        }

    selected_keys: list[np.ndarray] = []
    dataset_counts: dict[str, int] = {}
    for ds_name, groups in dataset.grouped_pert_data_indices.items():
        perturb_split = np.asarray(
            dataset.data_indices[ds_name]["perturb"], dtype=np.int64
        )
        dataset_id = int(dataset_ids[ds_name])
        current: list[np.ndarray] = []
        for positions in groups.values():
            positions = np.asarray(positions, dtype=np.int64)
            base.require(
                bool(
                    positions.size
                    and positions.min() >= 0
                    and positions.max() < perturb_split.size
                ),
                f"selected split positions are invalid for {ds_name}",
            )
            raw_indices = perturb_split[positions]
            base.require(
                np.unique(raw_indices).size == raw_indices.size,
                f"selected Tahoe group repeats physical cells in {ds_name}",
            )
            current.append(dataset_id * (2**48) + raw_indices)
        if current:
            keys = np.concatenate(current)
            dataset_counts[ds_name] = int(keys.size)
            selected_keys.append(keys)

    all_keys = (
        np.concatenate(selected_keys)
        if selected_keys
        else np.empty((0,), dtype=np.int64)
    )
    expected = int(selection["selected_cells"])
    unique = int(np.unique(all_keys).size)
    base.require(all_keys.size == expected, "selected cell identity count changed")
    base.require(unique == expected, "selected Tahoe screen repeats physical cells")
    selection["cell_identity_audit"] = {
        "passed": True,
        "encoding": "dataset_id*2^48+raw_cell_index",
        "selected_cell_keys": int(all_keys.size),
        "unique_cell_keys": unique,
        "duplicate_cell_keys": int(all_keys.size - unique),
        "dataset_counts": dataset_counts,
    }
    return selection


base.select_screen_groups = select_screen_groups


@hydra.main(version_base=None, config_path="../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg: omegaconf.DictConfig) -> None:
    audit_holdout_labels(cfg)
    base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
