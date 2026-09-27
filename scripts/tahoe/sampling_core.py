from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import anndata
import hydra
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(os.environ.get("CHARM_RESOURCE_ROOT", str(ROOT / "resources")))
DATA_ROOT = Path(os.environ.get("CHARM_DATA_ROOT", str(RESOURCE_ROOT / "data/PerturbDiff_data/perturb_data")))
ARTIFACT_ROOT = Path(os.environ.get("CHARM_ARTIFACT_ROOT", str(RESOURCE_ROOT / "artifacts")))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.apps.sampling.sampling_generation as sampling_generation  # noqa: E402
from src.apps.sampling.sampling_setup import (  # noqa: E402
    build_sampling_datamodule, load_sampling_model, populate_covariate_cfg,
)
from src.apps.sampling.sampling_utils import setup_device  # noqa: E402
from src.apps.training.training_model_checkpoint import (  # noqa: E402
    allow_omegaconf_checkpoint_unpickling,
)
from src.common.utils import setup_loggings  # noqa: E402

FINAL_EXPORT = ROOT / "weights/tahoe.pt"
FINAL_EXPORT_SHA256 = "2c31ef6c936d7447252082406e2e0b64c3fafc91a2f059271d041f2927a2aee4"
EXPECTED_ARTIFACTS = {
    "parent_residual_artifact_path": (
        ARTIFACT_ROOT / "parent_residual_gene_dit/tahoe100m_parent_residual_ocoot_v1.npz",
        "d12b79060c60f2c475b0bb1c1f041e28fb9ecc140f9b7cf2644bdad825fbb467",
    ),
    "parent_residual_set_pca_path": (
        ARTIFACT_ROOT / "ocoot/tahoe100m_control_pca64_div10_v2.npz",
        "989aa9bb8488567b653016d07da5cc388e1969eb75359f974715801b6ce61d63",
    ),
    "parent_residual_real_control_reservoir_path": (
        ARTIFACT_ROOT / "ocoot/tahoe100m_recursive128_minleaf20_real_control_div10_v2.npz",
        "48a5b5fe5706e8d38b60ef380703766fa478d064f5c9997dadcf3dcbb853d9bd",
    ),
}
CONTROL = DATA_ROOT / "tmp_ourtahoe_ctrl.h5ad"
CELL_LINE_MAP = DATA_ROOT / "meta_data/cellname_to_cellline.pkl"
SCENARIOS = ("drug_ood", "cell_line_ood", "double_ood")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def same_path(left: Any, right: str | Path) -> bool:
    return Path(str(left)).expanduser().resolve() == Path(right).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def strings(value: Any) -> list[str]:
    return [] if value is None else [str(item) for item in value]


def audit_cfg(cfg: omegaconf.DictConfig) -> dict[str, Any]:
    require(same_path(cfg.model_checkpoint_path, FINAL_EXPORT), "wrong Tahoe final export")
    require(sha256(FINAL_EXPORT) == FINAL_EXPORT_SHA256, "Tahoe final export SHA changed")
    require(str(cfg.data.data_name) == "Tahoe100mFinetune", "screen requires Tahoe data")
    require(str(cfg.sampling.split) == "test", "screen is test-only")
    require(strings(cfg.data.evaluation_splits) == ["test"], "evaluation_splits must be [test]")
    require(int(cfg.data.use_cell_set) == 1, "screen requires use_cell_set=1")
    require(int(cfg.data.normalize_counts) == 10, "screen requires /10 normalization")
    batch = int(cfg.optimization.micro_batch_size)
    cells_per_group = int(cfg.sampling.tahoe_cells_per_group)
    require(batch == int(cfg.sampling.batch_size), "batch sizes differ")
    require(batch % cells_per_group == 0, "batch must divide by cells_per_group")
    for key in ("num_sampled_batches", "max_perturbations", "fixed_cells_per_perturbation"):
        require(cfg.sampling.get(key) is None, f"screen forbids truncation: {key}")
    require(bool(cfg.sampling.use_ddim), "screen requires DDIM")
    require(str(cfg.sampling.initial_state) == "parent_residual", "wrong initial state")
    require(float(cfg.sampling.guidance_strength) == 0.0, "screen requires zero guidance")
    require(int(cfg.sampling.tahoe_groups_per_scenario) > 0, "invalid group count")
    require(cells_per_group >= 32, "Parent-lock groups require >=32 cells")
    require(FINAL_EXPORT.is_file() and CONTROL.is_file() and CELL_LINE_MAP.is_file(), "missing inputs")
    allow_omegaconf_checkpoint_unpickling()
    checkpoint = torch.load(FINAL_EXPORT, map_location="cpu", weights_only=True, mmap=True)
    require(int(checkpoint.get("global_step", -1)) == 14000, "export is not step 14000")
    model_cfg = checkpoint["hyper_parameters"]["model_cfg"]
    require(bool(model_cfg.get("parent_residual_locked_flow_enabled")), "Parent-lock disabled")
    require(bool(model_cfg.get("parent_residual_semi_balanced_ot_enabled")), "UOT disabled")
    require(float(model_cfg.get("parent_residual_semi_balanced_ot_rho")) == 1.0, "UOT rho changed")
    artifacts = {}
    for key, (path, expected_sha) in EXPECTED_ARTIFACTS.items():
        require(same_path(model_cfg.get(key), path), f"checkpoint {key} changed")
        require(str(model_cfg.get(key.replace("_path", "_sha256"))) == expected_sha, f"{key} SHA changed")
        require(sha256(path) == expected_sha, f"artifact bytes changed: {path}")
        artifacts[key] = {"path": str(path.resolve()), "sha256": expected_sha}
    return {
        "schema_version": 1, "passed": True, "checkpoint": str(FINAL_EXPORT.resolve()),
        "checkpoint_sha256": FINAL_EXPORT_SHA256, "global_step": 14000,
        "sampling_seed": int(cfg.optimization.seed),
        "groups_per_scenario": int(cfg.sampling.tahoe_groups_per_scenario),
        "cells_per_group": cells_per_group, "artifacts": artifacts,
    }


def hash_rank(seed: int, *parts: Any) -> int:
    text = "\0".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def scenario_for(pert: str, line: str, test_perts: set[str], val_perts: set[str], heldout: set[str]) -> str | None:
    pert_ood, line_ood = pert in test_perts, line in heldout
    if pert_ood and line_ood:
        return "double_ood"
    if pert_ood:
        return "drug_ood"
    if line_ood and pert not in val_perts:
        return "cell_line_ood"
    return None


def select_balanced(candidates: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    remaining, selected = list(candidates), []
    used_perts, used_lines, used_plates = set(), set(), {}
    while remaining and len(selected) < target:
        best = min(remaining, key=lambda row: (
            used_plates.get(row["dataset"], 0), row["perturbation"] in used_perts,
            row["cell_line"] in used_lines, hash_rank(seed, row["dataset"], row["key"]),
        ))
        selected.append(best)
        remaining.remove(best)
        used_perts.add(best["perturbation"])
        used_lines.add(best["cell_line"])
        used_plates[best["dataset"]] = used_plates.get(best["dataset"], 0) + 1
    require(len(selected) == target, f"only {len(selected)}/{target} eligible groups")
    return selected


def select_screen_groups(datamodule: Any, cfg: omegaconf.DictConfig) -> dict[str, Any]:
    dataset = datamodule.test_dataset
    cells_per_group = int(cfg.sampling.tahoe_cells_per_group)
    target, seed = int(cfg.sampling.tahoe_groups_per_scenario), int(cfg.optimization.seed)
    test_perts, val_perts = set(strings(cfg.data.holdout_pert.test)), set(strings(cfg.data.holdout_pert.validation))
    heldout = set(strings(cfg.data.holdout_celltype))
    candidates, total_before = {name: [] for name in SCENARIOS}, 0
    for ds_name, path in dataset.dataset_path_map.items():
        cache = dataset.meta_cache._cache[path]
        perts = [str(value) for value in cache.pert_categories]
        lines = [str(value) for value in cache.cell_type_categories]
        batches = [str(value) for value in cache.batch_categories]
        groups = dataset.grouped_pert_data_indices[ds_name]
        total_before += sum(len(values) for values in groups.values())
        for key, values in groups.items():
            if len(values) < cells_per_group:
                continue
            pert_code, line_code, batch_code = map(int, key)
            pert, line, plate = perts[pert_code], lines[line_code], batches[batch_code]
            scenario = scenario_for(pert, line, test_perts, val_perts, heldout)
            if pert != str(cfg.data.control_pert) and scenario is not None:
                candidates[scenario].append({
                    "dataset": ds_name, "key": tuple(map(int, key)), "perturbation": pert,
                    "cell_line": line, "plate": plate, "available_cells": int(len(values)),
                })
    chosen = []
    for index, scenario in enumerate(SCENARIOS):
        rows = select_balanced(candidates[scenario], target, seed + 1009 * (index + 1))
        for row in rows:
            row["scenario"] = scenario
        chosen.extend(rows)
    by_dataset: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    for row in chosen:
        by_dataset.setdefault(row["dataset"], {})[row["key"]] = row
    for ds_name in dataset.dataset_path_map:
        full_groups = dataset.grouped_pert_data_indices[ds_name]
        selected_groups, selected_counts = {}, {}
        for key, row in by_dataset.get(ds_name, {}).items():
            values = np.asarray(full_groups[key], dtype=np.int64)
            rng = np.random.default_rng(hash_rank(seed, ds_name, key) % (2 ** 32))
            sampled = np.sort(rng.choice(values, size=cells_per_group, replace=False))
            selected_groups[key], selected_counts[key] = sampled, cells_per_group
            row["sampled_indices_sha256"] = hashlib.sha256(sampled.tobytes()).hexdigest()
        dataset.grouped_pert_data_indices[ds_name] = selected_groups
        dataset.grouped_pert_num_cell[ds_name] = selected_counts
    chosen.sort(key=lambda row: (SCENARIOS.index(row["scenario"]), row["dataset"], row["key"]))
    counts = {name: sum(row["scenario"] == name for row in chosen) for name in SCENARIOS}
    plates = {name: len({row["dataset"] for row in chosen if row["scenario"] == name}) for name in SCENARIOS}
    return {
        "schema_version": 1, "passed": True, "kind": "tahoe_u2_stratified_screen_selection",
        "full_test_cells_before_filter": int(total_before), "selected_cells": len(chosen) * cells_per_group,
        "selected_groups": len(chosen), "groups_by_scenario": counts, "plates_by_scenario": plates,
        "groups": [{**{key: value for key, value in row.items() if key != "key"},
                    "key_codes": list(row["key"])} for row in chosen],
    }


def install_empty_control(cfg: omegaconf.DictConfig) -> None:
    def empty_control(_cfg: Any):
        with Path(str(cfg.data.selected_gene_file)).open("rb") as handle:
            loaded = pickle.load(handle)
        genes = sorted(loaded) if isinstance(loaded, set) else list(loaded)
        obs = pd.DataFrame({"drugname_drugconc": [], "cell_name": [], "plate": []})
        return anndata.AnnData(X=sparse.csr_matrix((0, 2000)), obs=obs,
                              var=pd.DataFrame(index=pd.Index(genes))), genes
    sampling_generation.load_ctrl_adata = empty_control


def dense(matrix: Any) -> np.ndarray:
    return np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float64)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    return 0.0 if float(left.std()) == 0.0 or float(right.std()) == 0.0 else float(np.corrcoef(left, right)[0, 1])


def r2(truth: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.square(truth - truth.mean()).sum())
    return 0.0 if denominator == 0.0 else 1.0 - float(np.square(truth - pred).sum()) / denominator


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def control_means(groups: pd.DataFrame, genes: list[str]) -> dict[tuple[str, str], np.ndarray]:
    with CELL_LINE_MAP.open("rb") as handle:
        cell_name_to_line = pickle.load(handle)
    control = anndata.read_h5ad(CONTROL, backed="r")
    try:
        require(list(map(str, control.var_names)) == genes, "control gene order changed")
        obs = control.obs.copy()
        obs["cell_line"] = obs["cell_name"].map(cell_name_to_line).astype(str)
        plate_names = sorted(set(obs["plate"].astype(str)), reverse=True)
        plate_map = {name: f"plate{index + 1}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab"
                     for index, name in enumerate(plate_names)}
        obs["canonical_plate"] = obs["plate"].astype(str).map(plate_map)
        requested = set(zip(groups["cell_line"].astype(str), groups["plate"].astype(str)))
        result = {}
        for line, plate in sorted(requested):
            mask = (obs["cell_line"] == line) & (obs["canonical_plate"] == plate)
            if not bool(mask.any()):
                mask = obs["cell_line"] == line
            require(bool(mask.any()), f"no DMSO controls for {(line, plate)}")
            result[(line, plate)] = dense(control.X[np.flatnonzero(mask.to_numpy()), :]).mean(axis=0)
        return result
    finally:
        control.file.close()


def evaluate_outputs(pred_path: Path, true_path: Path, selection: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    pred, truth = anndata.read_h5ad(pred_path), anndata.read_h5ad(true_path)
    require(pred.shape == truth.shape and pred.shape[1] == 2000, "prediction/truth shape changed")
    require(pred.obs.equals(truth.obs), "prediction/truth obs differ")
    pred_x, true_x = dense(pred.X), dense(truth.X)
    require(np.isfinite(pred_x).all() and float(pred_x.min()) >= -1.0e-7, "invalid prediction matrix")
    lookup = {(str(row["perturbation"]), str(row["cell_line"]), str(row["plate"])): str(row["scenario"])
              for row in selection["groups"]}
    obs = pred.obs.reset_index(drop=True).copy()
    grouped = obs.groupby(["drugname_drugconc", "cell_line", "plate"], sort=True, observed=True).indices
    group_frame = pd.DataFrame([{"perturbation": key[0], "cell_line": key[1], "plate": key[2]} for key in grouped])
    controls, rows = control_means(group_frame, list(map(str, pred.var_names))), []
    for key, indices in grouped.items():
        pert, line, plate = map(str, key)
        require((pert, line, plate) in lookup, f"unexpected group {(pert, line, plate)}")
        indices = np.asarray(indices, dtype=np.int64)
        pred_cells, true_cells = pred_x[indices], true_x[indices]
        pred_mean, true_mean = pred_cells.mean(0), true_cells.mean(0)
        ctrl_mean = controls[(line, plate)]
        pred_delta, true_delta = pred_mean - ctrl_mean, true_mean - ctrl_mean
        pred_error, ctrl_error = pred_mean - true_mean, ctrl_mean - true_mean
        mse, ctrl_mse = float(np.square(pred_error).mean()), float(np.square(ctrl_error).mean())
        true_top, pred_top = np.argpartition(np.abs(true_delta), -100)[-100:], np.argpartition(np.abs(pred_delta), -100)[-100:]
        rows.append({
            "scenario": lookup[(pert, line, plate)], "perturbation": pert, "cell_line": line,
            "plate": plate, "n_cells": int(len(indices)), "r2": r2(true_mean, pred_mean),
            "pearson": pearson(true_mean, pred_mean), "mae": float(np.abs(pred_error).mean()), "mse": mse,
            "control_r2": r2(true_mean, ctrl_mean), "control_mae": float(np.abs(ctrl_error).mean()),
            "control_mse": ctrl_mse, "mse_gain_vs_control": ctrl_mse - mse,
            "relative_mse_improvement_vs_control": (ctrl_mse - mse) / max(ctrl_mse, 1e-12),
            "delta_pearson": pearson(true_delta, pred_delta), "delta_cosine": cosine(true_delta, pred_delta),
            "delta_mse": float(np.square(true_delta - pred_delta).mean()),
            "delta_top100_overlap": len(set(true_top.tolist()) & set(pred_top.tolist())) / 100.0,
            "delta_top100_sign_agreement": float(np.mean(np.sign(true_delta[true_top]) == np.sign(pred_delta[true_top]))),
            "delta_norm_ratio": float(np.linalg.norm(pred_delta) / max(float(np.linalg.norm(true_delta)), 1e-12)),
            "variance_ratio": float(pred_cells.var(0).mean() / max(float(true_cells.var(0).mean()), 1e-12)),
        })
    per_group = pd.DataFrame(rows).sort_values(["scenario", "plate", "cell_line", "perturbation"]).reset_index(drop=True)
    require(len(per_group) == int(selection["selected_groups"]), "group coverage changed")
    expected_cells = int(selection["selected_cells"] / selection["selected_groups"])
    require((per_group["n_cells"] == expected_cells).all(), "per-group cell count changed")
    excluded = {"scenario", "perturbation", "cell_line", "plate", "n_cells"}
    metrics = [column for column in per_group.columns if column not in excluded]
    summary_rows = []
    for scenario, frame in [("overall", per_group), *list(per_group.groupby("scenario", sort=True))]:
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=np.float64)
            require(np.isfinite(values).all(), f"non-finite metric {metric}")
            summary_rows.append({"scenario": scenario, "metric": metric, "mean": float(values.mean()),
                                 "median": float(np.median(values)), "std": float(values.std(ddof=1)),
                                 "count": int(len(values)), "min": float(values.min()), "max": float(values.max())})
    summary = pd.DataFrame(summary_rows)
    headline_names = ("r2", "control_r2", "mae", "mse", "control_mse", "mse_gain_vs_control",
                      "relative_mse_improvement_vs_control", "delta_pearson", "delta_cosine",
                      "delta_top100_overlap", "delta_top100_sign_agreement", "delta_norm_ratio", "variance_ratio")
    def value(scenario: str, metric: str) -> float:
        return float(summary.loc[(summary["scenario"] == scenario) & (summary["metric"] == metric), "mean"].iloc[0])
    per_group_path, summary_path = output_dir / "per_group_metrics.csv", output_dir / "scenario_summary.csv"
    per_group.to_csv(per_group_path, index=False)
    summary.to_csv(summary_path, index=False)
    result = {
        "schema_version": 1, "passed": True, "kind": "tahoe_u2_stratified_screen_metrics",
        "pred": str(pred_path.resolve()), "pred_sha256": sha256(pred_path),
        "truth": str(true_path.resolve()), "truth_sha256": sha256(true_path),
        "groups": int(len(per_group)), "cells": int(len(obs)), "genes": int(pred.shape[1]),
        "headline_metrics": {metric: value("overall", metric) for metric in headline_names},
        "scenario_metrics": {scenario: {metric: value(scenario, metric) for metric in headline_names}
                             for scenario in SCENARIOS},
        "per_group_metrics": str(per_group_path.resolve()), "scenario_summary": str(summary_path.resolve()),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["metrics"] = str(metrics_path.resolve())
    return result


@hydra.main(version_base=None, config_path="../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg: omegaconf.DictConfig) -> None:
    omegaconf.OmegaConf.resolve(cfg)
    audit, logger = audit_cfg(cfg), setup_loggings(cfg)
    seed = int(cfg.optimization.seed)
    pl.seed_everything(seed, workers=True)
    datamodule = build_sampling_datamodule(cfg, logger)
    datamodule.evaluation_split_names = ["test"]
    populate_covariate_cfg(cfg, datamodule)
    datamodule.setup_dataset()
    selection = select_screen_groups(datamodule, cfg)
    output_dir = Path(str(cfg.sampling.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screen_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (output_dir / "selection_manifest.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    logger.info("TAHOE_U2_SCREEN_SELECTION %s", json.dumps({key: selection[key] for key in (
        "full_test_cells_before_filter", "selected_cells", "selected_groups", "groups_by_scenario", "plates_by_scenario")}, sort_keys=True))
    if os.environ.get("TAHOE_SCREEN_AUDIT_ONLY", "false").lower() == "true":
        print(json.dumps(selection, sort_keys=True))
        return
    model = load_sampling_model(cfg, logger, datamodule)
    pl.seed_everything(seed, workers=True)
    install_empty_control(cfg)
    device = setup_device(cfg, logger)
    before = set(output_dir.glob("diffusion_predict_*.h5ad")) | set(output_dir.glob("diffusion_true_*.h5ad"))
    _, samples, _, _ = sampling_generation.generate_samples(model, model.diffusion, cfg, device, logger, datamodule, pca_for_decode=None)
    require(tuple(samples.shape) == (int(selection["selected_cells"]), 2000), "sample shape changed")
    created = (set(output_dir.glob("diffusion_predict_*.h5ad")) | set(output_dir.glob("diffusion_true_*.h5ad"))) - before
    pred_paths = sorted(path for path in created if path.name.startswith("diffusion_predict_"))
    true_paths = sorted(path for path in created if path.name.startswith("diffusion_true_"))
    require(len(pred_paths) == len(true_paths) == 1, "expected one fresh prediction/truth pair")
    metrics = evaluate_outputs(pred_paths[0], true_paths[0], selection, output_dir)
    logger.info("TAHOE_U2_SCREEN_COMPLETE %s", json.dumps(metrics, sort_keys=True))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
