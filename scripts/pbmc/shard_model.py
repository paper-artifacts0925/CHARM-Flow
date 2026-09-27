from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import sys

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
    build_sampling_datamodule,
    load_sampling_model,
    populate_covariate_cfg,
)
from src.apps.sampling.sampling_utils import setup_device  # noqa: E402
from src.apps.training.training_model_checkpoint import (  # noqa: E402
    allow_omegaconf_checkpoint_unpickling,
)
from src.common.utils import setup_loggings  # noqa: E402


RUN = "pbmc_ctxw256_trainonly_hs3_14k_s42_v1_formal14000"
FINAL = ROOT / "weights/pbmc.pt"
FINAL_SHA256 = "a083cff84245b644ebf0e67d7cf0da6963ac77467edb2f7d511ee0d6bed610ab"
PROTOCOL = ROOT / "configs/pbmc_u2_attempt2_full_test_v1.json"
CONTEXT = ARTIFACT_ROOT / "ocoot/pbmc_recursive128_minleaf20_context_v2.runtime.pkl"
CONTEXT_SHA256 = "042781bccbe65c8b77f18cb0c16bec5ecc52b3fd06a2569172708ce688494f90"
RESERVOIR = ARTIFACT_ROOT / "ocoot/pbmc_recursive128_minleaf20_real_control_div10_v2.npz"
RESERVOIR_SHA256 = "79c958ab60dbdb277876a062d453583a9a2b02b546c5a2fd3a974c04f35c818a"
GROUP_CACHE = (
    DATA_ROOT / "indices_cache/"
    "grouped_pert_data_indices_Parse_10M_PBMC_cytokines_processed_Xselected_test.pkl"
)
GROUP_CACHE_SHA256 = "feb67093ad59f2eff2d82cff649323ad037d72740c5c595d1edef088b6fb56e7"
COUNT_CACHE = (
    DATA_ROOT / "indices_cache/"
    "grouped_pert_num_cell_Parse_10M_PBMC_cytokines_processed_Xselected_test.pkl"
)
COUNT_CACHE_SHA256 = "8ac332a419a3c978e49ce907b6063a7419291ec494d7f51e96bcc43990f68219"
RUN_RE = re.compile(
    rf"^{RUN}_pbmc_full_test_sampleseed(42|20260811)_shard(0[0-6])$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def same_path(left: object, right: Path) -> bool:
    return Path(str(left)).expanduser().resolve() == right.resolve()


def as_mapping(value: object, label: str) -> Mapping:
    if omegaconf.OmegaConf.is_config(value):
        value = omegaconf.OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise AssertionError(f"{label} must be a mapping")
    return value


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = value["expected"]
    assert value["schema_version"] == 1
    assert value["split"] == "test"
    assert value["control"] == "PBS"
    assert len(value["official_test_cytokines"]) == expected["cytokines"] == 62
    assert len(set(value["official_test_cytokines"])) == 62
    assert len(value["cell_types"]) == expected["cell_types"] == 18
    assert len(value["shards"]) == 7
    flattened = [x for shard in value["shards"] for x in shard["cytokines"]]
    assert len(flattened) == len(set(flattened)) == 62
    assert set(flattened) == set(value["official_test_cytokines"])
    assert sum(x["expected_treated_cells"] for x in value["shards"]) == 2_260_453
    assert sum(x["expected_nonempty_pairs"] for x in value["shards"]) == 1_115
    return value


def audit_cfg(cfg: omegaconf.DictConfig) -> tuple[dict, dict]:
    protocol = load_protocol()
    match = RUN_RE.fullmatch(str(cfg.run_name))
    if match is None:
        raise AssertionError("PBMC shard run_name is outside the frozen grid")
    seed, shard_text = map(int, match.groups())
    shard_id = int(cfg.sampling.pbmc_shard_id)
    if shard_id != shard_text:
        raise AssertionError("run_name and pbmc_shard_id disagree")
    if int(cfg.optimization.seed) != seed:
        raise AssertionError("run_name and optimization.seed disagree")
    if seed not in (42, 20260811):
        raise AssertionError("unsupported sampling seed")
    if str(cfg.sampling.split) != "test":
        raise AssertionError("PBMC shard sampler is test-only")
    if list(map(str, cfg.data.evaluation_splits)) != ["test"]:
        raise AssertionError("evaluation_splits must be exactly [test]")
    if int(cfg.data.use_cell_set) != 1:
        raise AssertionError("PBMC full test requires use_cell_set=1")
    if cfg.sampling.num_sampled_batches is not None:
        raise AssertionError("formal PBMC shards forbid num_sampled_batches truncation")
    if cfg.sampling.fixed_cells_per_perturbation is not None:
        raise AssertionError("formal PBMC shards forbid fixed-source replication")
    if cfg.sampling.max_perturbations is not None:
        raise AssertionError("formal PBMC shards forbid max_perturbations truncation")
    if bool(cfg.sampling.pbmc_append_controls):
        raise AssertionError("shards must not append duplicated PBS rows")
    if str(cfg.data.data_name) != "PBMCFinetune":
        raise AssertionError("wrong dataset for PBMC full test")
    if str(cfg.data.pert_col) != "cytokine" or str(cfg.data.cell_line_key) != "cell_type":
        raise AssertionError("PBMC observation keys changed")
    if str(cfg.data.perturbseq_batch_col) != "donor" or str(cfg.data.control_pert) != "PBS":
        raise AssertionError("PBMC donor/control contract changed")
    if list(map(str, cfg.data.holdout_batches)) != protocol["heldout_donors"]:
        raise AssertionError("PBMC held-out donors changed")
    if list(map(str, cfg.data.holdout_pert.test)) != protocol["official_test_cytokines"]:
        raise AssertionError("the shared full-test split must retain all 62 cytokines")
    if bool(cfg.data.skip_cached_indices):
        raise AssertionError("PBMC shard sampler must consume frozen grouped caches")
    if not same_path(cfg.model_checkpoint_path, FINAL):
        raise AssertionError("wrong PBMC final export")
    if not same_path(cfg.sampling.pbmc_protocol_path, PROTOCOL):
        raise AssertionError("wrong PBMC protocol manifest")
    if not same_path(cfg.data.control_distribution_context_path, CONTEXT):
        raise AssertionError("wrong PBMC Child context")
    shard = protocol["shards"][shard_id]
    if int(cfg.sampling.pbmc_expected_treated_cells) != shard["expected_treated_cells"]:
        raise AssertionError("shard expected treated count changed")
    if int(cfg.sampling.pbmc_expected_nonempty_pairs) != shard["expected_nonempty_pairs"]:
        raise AssertionError("shard expected pair count changed")
    for path, expected in (
        (FINAL, FINAL_SHA256), (CONTEXT, CONTEXT_SHA256),
        (RESERVOIR, RESERVOIR_SHA256), (GROUP_CACHE, GROUP_CACHE_SHA256),
        (COUNT_CACHE, COUNT_CACHE_SHA256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"frozen PBMC artifact changed: {path}")

    allow_omegaconf_checkpoint_unpickling()
    checkpoint = torch.load(FINAL, map_location="cpu", weights_only=True, mmap=True)
    assert int(checkpoint.get("global_step", -1)) == 14_000
    assert int(checkpoint.get("epoch", -1)) == 2
    state = as_mapping(checkpoint.get("state_dict"), "state_dict")
    assert len(state) == 637
    assert all(torch.is_tensor(x) and bool(torch.isfinite(x).all()) for x in state.values())
    model = as_mapping(checkpoint["hyper_parameters"]["model_cfg"], "model_cfg")
    expected_model = {
        "model_type": "parent_residual_gene_dit",
        "parent_residual_enabled": True,
        "parent_residual_locked_flow_enabled": True,
        "parent_residual_matched_set_enabled": True,
        "parent_residual_positive_multiplicity_cap_to_active_children": True,
        "parent_residual_semi_balanced_ot_enabled": True,
        "parent_residual_semi_balanced_ot_epsilon": 0.1,
        "parent_residual_semi_balanced_ot_rho": 1.0,
        "parent_residual_semi_balanced_ot_iterations": 50,
        "parent_residual_semi_balanced_ot_cost_median_normalization": True,
        "parent_residual_de_aware_enabled": True,
        "parent_residual_hurdle_geometry_enabled": True,
        "parent_residual_sparse_support_enabled": True,
        "parent_residual_sparse_joint_context_enabled": False,
        "parent_residual_soft_gated_regression_enabled": False,
        "parent_residual_gene_relation_enabled": False,
        "parent_residual_hurdle_alpha": 1.0,
        "parent_residual_de_aware_artifact_sha256": "416beddd77cff1c675d8ab6fca3dcaa9deed39b0ea762d8dd88242a92cf24f47",
        "parent_residual_hurdle_artifact_sha256": "6acdae0a9ab7d9da9cba62614a753a6c47c421bdc05003aa2763089adb0bcd19",
        "parent_residual_hurdle_source_h5ad_sha256": "30f125ebe41ed9342a7ff3d977acbe9a4f356365c081432d523be9a6d23f5767",
    }
    for key, expected in expected_model.items():
        if model.get(key, object()) != expected:
            raise AssertionError(f"PBMC checkpoint mechanism mismatch: {key}")
    if not same_path(model["parent_residual_real_control_reservoir_path"], RESERVOIR):
        raise AssertionError("wrong PBMC real-control reservoir")
    if model["parent_residual_real_control_reservoir_sha256"] != RESERVOIR_SHA256:
        raise AssertionError("PBMC reservoir hash pin changed")
    return protocol, shard


def filter_test_dataset(datamodule, protocol: dict, shard: dict) -> dict:
    dataset = datamodule.test_dataset
    if len(dataset.dataset_path_map) != 1:
        raise AssertionError("PBMC shard filter requires exactly one dataset")
    ds_name = next(iter(dataset.dataset_path_map))
    cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
    categories = [str(x) for x in cache.pert_categories]
    allowed_names = set(map(str, shard["cytokines"]))
    allowed_codes = {i for i, name in enumerate(categories) if name in allowed_names}
    if len(allowed_codes) != len(allowed_names):
        raise AssertionError("not all shard cytokines exist in PBMC metadata")
    full_groups = dataset.grouped_pert_data_indices[ds_name]
    full_counts = dataset.grouped_pert_num_cell[ds_name]
    original_cells = sum(len(values) for values in full_groups.values())
    if original_cells != protocol["expected"]["treated_cells"]:
        raise AssertionError("shared grouped test cache is not the official full test")
    filtered_groups = {
        key: values for key, values in full_groups.items()
        if int(key[0]) in allowed_codes and len(values)
    }
    filtered_counts = {
        key: int(full_counts[key]) for key in filtered_groups
    }
    observed_cells = sum(len(values) for values in filtered_groups.values())
    observed_pairs = len({(int(key[0]), int(key[1])) for key in filtered_groups})
    observed_names = {categories[int(key[0])] for key in filtered_groups}
    if observed_cells != int(shard["expected_treated_cells"]):
        raise AssertionError("PBMC shard cell count differs from frozen protocol")
    if observed_pairs != int(shard["expected_nonempty_pairs"]):
        raise AssertionError("PBMC shard nonempty-pair count differs from protocol")
    if observed_names != allowed_names:
        raise AssertionError("PBMC shard cytokine coverage differs from protocol")
    dataset.grouped_pert_data_indices[ds_name] = filtered_groups
    dataset.grouped_pert_num_cell[ds_name] = filtered_counts
    if sha256(GROUP_CACHE) != GROUP_CACHE_SHA256 or sha256(COUNT_CACHE) != COUNT_CACHE_SHA256:
        raise AssertionError("shared grouped cache changed during in-memory filtering")
    return {
        "schema_version": 1,
        "passed": True,
        "shard_id": int(shard["id"]),
        "cytokines": list(shard["cytokines"]),
        "observed_treated_cells": observed_cells,
        "observed_nonempty_cytokine_cell_type_pairs": observed_pairs,
        "full_test_treated_cells_before_filter": original_cells,
        "controls_appended_to_shard": 0,
        "group_cache_sha256_before_after": GROUP_CACHE_SHA256,
        "count_cache_sha256_before_after": COUNT_CACHE_SHA256,
    }


def install_empty_control_output(cfg: omegaconf.DictConfig) -> None:
    """Prevent seven duplicate PBS matrices; controls are appended once at merge."""

    def empty_control(_cfg):
        import pickle

        with Path(str(cfg.data.selected_gene_file)).open("rb") as handle:
            genes = list(pickle.load(handle))
        obs = pd.DataFrame({"cytokine": [], "cell_type": [], "donor": []})
        var = pd.DataFrame(index=pd.Index(genes, name=None))
        return anndata.AnnData(X=sparse.csr_matrix((0, len(genes))), obs=obs, var=var), genes

    sampling_generation.load_ctrl_adata = empty_control


@hydra.main(version_base=None, config_path="../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg: omegaconf.DictConfig) -> None:
    omegaconf.OmegaConf.resolve(cfg)
    protocol, shard = audit_cfg(cfg)
    logger = setup_loggings(cfg)
    seed = int(cfg.optimization.seed)
    pl.seed_everything(seed, workers=True)
    datamodule = build_sampling_datamodule(cfg, logger)
    datamodule.evaluation_split_names = ["test"]
    populate_covariate_cfg(cfg, datamodule)
    datamodule.setup_dataset()
    shard_audit = filter_test_dataset(datamodule, protocol, shard)
    logger.info("PBMC_U2_SHARD_FILTER_AUDIT %s", json.dumps(shard_audit, sort_keys=True))
    if os.environ.get("PBMC_SHARD_AUDIT_ONLY", "false").lower() == "true":
        print(json.dumps(shard_audit, sort_keys=True))
        return
    model = load_sampling_model(cfg, logger, datamodule)
    pl.seed_everything(seed, workers=True)
    logger.info(
        "PBMC_U2_ATTEMPT2_POST_LOAD_RESEED seed=%d shard=%02d", seed, int(shard["id"])
    )
    install_empty_control_output(cfg)
    device = setup_device(cfg, logger)
    _, samples, _, _ = sampling_generation.generate_samples(
        model, model.diffusion, cfg, device, logger, datamodule, pca_for_decode=None
    )
    if int(samples.shape[0]) != int(shard["expected_treated_cells"]):
        raise AssertionError("saved PBMC shard sample count changed")
    logger.info(
        "PBMC_U2_ATTEMPT2_SHARD_COMPLETE seed=%d shard=%02d shape=%s",
        seed, int(shard["id"]), tuple(samples.shape),
    )


if __name__ == "__main__":
    main()
