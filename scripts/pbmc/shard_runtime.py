import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time

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
BASE_PATH = ROOT / "scripts/pbmc/shard_sampler_core.py"
SPEC = importlib.util.spec_from_file_location("_pbmc_u2_shard_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import PBMC shard base: {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

import src.apps.sampling.sampling_generation as sampling_generation  # noqa: E402
from src.apps.sampling.sampling_provenance import annotate_control_source  # noqa: E402
from src.apps.sampling.sampling_setup import (  # noqa: E402
    build_sampling_datamodule,
    load_sampling_model,
    populate_covariate_cfg,
)
from src.apps.sampling.sampling_utils import setup_device  # noqa: E402
from src.common.utils import setup_loggings  # noqa: E402


SHARDED_PROTOCOL = ROOT / "configs/pbmc_u2_attempt2_sharded_sampling_v1.json"
SHARDED_PROTOCOL_SHA256 = "87501e58123b35f805426fbdad3e6f0127c65cfd6a4181e58a70cf2bde818784"
OFFICIAL_CACHE_DIR = base.GROUP_CACHE.parent
DATASET = (
    DATA_ROOT / "finetune_data/pbmc_new/"
    "Parse_10M_PBMC_cytokines_processed_Xselected.h5ad"
)
DATASET_SIZE = 796_808_314_640
SELECTED_GENES = DATA_ROOT / "selected_genes/pbmc_real_selected_genes.pkl"
SELECTED_GENES_SHA256 = "d0593d524caeaf431a9a0ac3ffeb5d192aea20e9bc891ea059fc94180ff04089"
PBS_CONTROL = DATA_ROOT / "tmp_pbmc_ctrl.h5ad"
PBS_CONTROL_SHA256 = "d298123fefd2b5e7a1095b9f8bb7910c1406a285bbfcf9b763af348b49a2c49f"
PARENT = ARTIFACT_ROOT / "parent_residual_gene_dit/pbmc_parent_residual_ocoot_v1.npz"
PARENT_SHA256 = "7a0a765541db4addb79683f4f4c34a1cab3eb92efd546e776ac26676d78995d3"
CONTEXT_NPZ = ARTIFACT_ROOT / "ocoot/pbmc_recursive128_minleaf20_context_v2.npz"
CONTEXT_NPZ_SHA256 = "ae2e55e3a7f4a0e333e65b9efa90ff91ce5736363d199a037e4110119312b17d"
SET_PCA = ARTIFACT_ROOT / "ocoot/pbmc_control_pca64_div10_v2.npz"
SET_PCA_SHA256 = "bf123768469db7d084339f039a268c1b41055f9d65089c354a6a45c02e58045e"
MICRO_BATCH = 1024


def derived_seed(base_seed: int, shard_id: int) -> int:
    return (int(base_seed) + 1_000_003 * (int(shard_id) + 1)) % 4_294_967_295


def audit_p0_cfg(cfg: omegaconf.DictConfig, shard: dict) -> dict:
    if not base.same_path(cfg.sampling.pbmc_sharded_protocol_path, SHARDED_PROTOCOL):
        raise AssertionError("wrong sharded_sampling_v1 protocol path")
    if base.sha256(SHARDED_PROTOCOL) != SHARDED_PROTOCOL_SHA256:
        raise AssertionError("sharded_sampling_v1 protocol changed")
    overlay = json.loads(SHARDED_PROTOCOL.read_text())
    declared_base = (ROOT / str(overlay["base_protocol"])).resolve()
    if declared_base != base.PROTOCOL.resolve():
        raise AssertionError("sharded overlay points to the wrong base protocol")
    if base.sha256(base.PROTOCOL) != overlay["base_protocol_sha256"]:
        raise AssertionError("base PBMC full-test protocol changed")
    if overlay["sampling_protocol"] != "sharded_sampling_v1":
        raise AssertionError("wrong PBMC sharded protocol kind")
    if overlay["monolithic_seed_bit_equivalent"] is not False:
        raise AssertionError("sharded protocol must not claim monolithic equivalence")
    if not base.same_path(cfg.data.indices_cache_dir, OFFICIAL_CACHE_DIR):
        raise AssertionError("PBMC shards must use the official full grouped-cache directory")
    if not base.same_path(cfg.data.dataset_path, DATASET):
        raise AssertionError("PBMC source H5AD path changed")
    if DATASET.stat().st_size != DATASET_SIZE:
        raise AssertionError("PBMC source H5AD size changed")
    if not base.same_path(cfg.data.selected_gene_file, SELECTED_GENES):
        raise AssertionError("PBMC selected-gene path changed")
    if not base.same_path(cfg.path.pbmc_ctrl_h5ad, PBS_CONTROL):
        raise AssertionError("canonical PBMC PBS-control path changed")
    if int(cfg.optimization.micro_batch_size) != MICRO_BATCH:
        raise AssertionError("PBMC sharded micro-batch changed")
    if int(cfg.sampling.batch_size) != MICRO_BATCH:
        raise AssertionError("PBMC sampling.batch_size changed")
    if int(cfg.data.use_cell_set) != 1:
        raise AssertionError("PBMC sharded use_cell_set changed")
    if int(cfg.data.normalize_counts) != 10:
        raise AssertionError("PBMC saved-unit normalization changed")
    if not bool(cfg.sampling.use_ddim):
        raise AssertionError("PBMC full test requires DDIM")
    if str(cfg.sampling.initial_state) != "parent_residual":
        raise AssertionError("PBMC full test requires parent_residual initial state")
    if float(cfg.sampling.guidance_strength) != 0.0:
        raise AssertionError("PBMC full test requires zero guidance")
    if bool(cfg.sampling.pbmc_append_controls):
        raise AssertionError("PBMC treated shards must contain no duplicate PBS")
    if str(cfg.sampling.pbmc_rng_protocol) != "sharded_sampling_v1":
        raise AssertionError("PBMC shard RNG protocol changed")
    for path, expected in (
        (SELECTED_GENES, SELECTED_GENES_SHA256), (PBS_CONTROL, PBS_CONTROL_SHA256),
        (PARENT, PARENT_SHA256), (CONTEXT_NPZ, CONTEXT_NPZ_SHA256),
        (SET_PCA, SET_PCA_SHA256),
    ):
        if not path.is_file() or base.sha256(path) != expected:
            raise AssertionError(f"PBMC P0 artifact changed: {path}")
    checkpoint = torch.load(base.FINAL, map_location="cpu", weights_only=True, mmap=True)
    model = base.as_mapping(checkpoint["hyper_parameters"]["model_cfg"], "model_cfg")
    expected_paths = {
        "parent_residual_artifact_path": (PARENT, PARENT_SHA256),
        "parent_residual_set_pca_path": (SET_PCA, SET_PCA_SHA256),
        "parent_residual_real_control_reservoir_path": (base.RESERVOIR, base.RESERVOIR_SHA256),
    }
    for key, (path, expected_sha) in expected_paths.items():
        if not base.same_path(model.get(key), path):
            raise AssertionError(f"checkpoint {key} changed")
        sha_key = key.replace("_path", "_sha256")
        if model.get(sha_key) != expected_sha:
            raise AssertionError(f"checkpoint {sha_key} changed")
    shard_id = int(shard["id"])
    base_seed = int(cfg.optimization.seed)
    return {
        "schema_version": 1, "passed": True,
        "sampling_protocol": "sharded_sampling_v1",
        "base_seed": base_seed, "shard_id": shard_id,
        "derived_seed": derived_seed(base_seed, shard_id),
        "monolithic_seed_bit_equivalent": False,
        "micro_batch_size": MICRO_BATCH,
    }


def filter_with_frozen_cache_audit(datamodule, protocol: dict, shard: dict) -> dict:
    dataset = datamodule.test_dataset
    if not base.same_path(dataset.data_args.indices_cache_dir, OFFICIAL_CACHE_DIR):
        raise AssertionError("runtime PBMC dataset escaped the official cache directory")
    if len(dataset.dataset_path_map) != 1:
        raise AssertionError("PBMC full test must contain one dataset")
    ds_name = next(iter(dataset.dataset_path_map))
    runtime_groups = dataset.grouped_pert_data_indices[ds_name]
    runtime_counts = dataset.grouped_pert_num_cell[ds_name]
    with base.GROUP_CACHE.open("rb") as handle:
        frozen_groups = pickle.load(handle)
    with base.COUNT_CACHE.open("rb") as handle:
        frozen_counts = pickle.load(handle)
    if set(runtime_groups) != set(frozen_groups) or set(runtime_counts) != set(frozen_counts):
        raise AssertionError("runtime grouped-cache keys differ from frozen pickles")
    for key, values in runtime_groups.items():
        if not np.array_equal(np.asarray(values), np.asarray(frozen_groups[key])):
            raise AssertionError(f"runtime grouped indices changed at {key}")
        if int(runtime_counts[key]) != int(frozen_counts[key]):
            raise AssertionError(f"runtime grouped count changed at {key}")
    record = base.filter_test_dataset(datamodule, protocol, shard)
    record.update({
        "sampling_protocol": "sharded_sampling_v1",
        "official_cache_directory": str(OFFICIAL_CACHE_DIR.resolve()),
        "frozen_group_indices_sha256": base.GROUP_CACHE_SHA256,
        "frozen_group_counts_sha256": base.COUNT_CACHE_SHA256,
        "runtime_groups_equal_frozen_pickle_before_filter": True,
    })
    return record


def install_empty_control_output(cfg: omegaconf.DictConfig) -> None:
    def empty_control(_cfg):
        with SELECTED_GENES.open("rb") as handle:
            loaded = pickle.load(handle)
        genes = sorted(loaded) if isinstance(loaded, set) else list(loaded)
        if len(genes) != 2000:
            raise AssertionError("PBMC selected-gene asset is not 2000 genes")
        obs = pd.DataFrame({"cytokine": [], "cell_type": [], "donor": []})
        var = pd.DataFrame(index=pd.Index(genes))
        controls = anndata.AnnData(
            X=sparse.csr_matrix((0, 2000)), obs=obs, var=var
        )
        annotate_control_source(
            controls, source_path=PBS_CONTROL, dataset_name="pbmc_control"
        )
        return controls, genes
    sampling_generation.load_ctrl_adata = empty_control


def setup_dataset_with_exclusive_lock(datamodule, base_seed: int) -> dict:
    """Serialize only the PBMC split/cache mutation phase across seven shards."""

    lock_path = (
        ROOT / "logs/pbmc_u2_attempt2_full_test"
        / f"seed{int(base_seed)}" / "shared_dataset_setup.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    wait_started = time.monotonic()
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        acquired = time.monotonic()
        try:
            datamodule.setup_dataset()
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    released = time.monotonic()
    return {
        "schema_version": 1,
        "passed": True,
        "path": str(lock_path.resolve()),
        "mechanism": "fcntl.flock_LOCK_EX",
        "scope": "datamodule.setup_dataset",
        "wait_seconds": acquired - wait_started,
        "held_seconds": released - acquired,
        "released_before_post_setup_filter": True,
        "released_before_model_load_and_sampling": True,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg: omegaconf.DictConfig) -> None:
    omegaconf.OmegaConf.resolve(cfg)
    protocol, shard = base.audit_cfg(cfg)
    p0_record = audit_p0_cfg(cfg, shard)
    logger = setup_loggings(cfg)
    base_seed = int(cfg.optimization.seed)
    shard_seed = int(p0_record["derived_seed"])
    pl.seed_everything(base_seed, workers=True)
    datamodule = build_sampling_datamodule(cfg, logger)
    datamodule.seed = shard_seed
    datamodule.evaluation_split_names = ["test"]
    populate_covariate_cfg(cfg, datamodule)
    setup_lock_record = setup_dataset_with_exclusive_lock(datamodule, base_seed)
    filter_record = filter_with_frozen_cache_audit(datamodule, protocol, shard)
    filter_record.update(p0_record)
    filter_record["shared_dataset_setup_lock"] = setup_lock_record
    output_dir = Path(str(cfg.sampling.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "pbmc_shard_filter_audit.json"
    audit_path.write_text(json.dumps(filter_record, indent=2, sort_keys=True) + "\n")
    logger.info("PBMC_U2_SHARD_FILTER_AUDIT %s", json.dumps(filter_record, sort_keys=True))
    if os.environ.get("PBMC_SHARD_AUDIT_ONLY", "false").lower() == "true":
        print(json.dumps(filter_record, sort_keys=True))
        return
    model = load_sampling_model(cfg, logger, datamodule)
    pl.seed_everything(shard_seed, workers=True)
    logger.info(
        "PBMC_U2_ATTEMPT2_POST_LOAD_RESEED protocol=sharded_sampling_v1 "
        "base_seed=%d derived_seed=%d shard=%02d monolithic_bit_equivalent=false",
        base_seed, shard_seed, int(shard["id"]),
    )
    install_empty_control_output(cfg)
    device = setup_device(cfg, logger)
    _, samples, _, _ = sampling_generation.generate_samples(
        model, model.diffusion, cfg, device, logger, datamodule, pca_for_decode=None
    )
    if tuple(samples.shape) != (int(shard["expected_treated_cells"]), 2000):
        raise AssertionError("PBMC shard output shape changed")
    logger.info(
        "PBMC_U2_ATTEMPT2_SHARD_COMPLETE base_seed=%d derived_seed=%d "
        "shard=%02d shape=%s",
        base_seed, shard_seed, int(shard["id"]), tuple(samples.shape),
    )


if __name__ == "__main__":
    main()
