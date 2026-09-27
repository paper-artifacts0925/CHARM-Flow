from collections.abc import Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import omegaconf


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(os.environ.get("CHARM_RESOURCE_ROOT", str(ROOT / "resources")))
DATA_ROOT = Path(os.environ.get("CHARM_DATA_ROOT", str(RESOURCE_ROOT / "data/PerturbDiff_data/perturb_data")))
ARTIFACT_ROOT = Path(os.environ.get("CHARM_ARTIFACT_ROOT", str(RESOURCE_ROOT / "artifacts")))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


impl = load_module(
    "_pbmc_donoraware_hardened_impl",
    ROOT / "scripts/pbmc/shard_runtime.py",
)
base = load_module(
    "_pbmc_donoraware_shard_base",
    ROOT / "scripts/pbmc/shard_model.py",
)
impl.base = base

RUN = "pbmc_donoraware_hs3_14k_s42_v2_boundfix1_formal14000"
FINAL = ROOT / "weights/pbmc.pt"
FINAL_SHA256 = "a083cff84245b644ebf0e67d7cf0da6963ac77467edb2f7d511ee0d6bed610ab"
TRAIN_CONFIG = ROOT / "configs/frozen/pbmc_train.yaml"
TRAIN_CONFIG_SHA256 = "be8f0c6fa869a3daa58288ff0aa81ed627e78b338c805e3b9565d161e35e6c58"
CONTEXT = ARTIFACT_ROOT / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_context_v1.npz"
CONTEXT_SHA256 = "dd91bcb3aae06d70a4591ff7c0b4cc1f93038f5fa3b9199bd9515cd6408b0e6d"
RESERVOIR = ARTIFACT_ROOT / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_real8_v1.npz"
RESERVOIR_SHA256 = "c7273173a5d2a9cfb7664b366f853b4ad431bfb60b9783921cd79c06cd5d88f5"
PARENT = ARTIFACT_ROOT / "parent_residual_gene_dit/pbmc_donor_celltype_parent_residual_ocoot_v1.npz"
PARENT_SHA256 = "c47bb7202b81f76c99125b9d5aeab145b5187fcc5c44393482f9fc0f23cb3d6b"
SET_PCA = ARTIFACT_ROOT / "ocoot/pbmc_control_pca64_div10_v2.npz"
SET_PCA_SHA256 = "bf123768469db7d084339f039a268c1b41055f9d65089c354a6a45c02e58045e"
SHARDED_PROTOCOL = ROOT / "configs/pbmc_ctxw256_trainonly_hs3_14k_full_test_sharded_sampling_v1.json"
RUN_RE = re.compile(rf"^{RUN}_pbmc_full_test_sampleseed(42|20260811)_shard(0[0-6])$")
MICRO_BATCH = 512


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


def audit_donoraware_cfg(cfg: omegaconf.DictConfig) -> tuple[dict, dict]:
    protocol = base.load_protocol()
    match = RUN_RE.fullmatch(str(cfg.run_name))
    if match is None:
        raise AssertionError("PBMC donor-aware run_name is outside the frozen grid")
    seed, shard_text = map(int, match.groups())
    shard_id = int(cfg.sampling.pbmc_shard_id)
    if shard_id != shard_text or int(cfg.optimization.seed) != seed:
        raise AssertionError("PBMC shard identity disagrees with run_name")
    if str(cfg.sampling.split) != "test" or list(map(str, cfg.data.evaluation_splits)) != ["test"]:
        raise AssertionError("PBMC donor-aware evaluation is test-only")
    if int(cfg.data.use_cell_set) != 1:
        raise AssertionError("PBMC full test requires use_cell_set=1")
    for key in ("num_sampled_batches", "fixed_cells_per_perturbation", "max_perturbations"):
        if getattr(cfg.sampling, key) is not None:
            raise AssertionError(f"formal PBMC sampling forbids {key}")
    if bool(cfg.sampling.pbmc_append_controls):
        raise AssertionError("treated shards must not append PBS")
    if str(cfg.data.data_name) != "PBMCFinetune":
        raise AssertionError("wrong PBMC dataset")
    if str(cfg.data.pert_col) != "cytokine" or str(cfg.data.cell_line_key) != "cell_type":
        raise AssertionError("PBMC observation keys changed")
    if str(cfg.data.perturbseq_batch_col) != "donor" or str(cfg.data.control_pert) != "PBS":
        raise AssertionError("PBMC donor/control contract changed")
    if str(cfg.data.meta_batch_parent_grouping) != "donor_celltype":
        raise AssertionError("donor-aware sampling requires donor_celltype batches")
    if str(cfg.data.control_bank_grouping) != "donor_celltype":
        raise AssertionError("donor-aware sampling requires donor_celltype control banks")
    if list(map(str, cfg.data.holdout_batches)) != protocol["heldout_donors"]:
        raise AssertionError("PBMC held-out donors changed")
    if list(map(str, cfg.data.holdout_pert.test)) != protocol["official_test_cytokines"]:
        raise AssertionError("PBMC full-test cytokines changed")
    if bool(cfg.data.skip_cached_indices):
        raise AssertionError("PBMC sampling must consume frozen grouped caches")
    if not same_path(cfg.model_checkpoint_path, FINAL):
        raise AssertionError("wrong donor-aware final export")
    if not same_path(cfg.sampling.pbmc_protocol_path, base.PROTOCOL):
        raise AssertionError("wrong PBMC protocol")
    if not same_path(cfg.data.control_distribution_context_path, CONTEXT):
        raise AssertionError("wrong donor-celltype Child context")
    shard = protocol["shards"][shard_id]
    if int(cfg.sampling.pbmc_expected_treated_cells) != int(shard["expected_treated_cells"]):
        raise AssertionError("PBMC shard treated-cell count changed")
    if int(cfg.sampling.pbmc_expected_nonempty_pairs) != int(shard["expected_nonempty_pairs"]):
        raise AssertionError("PBMC shard pair count changed")
    for path, expected in (
        (FINAL, FINAL_SHA256), (TRAIN_CONFIG, TRAIN_CONFIG_SHA256), (CONTEXT, CONTEXT_SHA256),
        (RESERVOIR, RESERVOIR_SHA256), (PARENT, PARENT_SHA256),
        (SET_PCA, SET_PCA_SHA256), (base.GROUP_CACHE, base.GROUP_CACHE_SHA256),
        (base.COUNT_CACHE, base.COUNT_CACHE_SHA256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"frozen donor-aware PBMC artifact changed: {path}")
    base.allow_omegaconf_checkpoint_unpickling()
    checkpoint = base.torch.load(FINAL, map_location="cpu", weights_only=True, mmap=True)
    if int(checkpoint.get("global_step", -1)) != 14_000 or int(checkpoint.get("epoch", -1)) != 2:
        raise AssertionError("donor-aware export training position changed")
    state = as_mapping(checkpoint.get("state_dict"), "state_dict")
    if len(state) != 643:
        raise AssertionError("donor-aware state_dict size changed")
    model = as_mapping(checkpoint["hyper_parameters"]["model_cfg"], "model_cfg")
    expected_model = {
        "model_type": "parent_residual_gene_dit",
        "parent_residual_enabled": True,
        "parent_residual_locked_flow_enabled": True,
        "parent_residual_matched_set_enabled": True,
        "parent_residual_parent_grouping": "donor_celltype",
        "parent_residual_real_control_grouping": "donor_celltype",
        "parent_residual_donor_aware_delta_enabled": True,
        "parent_residual_donor_aware_max_delta_rms_ratio": 0.75,
        "parent_residual_anchor_capacity": 128,
        "parent_residual_num_modules": 128,
        "parent_residual_context_hidden_dim": 256,
        "parent_residual_semi_balanced_ot_enabled": True,
        "parent_residual_semi_balanced_ot_epsilon": 0.1,
        "parent_residual_semi_balanced_ot_rho": 1.0,
        "parent_residual_semi_balanced_ot_iterations": 50,
        "parent_residual_de_aware_enabled": True,
        "parent_residual_hurdle_geometry_enabled": True,
        "parent_residual_sparse_support_enabled": True,
        "parent_residual_hurdle_alpha": 1.0,
    }
    for key, expected in expected_model.items():
        if model.get(key, object()) != expected:
            raise AssertionError(f"donor-aware PBMC mechanism mismatch: {key}")
    expected_paths = {
        "parent_residual_artifact_path": (PARENT, PARENT_SHA256),
        "parent_residual_set_pca_path": (SET_PCA, SET_PCA_SHA256),
        "parent_residual_real_control_reservoir_path": (RESERVOIR, RESERVOIR_SHA256),
    }
    for key, (path, expected_sha) in expected_paths.items():
        if not same_path(model.get(key), path) or model.get(key.replace("_path", "_sha256")) != expected_sha:
            raise AssertionError(f"donor-aware checkpoint {key} changed")
    return protocol, shard


# Rebind every model/artifact identity consumed by the hardened sharded runtime.
base.RUN = RUN
base.FINAL = FINAL
base.FINAL_SHA256 = FINAL_SHA256
base.CONTEXT = CONTEXT
base.CONTEXT_SHA256 = CONTEXT_SHA256
base.RESERVOIR = RESERVOIR
base.RESERVOIR_SHA256 = RESERVOIR_SHA256
base.RUN_RE = RUN_RE
base.audit_cfg = audit_donoraware_cfg
impl.SHARDED_PROTOCOL = SHARDED_PROTOCOL
impl.SHARDED_PROTOCOL_SHA256 = sha256(SHARDED_PROTOCOL)
impl.PARENT = PARENT
impl.PARENT_SHA256 = PARENT_SHA256
impl.CONTEXT_NPZ = CONTEXT
impl.CONTEXT_NPZ_SHA256 = CONTEXT_SHA256
impl.SET_PCA = SET_PCA
impl.SET_PCA_SHA256 = SET_PCA_SHA256
impl.MICRO_BATCH = MICRO_BATCH

_audit_sharded_contract = impl.audit_p0_cfg


def audit_donoraware_contract(cfg, shard: dict) -> dict:
    record = _audit_sharded_contract(cfg, shard)
    overlay = json.loads(SHARDED_PROTOCOL.read_text())
    if overlay.get("sampling_protocol") != "sharded_sampling_v1":
        raise AssertionError("wrong PBMC sharded sampling protocol")
    if int(overlay.get("shard_count", -1)) != 7 or overlay.get("gpu_map") != list(range(7)):
        raise AssertionError("frozen logical PBMC shard grid changed")
    record.update({
        "model_run": RUN,
        "model_global_step": 14_000,
        "model_state_dict_entries": 643,
        "model_parent_grouping": "donor_celltype",
        "model_final_export_sha256": FINAL_SHA256,
        "model_train_config_sha256": TRAIN_CONFIG_SHA256,
    })
    return record


def setup_dataset_with_donoraware_lock(datamodule, base_seed: int) -> dict:
    lock_path = ROOT / "logs/pbmc_donoraware_hs3_14k_full_test" / f"seed{int(base_seed)}" / "shared_dataset_setup.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    wait_started = time.monotonic()
    with lock_path.open("a+") as lock_handle:
        impl.fcntl.flock(lock_handle.fileno(), impl.fcntl.LOCK_EX)
        acquired = time.monotonic()
        try:
            datamodule.setup_dataset()
        finally:
            impl.fcntl.flock(lock_handle.fileno(), impl.fcntl.LOCK_UN)
    released = time.monotonic()
    return {
        "schema_version": 1, "passed": True, "path": str(lock_path.resolve()),
        "mechanism": "fcntl.flock_LOCK_EX", "scope": "datamodule.setup_dataset",
        "wait_seconds": acquired - wait_started, "held_seconds": released - acquired,
        "released_before_model_load_and_sampling": True, "model_run": RUN,
    }


impl.audit_p0_cfg = audit_donoraware_contract
impl.setup_dataset_with_exclusive_lock = setup_dataset_with_donoraware_lock


@impl.hydra.main(version_base=None, config_path="../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg) -> None:
    impl.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
