import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(os.environ.get("CHARM_RESOURCE_ROOT", str(ROOT / "resources")))
WORK_ROOT = ROOT
CELL_DETR = ROOT
RUNTIME_ROOT = Path(os.environ.get("CHARM_RUNTIME_ROOT", str(ROOT / "runs")))
DATA_ROOT = Path(os.environ.get("CHARM_DATA_ROOT", str(RESOURCE_ROOT / "data/PerturbDiff_data/perturb_data")))
ARTIFACT_ROOT = Path(os.environ.get("CHARM_ARTIFACT_ROOT", str(RESOURCE_ROOT / "artifacts")))
CKPT_ROOT = RUNTIME_ROOT / "legacy_checkpoints"
CACHE_ROOT = RUNTIME_ROOT / "cache"
LOG_ROOT = RUNTIME_ROOT / "legacy_logs"
OUTPUT_ROOT = ROOT / "runs"
RUNS_ROOT = ROOT / "runs"
RUN_INDEX_ROOT = RUNTIME_ROOT / "indexes"
PERTURB_DATA_ROOT = DATA_ROOT
SAMPLER = ROOT / "scripts/pbmc/sample_model.py"
MODEL = ROOT / "weights/pbmc.pt"
PROTOCOL = ROOT / "configs/pbmc_u2_attempt2_full_test_v1.json"
SHARDED_PROTOCOL = ROOT / "configs/pbmc_ctxw256_trainonly_hs3_14k_full_test_sharded_sampling_v1.json"
CONTEXT = ARTIFACT_ROOT / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_context_v1.npz"
RESERVOIR = ARTIFACT_ROOT / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_real8_v1.npz"
OFFICIAL_CACHE = PERTURB_DATA_ROOT / "indices_cache"
RUN = "pbmc_donoraware_hs3_14k_s42_v2_boundfix1_formal14000"
BASE_SEED = 42


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--shard-id", type=int, choices=range(7), required=True)
    value.add_argument("--audit-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.run_root.is_absolute():
        raise ValueError("--run-root must be absolute")
    protocol = json.loads(PROTOCOL.read_text())
    shard = protocol["shards"][args.shard_id]
    output = args.run_root / "shards" / f"shard{args.shard_id:02d}"
    output.mkdir(parents=True, exist_ok=True)
    label = f"{RUN}_pbmc_full_test_sampleseed{BASE_SEED}_shard{args.shard_id:02d}"
    overrides = [
        "trainer.use_distributed_sampler=false", "data.normalize_counts=10",
        "data.num_workers=2", "data.prefetch_factor=2", "data.persistent_workers=false",
        "lightning.ema.decay=0.99", "lightning.ema.update_steps=10", "path=local_path",
        "cov_encoding=trixie_onehot", "cov_encoding.batch_encoding=onehot",
        "model.p_drop_control=0", "data.keep_control_cell=false", "sampling.use_ddim=true",
        "data.pad_length=2000", "model.hidden_num=[2000,512]", "model.input_dim=2000",
        "data.embed_key=X_hvg", f"model_checkpoint_path={MODEL}", "data=pbmc_finetune",
        "cov_encoding.celltype_encoding=llm", "cov_encoding.replogle_gene_encoding=onehot",
        "data.sample_pbmc_only=true", "data.use_cell_set=1", "+data.evaluation_splits=[test]",
        f"data.indices_cache_dir={OFFICIAL_CACHE}", "data.skip_cached_indices=false",
        "+data.meta_batch_parent_grouping=donor_celltype",
        "+data.control_bank_grouping=donor_celltype",
        f"+data.control_distribution_context_path={CONTEXT}",
        "+data.control_context_mode=latent_cluster_mixture", "+data.control_bank_path=null",
        "+data.control_members_per_token=1", "+data.latent_control_candidate_strategy=all",
        "+data.latent_control_top_k=0", "+data.latent_control_fixed_max_k=true",
        "+data.latent_control_use_cluster_sets=false", "+data.cell_detr_unique_control_bank=true",
        "+data.hungarian_flow_combination_sampler=false",
        f"+model.parent_residual_real_control_reservoir_path={RESERVOIR}",
        "+model.parent_residual_real_control_reservoir_sha256=c7273173a5d2a9cfb7664b366f853b4ad431bfb60b9783921cd79c06cd5d88f5",
        "optimization.micro_batch_size=512", f"optimization.seed={BASE_SEED}",
        "sampling.split=test", "sampling.num_sampled_batches=null", "sampling.max_perturbations=null",
        "sampling.fixed_cells_per_perturbation=null", "sampling.guidance_strength=0.0",
        "sampling.progress=false", "sampling.initial_state=parent_residual",
        f"sampling.output_dir={output}", "++sampling.parent_locked_max_singleton_fraction=0.01",
        "++sampling.parent_locked_endpoint_mean_drift_tolerance=1e-5",
        f"+sampling.pbmc_shard_id={args.shard_id}",
        f"+sampling.pbmc_expected_treated_cells={shard['expected_treated_cells']}",
        f"+sampling.pbmc_expected_nonempty_pairs={shard['expected_nonempty_pairs']}",
        "+sampling.pbmc_append_controls=false", f"+sampling.pbmc_protocol_path={PROTOCOL}",
        f"+sampling.pbmc_sharded_protocol_path={SHARDED_PROTOCOL}",
        "+sampling.pbmc_rng_protocol=sharded_sampling_v1", f"run_name={label}",
        "lightning.logger._target_=pytorch_lightning.loggers.logger.DummyLogger",
        "~lightning.logger.project", "~lightning.logger.save_dir", "~lightning.logger.name",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "WORK_ROOT": str(RESOURCE_ROOT),
            "RUNTIME_ROOT": str(RUNTIME_ROOT),
            "PERTURB_RUNTIME_ROOT": str(RUNTIME_ROOT),
            "DATA_ROOT": str(DATA_ROOT),
            "ARTIFACT_ROOT": str(ARTIFACT_ROOT),
            "CKPT_ROOT": str(CKPT_ROOT),
            "CACHE_ROOT": str(CACHE_ROOT),
            "LOG_ROOT": str(LOG_ROOT),
            "OUTPUT_ROOT": str(OUTPUT_ROOT),
            "RUNS_ROOT": str(RUNS_ROOT),
            "RUN_INDEX_ROOT": str(RUN_INDEX_ROOT),
            "PERTURB_DATA_ROOT": str(PERTURB_DATA_ROOT),
        }
    )
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = ":".join(
        [
            str(CELL_DETR),
            str(WORK_ROOT),
            *([inherited_pythonpath] if inherited_pythonpath else []),
        ]
    )
    if args.audit_only:
        environment["PBMC_SHARD_AUDIT_ONLY"] = "true"
    completed = subprocess.run([sys.executable, str(SAMPLER), *overrides], cwd=WORK_ROOT, env=environment)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
