from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(os.environ.get("CHARM_RESOURCE_ROOT", str(ROOT / "resources")))
DATA_ROOT = Path(os.environ.get("CHARM_DATA_ROOT", str(RESOURCE_ROOT / "data/PerturbDiff_data/perturb_data")))
ARTIFACT_ROOT = Path(os.environ.get("CHARM_ARTIFACT_ROOT", str(RESOURCE_ROOT / "artifacts")))
CELL_DETR = ROOT
WORK_ROOT = ROOT
RUNTIME_ROOT = Path(os.environ.get("CHARM_RUNTIME_ROOT", str(ROOT / "runs")))
PYTHON = Path(os.environ.get("CHARM_TRAIN_PYTHON", sys.executable))
EVAL_PYTHON = Path(os.environ.get("CHARM_EVAL_PYTHON", sys.executable))
CELL_EVAL = Path(os.environ.get("CELL_EVAL_CLI", "cell-eval"))

SAMPLER = ROOT / "scripts/pbmc/sample_shard.py"
HARDENED_SAMPLER = ROOT / "scripts/pbmc/sample_model.py"
SHARDED_RUNTIME = ROOT / "scripts/pbmc/shard_runtime.py"
SHARD_BASE = ROOT / "scripts/pbmc/shard_model.py"
EVALUATOR = ROOT / "scripts/pbmc/evaluate_by_celltype.py"
PROTOCOL = ROOT / "configs/pbmc_u2_attempt2_full_test_v1.json"
CONTROLS = DATA_ROOT / "tmp_pbmc_ctrl.h5ad"
BULK_EVALUATOR = ROOT / "scripts/pbmc/evaluate_pbmc_table4_pooled_bulk.py"
POOLED_BUILDER = ROOT / "scripts/pbmc/build_pbmc_pooled_inputs.py"
POOLED_BUILDER_IMPL = ROOT / "src/evaluation/pbmc_pooled_builder.py"
POOLED_DE = ROOT / "scripts/pbmc/audit_pbmc_pooled_de.py"
POOLED_DE_IMPL = ROOT / "src/evaluation/pbmc_pooled_de.py"
PHYSICAL_IDENTITY = ROOT / "src/evaluation/pbmc_physical_identity.py"
PROVENANCE_ATTESTATION = ROOT / "src/evaluation/pbmc_provenance_attestation.py"
POSTPROCESS_CONTRACT = ROOT / "scripts/pbmc/pbmc_postprocess_contract.py"
RECOVER_CELLEVAL = ROOT / "scripts/pbmc/recover_pbmc_pooled_celleval_exact.py"

MODEL_EXPORT = ROOT / "weights/pbmc.pt"
TRAIN_CONFIG = ROOT / "configs/frozen/pbmc_train.yaml"
PARENT_ARTIFACT = ARTIFACT_ROOT / "parent_residual_gene_dit/pbmc_donor_celltype_parent_residual_ocoot_v1.npz"
CONTEXT = ARTIFACT_ROOT / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_context_v1.npz"
RESERVOIR = ARTIFACT_ROOT / "ocoot/pbmc_donor_celltype_k128_occ64_minsupport20_real8_v1.npz"

EXPECTED_MODEL_SHA256 = "a083cff84245b644ebf0e67d7cf0da6963ac77467edb2f7d511ee0d6bed610ab"
EXPECTED_CONFIG_SHA256 = "be8f0c6fa869a3daa58288ff0aa81ed627e78b338c805e3b9565d161e35e6c58"
EXPECTED_PARENT_SHA256 = "c47bb7202b81f76c99125b9d5aeab145b5187fcc5c44393482f9fc0f23cb3d6b"
EXPECTED_CONTEXT_SHA256 = "dd91bcb3aae06d70a4591ff7c0b4cc1f93038f5fa3b9199bd9515cd6408b0e6d"
EXPECTED_RESERVOIR_SHA256 = "c7273173a5d2a9cfb7664b366f853b4ad431bfb60b9783921cd79c06cd5d88f5"
EXPECTED_PROTOCOL_SHA256 = "7512c533971b85344f1a2f13a7b84af0729c46eddaf41e4e643482d5a331a976"
EXPECTED_TREATED = 2_260_453
EXPECTED_CONTROLS = 235_478
EXPECTED_TOTAL = 2_495_931
EXPECTED_GENES = 2_000
EXPECTED_TARGETS = 62
EXPECTED_STAGES = 18
SAMPLE_SEED = 42
PARENT_PARTITION_POLICY = "strict_unsplit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(resolved)
    value: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if hash_file:
        value["sha256"] = sha256(resolved)
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    text = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace differing record: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != text:
            raise
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another PBMC evaluation owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_logged(
    command: Sequence[str], log_path: Path, *, env: dict[str, str] | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        header = (
            f"\n[{dt.datetime.now(dt.timezone.utc).isoformat()}] command="
            + json.dumps(list(command))
            + "\n"
        ).encode()
        handle.write(header)
        completed = subprocess.run(
            list(command), cwd=WORK_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}; see {log_path}"
        )


def require_empty_or_absent(path: Path, *, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing partial or unbound {label} output: {path}")


def validate_model_inputs() -> None:
    required = [
        PYTHON,
        EVAL_PYTHON,
        CELL_EVAL,
        SAMPLER,
        HARDENED_SAMPLER,
        SHARDED_RUNTIME,
        SHARD_BASE,
        EVALUATOR,
        PROTOCOL,
        CONTROLS,
        BULK_EVALUATOR,
        POOLED_BUILDER,
        POOLED_BUILDER_IMPL,
        POOLED_DE,
        POOLED_DE_IMPL,
        PHYSICAL_IDENTITY,
        PROVENANCE_ATTESTATION,
        POSTPROCESS_CONTRACT,
        RECOVER_CELLEVAL,
        MODEL_EXPORT,
        TRAIN_CONFIG,
        PARENT_ARTIFACT,
        CONTEXT,
        RESERVOIR,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if sha256(MODEL_EXPORT) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("donor-aware final export SHA256 changed")
    if sha256(TRAIN_CONFIG) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("donor-aware resolved training config SHA256 changed")
    for path, expected in (
        (PARENT_ARTIFACT, EXPECTED_PARENT_SHA256),
        (CONTEXT, EXPECTED_CONTEXT_SHA256),
        (RESERVOIR, EXPECTED_RESERVOIR_SHA256),
        (PROTOCOL, EXPECTED_PROTOCOL_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen PBMC dependency SHA256 changed: {path}")


def freeze_pipeline(run_root: Path, gpu_ids: list[int]) -> None:
    assets = [
        Path(__file__), SAMPLER, HARDENED_SAMPLER, SHARDED_RUNTIME, SHARD_BASE,
        CELL_DETR / "src/models/lightning/lightning_factories.py",
        CELL_DETR / "src/models/donor_aware_parent_delta/artifact_model.py",
        CELL_DETR / "src/models/donor_aware_parent_delta/head.py",
        CELL_DETR / "src/models/donor_aware_parent_delta/config.py",
        CELL_DETR / "src/apps/sampling/sampling_generation.py",
        CELL_DETR / "src/apps/sampling/sampling_setup.py",
        CELL_DETR / "src/apps/sampling/sampling_provenance.py",
        CELL_DETR / "src/data/dataset/dataset_core.py",
        CELL_DETR / "src/evaluation/pbmc_physical_identity.py",
        EVALUATOR, PROTOCOL, BULK_EVALUATOR,
        POOLED_BUILDER, POOLED_BUILDER_IMPL, POOLED_DE, POOLED_DE_IMPL,
        PHYSICAL_IDENTITY, PROVENANCE_ATTESTATION, POSTPROCESS_CONTRACT,
        RECOVER_CELLEVAL, MODEL_EXPORT, TRAIN_CONFIG,
        PARENT_ARTIFACT, CONTEXT, RESERVOIR, CONTROLS,
    ]
    freeze = {
        "schema_version": 1,
        "passed": True,
        "kind": "pbmc_donoraware_hs3_v2_boundfix1_full_evaluation/v1",
        "sampling_seed": SAMPLE_SEED,
        "logical_shards": list(range(7)),
        "runtime_gpu_map": {str(i): gpu_ids[i % len(gpu_ids)] for i in range(7)},
        "derived_seeds": {
            str(i): (SAMPLE_SEED + 1_000_003 * (i + 1)) % 4_294_967_295
            for i in range(7)
        },
        "parent_partition_policy": PARENT_PARTITION_POLICY,
        "assets": {
            str(path.resolve(strict=True)): {
                "sha256": sha256(path), "size_bytes": path.stat().st_size
            }
            for path in assets
        },
    }
    freeze_path = run_root / "logs/audit/postprocess_protocol_freeze_v1.json"
    write_json_once(freeze_path, freeze)


def shard_files(shard_root: Path) -> tuple[Path, Path]:
    preds = sorted(shard_root.glob("diffusion_predict_*.h5ad")) if shard_root.exists() else []
    reals = sorted(shard_root.glob("diffusion_true_*.h5ad")) if shard_root.exists() else []
    if len(preds) != 1 or len(reals) != 1:
        raise RuntimeError(
            f"expected exactly one prediction/truth pair in {shard_root}; "
            f"found pred={len(preds)} real={len(reals)}"
        )
    return preds[0], reals[0]


def validate_shard_record(path: Path, shard_id: int, pred: Path, real: Path) -> None:
    value = load_json(path)
    expected = {
        "schema_version": 2,
        "passed": True,
        "kind": "pbmc_u2_attempt2_treated_shard_audit",
        "shard_id": shard_id,
        "sampling_seed": SAMPLE_SEED,
        "control_cells": 0,
        "genes": EXPECTED_GENES,
        "parent_gate_passed": True,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"stale shard record {path}: {key}")
    if Path(value["pred"]).resolve() != pred.resolve():
        raise RuntimeError(f"shard prediction path changed: {path}")
    if Path(value["real"]).resolve() != real.resolve():
        raise RuntimeError(f"shard truth path changed: {path}")
    if Path(value.get("protocol", "")).resolve() != PROTOCOL.resolve():
        raise RuntimeError(f"shard protocol path changed: {path}")
    if value.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"shard protocol SHA256 changed: {path}")
    if sha256(pred) != value.get("pred_sha256"):
        raise RuntimeError(f"shard prediction changed after audit: {pred}")
    if sha256(real) != value.get("real_sha256"):
        raise RuntimeError(f"shard truth changed after audit: {real}")
    identity = value.get("physical_identity", {})
    if identity.get("passed") is not True:
        raise RuntimeError(f"shard physical identity is not proven: {path}")


def sample_and_audit_shard(run_root: Path, shard_id: int, gpu: int) -> Path:
    shard_root = run_root / "shards" / f"shard{shard_id:02d}"
    status = run_root / "logs/status" / f"shard{shard_id:02d}.sample.json"
    sample_log = run_root / "logs/sample" / f"shard{shard_id:02d}.log"
    existing = list(shard_root.iterdir()) if shard_root.exists() else []
    if not existing:
        shard_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(CELL_DETR),
                "HYDRA_FULL_ERROR": "1",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
            }
        )
        run_logged(
            [
                str(PYTHON), str(SAMPLER), "--run-root", str(run_root),
                "--shard-id", str(shard_id),
            ],
            sample_log,
            env=env,
        )
    pred, real = shard_files(shard_root)
    filter_audit = shard_root / "pbmc_shard_filter_audit.json"
    if not filter_audit.is_file() or load_json(filter_audit).get("passed") is not True:
        raise RuntimeError(f"missing or failed shard filter audit: {filter_audit}")
    if status.exists():
        validate_shard_record(status, shard_id, pred, real)
        return status
    run_logged(
        [
            str(EVAL_PYTHON), str(EVALUATOR), "audit-shard",
            "--protocol", str(PROTOCOL), "--shard-id", str(shard_id),
            "--sampling-seed", str(SAMPLE_SEED), "--pred", str(pred),
            "--real", str(real), "--output", str(status),
        ],
        run_root / "logs/audit_shard" / f"shard{shard_id:02d}.log",
    )
    validate_shard_record(status, shard_id, pred, real)
    return status


def run_shard_queue(run_root: Path, gpu: int, shard_ids: list[int]) -> list[Path]:
    return [sample_and_audit_shard(run_root, shard_id, gpu) for shard_id in shard_ids]


def validate_merge(run_root: Path, records: list[Path]) -> bool:
    path = run_root / "by_cell_type/merge_manifest.json"
    if not path.exists():
        return False
    value = load_json(path)
    expected = {
        "schema_version": 2, "passed": True,
        "kind": "pbmc_u2_attempt2_celltype_merge",
        "sampling_seed": SAMPLE_SEED,
        "parent_partition_policy": PARENT_PARTITION_POLICY,
        "treated_cells": EXPECTED_TREATED, "control_cells": EXPECTED_CONTROLS,
        "total_cells": EXPECTED_TOTAL, "genes": EXPECTED_GENES,
        "cell_types": EXPECTED_STAGES,
        "all_celltype_parent_gates_passed": True,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"stale merge manifest {path}: {key}")
    if [Path(x).resolve() for x in value.get("shard_records", [])] != [
        x.resolve() for x in records
    ]:
        raise RuntimeError("merge manifest is bound to different shard records")
    for stage in value.get("stages", []):
        for side in ("pred", "real"):
            stage_path = Path(stage[side])
            if not stage_path.is_file() or sha256(stage_path) != stage[f"{side}_sha256"]:
                raise RuntimeError(f"merged stage changed: {stage_path}")
    return True


def merge_shards(run_root: Path, records: list[Path]) -> None:
    if validate_merge(run_root, records):
        return
    output = run_root / "by_cell_type"
    require_empty_or_absent(output, label="merge")
    command = [
        str(EVAL_PYTHON), str(EVALUATOR), "merge", "--protocol", str(PROTOCOL),
    ]
    for path in records:
        command.extend(["--shard-record", str(path)])
    command.extend(
        [
            "--controls", str(CONTROLS), "--sampling-seed", str(SAMPLE_SEED),
            "--output-dir", str(output), "--parent-partition-policy",
            PARENT_PARTITION_POLICY,
        ]
    )
    run_logged(command, run_root / "logs/merge.log")
    if not validate_merge(run_root, records):
        raise RuntimeError("merge completed without a valid manifest")


def post_contract(run_root: Path, action: str, stage: str, threads: int) -> dict[str, Any]:
    subprocess.run(
        [
            str(EVAL_PYTHON), str(POSTPROCESS_CONTRACT), action, stage,
            "--run-root", str(run_root), "--threads", str(threads),
        ],
        cwd=WORK_ROOT,
        check=True,
    )
    return load_json(run_root / "logs/audit" / f"postprocess_{stage}_contract_v1.json")


def run_bound_stage(
    run_root: Path, stage: str, output: Path, command: Sequence[str], threads: int
) -> None:
    state = post_contract(run_root, "start", stage, threads)
    if state.get("phase") == "complete":
        return
    run_logged(command, run_root / "logs/postprocess" / f"{stage}.log")
    state = post_contract(run_root, "complete", stage, threads)
    if state.get("phase") != "complete" or not output.exists():
        raise RuntimeError(f"postprocess stage did not seal: {stage}")


def postprocess(run_root: Path, threads: int) -> None:
    run_bound_stage(
        run_root,
        "bulk",
        run_root / "pooled_bulk/PASS",
        [
            str(EVAL_PYTHON), str(BULK_EVALUATOR), "--input-root",
            str(run_root / "by_cell_type"), "--output-dir",
            str(run_root / "pooled_bulk"), "--chunk-rows", "8192",
        ],
        threads,
    )
    run_bound_stage(
        run_root,
        "builder",
        run_root / "pooled_inputs/builder_manifest.json",
        [
            str(EVAL_PYTHON), str(POOLED_BUILDER), "--merge-manifest",
            str(run_root / "by_cell_type/merge_manifest.json"), "--output-dir",
            str(run_root / "pooled_inputs"), "--max-loaded-elems", "25000000",
        ],
        threads,
    )
    run_bound_stage(
        run_root,
        "de",
        run_root / "pooled_de/de_cache_manifest.json",
        [
            str(EVAL_PYTHON), str(POOLED_DE),
            "--real", str(run_root / "pooled_inputs/real_pooled.h5ad"),
            "--pred", str(run_root / "pooled_inputs/pred_pooled.h5ad"),
            "--output-dir", str(run_root / "pooled_de"),
            "--group-key", "cytokine", "--reference", "PBS",
            "--identity-columns", "source_namespace", "source_dataset_name",
            "source_physical_row", "source_obs_name", "source_role",
            "--identity-provenance-manifest",
            str(run_root / "pooled_inputs/physical_identity_provenance.json"),
            "--stage-manifest-root", str(run_root / "by_cell_type"),
            "--expected-targets-file", str(PROTOCOL),
            "--expected-target-count", str(EXPECTED_TARGETS),
            "--expected-stage-count", str(EXPECTED_STAGES),
            "--expected-total-cells", str(EXPECTED_TOTAL),
            "--expected-treated-cells", str(EXPECTED_TREATED),
            "--expected-reference-cells", str(EXPECTED_CONTROLS),
            "--expected-genes", str(EXPECTED_GENES),
            "--gene-chunk-size", "128", "--row-chunk-size", "4096",
            "--alternative", "two-sided", "--method", "auto",
            "--use-continuity", "--is-log1p", "--exp-post-agg",
            "--fold-change-clip", "20",
        ],
        threads,
    )

    state = post_contract(run_root, "start", "celleval", threads)
    if state.get("phase") != "complete":
        run_logged(
            [
                str(EVAL_PYTHON), str(RECOVER_CELLEVAL), "execute",
                "--run-root", str(run_root), "--work-root", str(WORK_ROOT),
                "--eval-python", str(EVAL_PYTHON), "--cell-eval-cli",
                str(CELL_EVAL), "--global-celleval-lock",
                str(RUNTIME_ROOT / "legacy_logs/parent_residual_gene_dit_eval/celleval.lock"),
                "--num-threads", str(threads), "--attempt-id", "attempt-001",
                "--publish-for-frozen-resume",
            ],
            run_root / "logs/postprocess/pooled_celleval_exact.log",
        )
    completion = run_root / "pooled_celleval_exact_v1/completion.json"
    if not completion.is_file() or load_json(completion).get("passed") is not True:
        raise RuntimeError("exact pooled CellEval did not complete")


def parse_gpu_ids(text: str) -> list[int]:
    try:
        values = [int(item) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gpu-ids must be comma-separated integers") from exc
    if not values or len(values) > 7 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("gpu-ids must be 1..7 unique IDs")
    if any(value < 0 or value > 7 for value in values):
        raise argparse.ArgumentTypeError("GPU IDs must be in [0,7]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3,4,5"))
    parser.add_argument("--eval-threads", type=int, default=32)
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Freeze and validate dependencies without sampling or postprocessing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_root.is_absolute():
        raise ValueError("--run-root must be absolute")
    if args.eval_threads < 1:
        raise ValueError("--eval-threads must be positive")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run_root / "logs/pipeline.lock"):
        validate_model_inputs()
        freeze_pipeline(run_root, args.gpu_ids)
        if args.audit_only:
            print(canonical_json({"passed": True, "audit_only": True, "run_root": str(run_root)}))
            return 0
        queues = {gpu: [] for gpu in args.gpu_ids}
        for shard_id in range(7):
            queues[args.gpu_ids[shard_id % len(args.gpu_ids)]].append(shard_id)
        records: list[Path] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queues)) as executor:
            futures = {
                executor.submit(run_shard_queue, run_root, gpu, shard_ids): gpu
                for gpu, shard_ids in queues.items()
            }
            for future in concurrent.futures.as_completed(futures):
                records.extend(future.result())
        records.sort(key=lambda path: path.name)
        merge_shards(run_root, records)
        postprocess(run_root, args.eval_threads)
        completion = {
            "schema_version": 1,
            "passed": True,
            "kind": "pbmc_donoraware_hs3_v2_boundfix1_full_evaluation_complete/v1",
            "run_root": str(run_root),
            "sampling_seed": SAMPLE_SEED,
            "gpu_ids": args.gpu_ids,
            "shard_records": [record(path) for path in records],
            "merge_manifest": record(run_root / "by_cell_type/merge_manifest.json"),
            "bulk_manifest": record(run_root / "pooled_bulk/audit_manifest.json"),
            "pooled_builder_manifest": record(run_root / "pooled_inputs/builder_manifest.json"),
            "pooled_de_manifest": record(run_root / "pooled_de/de_cache_manifest.json"),
            "pooled_celleval_completion": record(
                run_root / "pooled_celleval_exact_v1/completion.json"
            ),
        }
        write_json_once(run_root / "PIPELINE_COMPLETE.json", completion)
        print(canonical_json(completion))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
