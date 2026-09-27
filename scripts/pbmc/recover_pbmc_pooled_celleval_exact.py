from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence


EXPECTED_ENVIRONMENT = {
    "anndata": "0.12.19",
    "cell-eval": "0.6.6",
    "pdex": "0.1.26",
    "polars": "1.42.1",
    "scipy": "1.17.1",
}
EXPECTED_COVERAGE = {
    "genes": 2000,
    "reference_rows": 235478,
    "rows": 2495931,
    "stages": 18,
    "treated_rows": 2260453,
}
EXPECTED_PERTURBATIONS = 62
EXPECTED_DE_ROWS = 124000
REQUIRED_RESULT_COLUMNS = (
    "perturbation",
    "overlap_at_N",
    "precision_at_N",
    "de_spearman_sig",
    "de_direction_match",
    "de_spearman_lfc_sig",
    "pr_auc",
    "roc_auc",
    "pearson_delta",
    "mse",
    "mae",
    "discrimination_score_l1",
    "discrimination_score_l2",
    "discrimination_score_cosine",
)
SKIP_METRICS = (
    "mse_delta",
    "mae_delta",
    "pearson_edistance",
    "clustering_agreement",
    "de_sig_genes_recall",
    "de_nsig_counts",
    "overlap_at_50",
    "overlap_at_100",
    "overlap_at_200",
    "overlap_at_500",
    "precision_at_50",
    "precision_at_100",
    "precision_at_200",
    "precision_at_500",
)
ATTEMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, digest: str | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Atomically create a JSON file; identical reruns are harmless."""

    text = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace differing artifact: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"racing artifact differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_lock(path: Path, *, blocking: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process holds recovery lock: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(resolved)
    return resolved


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_root = args.run_root.expanduser().resolve(strict=True)
    work_root = args.work_root.expanduser().resolve(strict=True)
    exact_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else run_root / "pooled_celleval_exact_v1"
    )
    if exact_root == run_root / "pooled_celleval":
        raise ValueError("exact output must not be the failed pooled_celleval directory")
    try:
        exact_root.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("exact output must be inside the PBMC run root") from exc
    runtime_root = work_root.parent / "runtime"
    return {
        "run_root": run_root,
        "work_root": work_root,
        "exact_root": exact_root,
        "audit_root": exact_root / "audit",
        "compat_root": run_root / "pooled_celleval",
        "pred": run_root / "pooled_inputs/pred_pooled.h5ad",
        "real": run_root / "pooled_inputs/real_pooled.h5ad",
        "builder_manifest": run_root / "pooled_inputs/builder_manifest.json",
        "de_manifest": run_root / "pooled_de/de_cache_manifest.json",
        "pred_de": run_root / "pooled_de/pred_de.csv",
        "real_de": run_root / "pooled_de/real_de.csv",
        "stage_contract": run_root
        / "logs/audit/postprocess_celleval_contract_v1.json",
        "suite_lock": run_root / "logs/postprocess/suite.lock",
        "recovery_lock": run_root
        / "logs/postprocess/pooled_celleval_exact_v1.lock",
        "global_celleval_lock": (
            args.global_celleval_lock.expanduser().resolve()
            if args.global_celleval_lock is not None
            else runtime_root / "legacy_logs/parent_residual_gene_dit_eval/celleval.lock"
        ),
        "contract_adapter": work_root
        / "scripts/pbmc/pbmc_postprocess_contract.py",
    }


def validate_builder_manifest(
    paths: dict[str, Path], *, verify_hashes: bool
) -> dict[str, dict[str, Any]]:
    manifest_path = require_file(paths["builder_manifest"])
    manifest = load_object(manifest_path)
    if manifest.get("passed") is not True:
        raise ValueError("pooled input builder manifest did not pass")
    if manifest.get("schema_version") != "pbmc-pooled-input-builder/v1":
        raise ValueError("unexpected pooled input builder schema")
    if manifest.get("coverage") != EXPECTED_COVERAGE:
        raise ValueError(f"unexpected pooled input coverage: {manifest.get('coverage')}")
    records: dict[str, dict[str, Any]] = {}
    for side in ("pred", "real"):
        expected = require_file(paths[side])
        stored = manifest.get(side, {})
        if not same_path(stored.get("path", ""), expected):
            raise ValueError(f"builder {side} path does not match the pooled input")
        digest = str(stored.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError(f"builder {side} SHA is malformed")
        if verify_hashes and sha256_file(expected) != digest:
            raise ValueError(f"pooled {side} H5AD changed after builder completion")
        records[side] = file_record(expected, digest=digest)
    records["builder_manifest"] = file_record(
        manifest_path,
        digest=sha256_file(manifest_path) if verify_hashes else sha256_file(manifest_path),
    )
    return records


def validate_de_manifest(
    paths: dict[str, Path], *, verify_hashes: bool
) -> dict[str, dict[str, Any]]:
    manifest_path = require_file(paths["de_manifest"])
    manifest = load_object(manifest_path)
    if manifest.get("passed") is not True or manifest.get("phase") != "complete":
        raise ValueError("pooled DE manifest is not complete")
    if manifest.get("schema_version") != "pbmc-pooled-de-cache/v1":
        raise ValueError("unexpected pooled DE manifest schema")
    if int(manifest.get("expected_tests_per_side", -1)) != EXPECTED_DE_ROWS:
        raise ValueError("unexpected pooled DE family size")
    records: dict[str, dict[str, Any]] = {}
    for side in ("pred", "real"):
        expected = require_file(paths[f"{side}_de"])
        stored = manifest.get("de", {}).get(side, {})
        if int(stored.get("rows", -1)) != EXPECTED_DE_ROWS:
            raise ValueError(f"unexpected {side} DE row count")
        if not same_path(stored.get("path", ""), expected):
            raise ValueError(f"DE manifest {side} path mismatch")
        if int(stored.get("bytes", -1)) != expected.stat().st_size:
            raise ValueError(f"DE manifest {side} size mismatch")
        digest = str(stored.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError(f"DE manifest {side} SHA is malformed")
        if verify_hashes and sha256_file(expected) != digest:
            raise ValueError(f"pooled {side} DE CSV changed after completion")
        records[f"{side}_de"] = file_record(expected, digest=digest)
    records["de_manifest"] = file_record(
        manifest_path,
        digest=sha256_file(manifest_path),
    )
    return records


def collect_environment(eval_python: Path, cell_eval_cli: Path) -> dict[str, Any]:
    eval_python = require_file(eval_python)
    cell_eval_cli = require_file(cell_eval_cli)
    probe = """
import importlib.metadata as metadata
import json
import pathlib
import cell_eval._evaluator
import cell_eval.utils
names = ["cell-eval", "pdex", "anndata", "scipy", "polars"]
print(json.dumps({
    "versions": {name: metadata.version(name) for name in names},
    "evaluator": str(pathlib.Path(cell_eval._evaluator.__file__).resolve()),
    "utils": str(pathlib.Path(cell_eval.utils.__file__).resolve()),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(eval_python), "-c", probe],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    if payload.get("versions") != EXPECTED_ENVIRONMENT:
        raise ValueError(
            f"frozen CellEval environment drift: {payload.get('versions')}"
        )
    return {
        "versions": payload["versions"],
        "eval_python": file_record(eval_python),
        "cell_eval_cli": file_record(cell_eval_cli),
        "cell_eval_evaluator": file_record(Path(payload["evaluator"])),
        "cell_eval_utils": file_record(Path(payload["utils"])),
    }


def validate_stage_contract(
    paths: dict[str, Path], input_records: dict[str, dict[str, Any]], threads: int
) -> dict[str, Any]:
    path = require_file(paths["stage_contract"])
    value = load_object(path)
    if value.get("schema_version") != 1 or value.get("phase") not in {
        "started",
        "complete",
    }:
        raise ValueError("pooled CellEval postprocess contract is not resumable")
    contract = value.get("contract", {})
    if contract.get("stage") != "celleval" or not same_path(
        contract.get("run_root", ""), paths["run_root"]
    ):
        raise ValueError("unexpected pooled CellEval postprocess contract")
    if contract.get("parameters") != {"sampling_seed": 42, "eval_threads": threads}:
        raise ValueError("recovery threads differ from the frozen stage contract")
    frozen_inputs = contract.get("inputs", {})
    for name in ("pred", "real", "pred_de", "real_de", "de_manifest"):
        record = input_records[name]
        frozen = frozen_inputs.get(record["path"])
        if frozen != {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}:
            raise ValueError(f"frozen stage input disagrees with current {name}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "phase": value["phase"],
        "fingerprint": contract.get("fingerprint"),
    }


def build_cell_eval_command(
    *,
    cell_eval_cli: Path,
    pred: Path,
    real: Path,
    pred_de: Path,
    real_de: Path,
    outdir: Path,
    threads: int,
) -> list[str]:
    return [
        str(cell_eval_cli),
        "run",
        "--adata-pred",
        str(pred),
        "--adata-real",
        str(real),
        "--de-pred",
        str(pred_de),
        "--de-real",
        str(real_de),
        "--control-pert",
        "PBS",
        "--pert-col",
        "cytokine",
        "--outdir",
        str(outdir),
        "--num-threads",
        str(threads),
        "--batch-size",
        "64",
        "--profile",
        "full",
        "--allow-discrete",
        "--skip-metrics",
        ",".join(SKIP_METRICS),
    ]


def validate_results(path: Path) -> dict[str, Any]:
    path = require_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_RESULT_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing pooled CellEval columns: {missing}")
        rows = list(reader)
    if len(rows) != EXPECTED_PERTURBATIONS:
        raise ValueError(f"expected 62 perturbations, found {len(rows)}")
    perturbations = [str(row["perturbation"]) for row in rows]
    if len(set(perturbations)) != len(perturbations):
        raise ValueError("pooled CellEval perturbations are not unique")
    if "PBS" in set(perturbations):
        raise ValueError("control PBS unexpectedly appears in pooled results")
    means: dict[str, float] = {}
    for column in REQUIRED_RESULT_COLUMNS[1:]:
        try:
            values = [float(row[column]) for row in rows]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric pooled CellEval metric: {column}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite pooled CellEval metric: {column}")
        means[column] = sum(values) / len(values)
    return {
        "record": file_record(path),
        "rows": len(rows),
        "headline_means": means,
    }


def build_input_contract(
    args: argparse.Namespace, paths: dict[str, Path], *, verify_hashes: bool
) -> dict[str, Any]:
    builder = validate_builder_manifest(paths, verify_hashes=verify_hashes)
    de = validate_de_manifest(paths, verify_hashes=verify_hashes)
    records = {**builder, **de}
    environment = collect_environment(args.eval_python, args.cell_eval_cli)
    stage = validate_stage_contract(paths, records, args.num_threads)
    return {
        "schema_version": 1,
        "kind": "pbmc_pooled_celleval_exact_allow_discrete_input/v1",
        "hashes_verified": verify_hashes,
        "policy": {
            "adata_pred": "immutable_original_pred_pooled_h5ad",
            "allow_discrete": True,
            "cell_eval_0_6_6_semantics": (
                "continuous input is classified as already log-normalized with "
                "validate=False and is not transformed; negative values remain invalid"
            ),
            "bounded_projection": False,
            "de_policy": "reuse_sealed_pooled_de_csvs_without_recomputation",
        },
        "inputs": records,
        "environment": environment,
        "postprocess_stage_contract": stage,
        "recovery_script": file_record(Path(__file__)),
    }


def validate_log_exact_route(path: Path) -> dict[str, Any]:
    path = require_file(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    skipped = text.count(
        "Input is found to be log-normalized already - skipping transformation."
    )
    if skipped < 2:
        raise ValueError(
            "CellEval log does not prove both continuous H5AD inputs skipped transformation"
        )
    forbidden = (
        "Converting to norm-log",
        "Discovered integer data for pred",
        "Invalid scale:",
        "Traceback (most recent call last):",
    )
    present = [value for value in forbidden if value in text]
    if present:
        raise ValueError(f"CellEval log contradicts exact-input route: {present}")
    return {
        "record": file_record(path),
        "lognorm_inputs_skipped_transformation": skipped,
        "forbidden_messages_absent": list(forbidden),
    }


def validate_completion(path: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    path = require_file(path)
    value = load_object(path)
    if value.get("passed") is not True:
        raise ValueError("exact pooled CellEval completion did not pass")
    if value.get("kind") != "pbmc_ctxw256_14k_exact_allow_discrete_pooled_celleval":
        raise ValueError("unexpected exact pooled CellEval completion kind")
    if int(value.get("perturbations", -1)) != EXPECTED_PERTURBATIONS:
        raise ValueError("unexpected perturbation count in exact completion")
    if value.get("cell_eval") != "0.6.6" or value.get("pdex") != "0.1.26":
        raise ValueError("unexpected evaluator versions in exact completion")
    if value.get("input_policy", {}).get("source_overwritten") is not False:
        raise ValueError("completion does not attest immutable prediction")
    if value.get("de_reuse", {}).get("recomputed") is not False:
        raise ValueError("completion does not attest DE reuse")
    result = value.get("results", {})
    result_path = require_file(Path(result.get("path", "")))
    if verify_hashes and sha256_file(result_path) != result.get("sha256"):
        raise ValueError("exact pooled CellEval result SHA mismatch")
    return value


def execute(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    completion_path = paths["exact_root"] / "completion.json"
    if completion_path.exists():
        completion = validate_completion(completion_path)
        if args.publish_for_frozen_resume:
            publish(args, paths, completion=completion)
        return completion

    contract = build_input_contract(args, paths, verify_hashes=True)
    paths["audit_root"].mkdir(parents=True, exist_ok=True)
    write_json_once(paths["audit_root"] / "input_contract.json", contract)

    attempt_id = args.attempt_id
    if not ATTEMPT_RE.fullmatch(attempt_id):
        raise ValueError("attempt-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    attempt_root = paths["exact_root"] / "attempts" / attempt_id
    eval_root = attempt_root / "evaluation"
    result_path = eval_root / "results.csv"
    log_path = attempt_root / "celleval.log"
    status_path = attempt_root / "status.json"
    command = build_cell_eval_command(
        cell_eval_cli=args.cell_eval_cli.expanduser().resolve(strict=True),
        pred=paths["pred"],
        real=paths["real"],
        pred_de=paths["pred_de"],
        real_de=paths["real_de"],
        outdir=eval_root,
        threads=args.num_threads,
    )
    attempt_contract = {
        "schema_version": 1,
        "kind": "pbmc_pooled_celleval_exact_allow_discrete_attempt/v1",
        "attempt_id": attempt_id,
        "input_contract_sha256": sha256_file(
            paths["audit_root"] / "input_contract.json"
        ),
        "command": command,
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(args.num_threads),
            "MKL_NUM_THREADS": str(args.num_threads),
        },
    }
    write_json_once(attempt_root / "attempt_contract.json", attempt_contract)

    if not result_path.exists():
        if eval_root.exists() and any(eval_root.iterdir()):
            raise RuntimeError(
                f"partial CellEval output will not be overwritten; use a new --attempt-id: {eval_root}"
            )
        if log_path.exists() or status_path.exists():
            raise RuntimeError(
                f"attempt already has logs/status but no result; use a new --attempt-id: {attempt_root}"
            )
        eval_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(attempt_contract["environment_overrides"])
        with exclusive_lock(paths["global_celleval_lock"], blocking=True):
            with log_path.open("xb") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=paths["work_root"],
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        write_json_once(
            status_path,
            {
                "schema_version": 1,
                "returncode": completed.returncode,
                "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "log": file_record(log_path),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"CellEval failed with status {completed.returncode}; preserve this attempt and use a new attempt-id"
            )

    results = validate_results(result_path)
    log_audit = validate_log_exact_route(log_path)

    # Re-hash every large mutable input after CellEval.  This both proves that
    # the prediction was not overwritten and that the supplied DE files were
    # the already sealed inputs throughout evaluation.
    after = build_input_contract(args, paths, verify_hashes=True)
    if after["inputs"] != contract["inputs"]:
        raise RuntimeError("pooled input or DE artifact changed during CellEval")
    if after["environment"] != contract["environment"]:
        raise RuntimeError("CellEval environment changed during evaluation")

    completion = {
        "schema_version": 1,
        "passed": True,
        "kind": "pbmc_ctxw256_14k_exact_allow_discrete_pooled_celleval",
        "perturbations": EXPECTED_PERTURBATIONS,
        "cell_eval": "0.6.6",
        "pdex": "0.1.26",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "results": {**results["record"], "rows": results["rows"]},
        "headline_means": results["headline_means"],
        "input_policy": {
            "allow_discrete": True,
            "exact_original_prediction": contract["inputs"]["pred"],
            "source_overwritten": False,
            "bounded_projection": False,
            "runtime_log_audit": log_audit,
        },
        "real_input": contract["inputs"]["real"],
        "de_reuse": {
            "recomputed": False,
            "manifest": contract["inputs"]["de_manifest"],
            "pred": contract["inputs"]["pred_de"],
            "real": contract["inputs"]["real_de"],
        },
        "input_contract": file_record(paths["audit_root"] / "input_contract.json"),
        "attempt_contract": file_record(attempt_root / "attempt_contract.json"),
    }
    write_json_once(completion_path, completion)
    validate_completion(completion_path)
    if args.publish_for_frozen_resume:
        publish(args, paths, completion=completion)
    return completion


def publish(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    completion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exact_completion_path = paths["exact_root"] / "completion.json"
    exact = completion or validate_completion(exact_completion_path)
    validate_completion(exact_completion_path)

    compat_root = paths["compat_root"]
    compat_root.mkdir(parents=True, exist_ok=True)
    unexpected = [
        path for path in compat_root.iterdir() if path.name != "completion.json"
    ]
    if unexpected:
        raise RuntimeError(
            "failed pooled_celleval directory has unexpected entries; refusing publication: "
            + ", ".join(str(path) for path in unexpected)
        )
    compat_completion = compat_root / "completion.json"
    write_json_once(compat_completion, exact)

    adapter = require_file(paths["contract_adapter"])
    command = [
        str(args.eval_python.expanduser().resolve(strict=True)),
        str(adapter),
        "complete",
        "celleval",
        "--run-root",
        str(paths["run_root"]),
        "--threads",
        str(args.num_threads),
    ]
    completed = subprocess.run(
        command,
        cwd=paths["work_root"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "compatibility completion was published, but frozen contract sealing failed; "
            f"rerun publish after diagnosis. stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    seal = {
        "schema_version": 1,
        "kind": "pbmc_pooled_celleval_exact_frozen_resume_bridge/v1",
        "compatibility_completion": file_record(compat_completion),
        "exact_completion": file_record(exact_completion_path),
        "contract_adapter": file_record(adapter),
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }
    write_json_once(paths["audit_root"] / "frozen_resume_bridge.json", seal)
    return seal


def audit(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    value = build_input_contract(args, paths, verify_hashes=False)
    value["planned_output_root"] = str(paths["exact_root"])
    value["planned_compatibility_completion"] = str(
        paths["compat_root"] / "completion.json"
    )
    value["note"] = "read-only audit used manifest-bound SHA values; execute re-hashes inputs"
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "execute", "publish"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--eval-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--cell-eval-cli",
        type=Path,
        default=Path("cell-eval"),
    )
    parser.add_argument("--global-celleval-lock", type=Path)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--attempt-id", default="attempt-001")
    parser.add_argument("--publish-for-frozen-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_threads < 1:
        raise ValueError("num-threads must be positive")
    paths = resolve_paths(args)
    if args.action == "audit":
        result = audit(args, paths)
    else:
        with exclusive_lock(paths["recovery_lock"], blocking=False):
            with exclusive_lock(paths["suite_lock"], blocking=False):
                if args.action == "execute":
                    result = execute(args, paths)
                else:
                    result = publish(args, paths)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
