from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pbmc_pooled_de import MWUConfig, run_pooled_de_pipeline


def _load_expected_targets(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("targets", "official_test_cytokines", "cytokines"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise TypeError(f"expected target JSON must contain a list: {resolved}")
        return [str(value) for value in payload]
    return [
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _stage_manifests(args: argparse.Namespace) -> list[Path]:
    paths = list(args.stage_manifest or [])
    if args.stage_manifest_root is not None:
        root = args.stage_manifest_root.expanduser().resolve(strict=True)
        paths.extend(sorted(root.glob("*/stage_manifest.json")))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed PBMC pooled-DE audit: backed CSR chunks -> raw MWU cache -> "
            "one complete BH family per real/pred side"
        )
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-key", default="cytokine")
    parser.add_argument("--reference", default="PBS")
    parser.add_argument(
        "--identity-columns",
        nargs="+",
        required=True,
        help=(
            "Physical source identity columns. Production requires exactly the explicit "
            "semantics source_namespace source_dataset_name source_physical_row "
            "source_obs_name source_role (additional columns are allowed)."
        ),
    )
    parser.add_argument(
        "--identity-provenance-manifest",
        type=Path,
        help=(
            "Builder-produced attestation binding exact pooled/stage SHA values to a "
            "hashed physical-source-row mapping. If omitted, a failed input audit is "
            "written before any H5AD scan; historical generated obs names alone fail."
        ),
    )
    parser.add_argument("--stage-manifest", type=Path, action="append")
    parser.add_argument("--stage-manifest-root", type=Path)
    parser.add_argument(
        "--expected-targets-file",
        type=Path,
        required=True,
        help="Frozen official target IDs as a JSON list/protocol field or one ID per line.",
    )
    parser.add_argument("--expected-target-count", type=int, default=62)
    parser.add_argument("--expected-stage-count", type=int, default=18)
    parser.add_argument("--expected-total-cells", type=int, default=2_495_931)
    parser.add_argument("--expected-treated-cells", type=int, default=2_260_453)
    parser.add_argument("--expected-reference-cells", type=int, default=235_478)
    parser.add_argument("--expected-genes", type=int, default=2_000)
    parser.add_argument("--gene-chunk-size", type=int, default=128)
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=4_096,
        help="Bounds hidden full-row decoding performed by AnnData backed CSR slicing.",
    )
    parser.add_argument("--alternative", choices=["two-sided", "less", "greater"], default="two-sided")
    parser.add_argument("--method", choices=["auto", "asymptotic", "exact"], default="auto")
    parser.add_argument(
        "--use-continuity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--is-log1p", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exp-post-agg", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fold-change-clip", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifests = _stage_manifests(args)
    if not manifests:
        raise ValueError("at least one --stage-manifest or --stage-manifest-root is required")
    manifest = run_pooled_de_pipeline(
        real_path=args.real,
        pred_path=args.pred,
        output_dir=args.output_dir,
        group_key=args.group_key,
        reference=args.reference,
        identity_columns=args.identity_columns,
        stage_manifests=manifests,
        identity_provenance_manifest=args.identity_provenance_manifest,
        expected_targets=_load_expected_targets(args.expected_targets_file),
        expected_target_count=args.expected_target_count,
        expected_stage_count=args.expected_stage_count,
        expected_total_cells=args.expected_total_cells,
        expected_treated_cells=args.expected_treated_cells,
        expected_reference_cells=args.expected_reference_cells,
        expected_genes=args.expected_genes,
        require_aligned_identities=True,
        require_physical_identity_attestation=True,
        verify_stage_artifact_hashes=True,
        verify_stage_row_union=True,
        hash_inputs=True,
        gene_chunk_size=args.gene_chunk_size,
        row_chunk_size=args.row_chunk_size,
        config=MWUConfig(
            alternative=args.alternative,
            method=args.method,
            use_continuity=args.use_continuity,
            is_log1p=args.is_log1p,
            exp_post_agg=args.exp_post_agg,
            fold_change_clip=args.fold_change_clip,
        ),
    )
    print(
        "PBMC_POOLED_DE_COMPLETE "
        f"tests_per_side={manifest['expected_tests_per_side']} "
        f"output={args.output_dir.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
