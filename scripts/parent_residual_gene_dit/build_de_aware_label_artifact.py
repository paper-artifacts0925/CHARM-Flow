import argparse
import json
from pathlib import Path

from src.models.de_aware_residual import build_train_de_label_artifact


def _line_table(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("DE table must be CELL_LINE=resources/de.csv")
    line, path = value.split("=", 1)
    if not line or not path:
        raise argparse.ArgumentTypeError("DE table must be CELL_LINE=resources/de.csv")
    return line, Path(path)


def _condition(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "condition must be CELL_LINE=PERTURBATION"
        )
    line, perturbation = value.split("=", 1)
    if not line or not perturbation:
        raise argparse.ArgumentTypeError(
            "condition must be CELL_LINE=PERTURBATION"
        )
    return line, perturbation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--parent-sha256", default=None)
    parser.add_argument(
        "--de-table",
        action="append",
        type=_line_table,
        required=True,
        help="repeat once per cell line; every CSV must contain train conditions only",
    )
    parser.add_argument(
        "--audit-condition",
        action="append",
        type=_condition,
        default=[],
        help=(
            "repeat CELL_LINE=PERTURBATION; carved from Parent train and "
            "forbidden to DE labels/statistics"
        ),
    )
    parser.add_argument(
        "--parent-calibration",
        choices=("none", "no_intercept", "affine"),
        default="no_intercept",
        help=(
            "runtime Parent calibration; calibrated modes require audited "
            "ridge_support_mask provenance disjoint from audit"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    args = parser.parse_args()
    tables = dict(args.de_table)
    if len(tables) != len(args.de_table):
        raise ValueError("duplicate --de-table cell line")
    result = build_train_de_label_artifact(
        parent_artifact_path=args.parent_artifact,
        de_csv_by_cell_line=tables,
        output_path=args.output,
        expected_parent_sha256=args.parent_sha256,
        fdr_threshold=args.fdr_threshold,
        audit_conditions=args.audit_condition,
        parent_calibration=args.parent_calibration,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
