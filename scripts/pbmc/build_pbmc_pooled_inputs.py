import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pbmc_pooled_builder import build_pooled_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-loaded-elems",
        type=int,
        default=25_000_000,
        help="AnnData on-disk concatenation memory bound.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_loaded_elems <= 0:
        raise ValueError("--max-loaded-elems must be positive")
    result = build_pooled_inputs(
        merge_manifest_path=args.merge_manifest,
        output_dir=args.output_dir,
        max_loaded_elems=args.max_loaded_elems,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
