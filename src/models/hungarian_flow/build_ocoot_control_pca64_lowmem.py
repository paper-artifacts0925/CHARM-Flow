#!/usr/bin/env python
"""PCA64 CLI using the bounded-memory OCOO-T HDF5 control loader."""

from __future__ import annotations

import json

from src.models.hungarian_flow import build_ocoot_control_pca64 as pca_builder
from src.models.hungarian_flow.build_ocoot_recursive_artifacts_v2_lowmem import (
    load_ocoot_control_rows_v2_lowmem,
)


def main(argv=None) -> int:
    args = pca_builder.parse_args(argv)
    original = pca_builder.load_ocoot_control_rows_v2
    pca_builder.load_ocoot_control_rows_v2 = load_ocoot_control_rows_v2_lowmem
    try:
        report = pca_builder.build_ocoot_control_pca64(
            dataset=args.dataset,
            input_path=args.input,
            output=args.output,
            selected_gene_file=args.selected_gene_file,
            feature_key=args.feature_key,
            normalize_counts=args.normalize_counts,
            seed=args.seed,
        )
    finally:
        pca_builder.load_ocoot_control_rows_v2 = original
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
