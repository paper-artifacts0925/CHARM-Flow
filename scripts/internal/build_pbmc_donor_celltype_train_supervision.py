from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.internal import build_pbmc_train_supervision as _base  # noqa: E402


def _require_donor_celltype_parent(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as parent:
        if "parent_context_kind" not in parent.files:
            raise ValueError(
                "donor-celltype supervision requires Parent parent_context_kind"
            )
        grouping = _base._scalar_string(
            np.asarray(parent["parent_context_kind"]), "parent_context_kind"
        ).strip().lower()
        if grouping != _base.PARENT_GROUPING_DONOR_CELLTYPE:
            raise ValueError(
                "donor-celltype supervision refuses Parent grouping "
                f"{grouping!r}"
            )
        if "cellline_names" not in parent.files:
            raise KeyError("Parent artifact is missing cellline_names")
        lines = np.asarray(parent["cellline_names"]).astype(str)
        if lines.shape != (_base.EXPECTED_DONOR_CELLTYPE_LINES,):
            raise ValueError(
                "donor-celltype Parent must contain "
                f"{_base.EXPECTED_DONOR_CELLTYPE_LINES} ordered rows"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _base.parse_args(argv)
    _require_donor_celltype_parent(args.parent)
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
