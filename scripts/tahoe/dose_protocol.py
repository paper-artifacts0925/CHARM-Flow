from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V2_PATH = ROOT / "scripts/tahoe/treated_only.py"
SPEC = importlib.util.spec_from_file_location("_tahoe_u2_screen_v2", V2_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import Tahoe screen v2: {V2_PATH}")
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)
base = v2.base

# Tahoe's repository split is the intersection of held-out perturbations and
# held-out cell lines (double OOD). Stratify the official test by dose.
base.SCENARIOS = ("dose_0.05uM", "dose_0.5uM", "dose_5.0uM")


def scenario_for(
    pert: str,
    line: str,
    test_perts: set[str],
    _val_perts: set[str],
    heldout: set[str],
) -> str | None:
    if pert not in test_perts or line not in heldout:
        return None
    try:
        condition = ast.literal_eval(pert)
        dose = float(condition[0][1])
        unit = str(condition[0][2])
    except (ValueError, SyntaxError, TypeError, IndexError):
        return None
    if unit != "uM":
        return None
    for expected, name in (
        (0.05, "dose_0.05uM"),
        (0.5, "dose_0.5uM"),
        (5.0, "dose_5.0uM"),
    ):
        if np.isclose(dose, expected):
            return name
    return None


base.scenario_for = scenario_for


if __name__ == "__main__":
    base.main()
