import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = Path(os.environ.get("CHARM_RESOURCE_ROOT", str(ROOT / "resources")))
DATA_ROOT = Path(os.environ.get("CHARM_DATA_ROOT", str(RESOURCE_ROOT / "data/PerturbDiff_data/perturb_data")))
ARTIFACT_ROOT = Path(os.environ.get("CHARM_ARTIFACT_ROOT", str(RESOURCE_ROOT / "artifacts")))
DEFAULT_CONTROL = DATA_ROOT / "tmp_ourtahoe_ctrl.h5ad"
CONTROL_SHA256 = "be3482a238d8831fa66bbbdf568cc6448ff4ba2e35857820ec7abd2be2707a36"
CONTROL_LABEL = "[('DMSO_TF', 0.0, 'uM')]"
EXPECTED_GROUPS = 735
CELLS_PER_GROUP = 64
EXPECTED_TREATED = EXPECTED_GROUPS * CELLS_PER_GROUP


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def build_one(source: Path, control: ad.AnnData, destination: Path) -> dict[str, object]:
    treated = ad.read_h5ad(source)
    require(treated.shape == (EXPECTED_TREATED, 2000), f"unexpected treated shape: {treated.shape}")
    require("drugname_drugconc" in treated.obs, "treated perturbation column missing")
    require(CONTROL_LABEL not in set(treated.obs["drugname_drugconc"].astype(str)),
            "treated input already contains controls")
    control_copy = control.copy()
    control_copy.var_names = treated.var_names.copy()
    control_copy.obs = pd.DataFrame(
        {"drugname_drugconc": np.full(control_copy.n_obs, CONTROL_LABEL, dtype=object)},
        index=control_copy.obs_names.copy(),
    )
    combined = ad.concat(
        [treated, control_copy], axis=0, join="inner", merge="same",
        label="celleval_role", keys=["treated", "control"], index_unique="-",
    )
    require(combined.shape == (EXPECTED_TREATED + 204_536, 2000),
            f"combined shape changed: {combined.shape}")
    counts = combined.obs["drugname_drugconc"].astype(str).value_counts()
    require(int(counts.loc[CONTROL_LABEL]) == 204_536, "control count changed")
    require(int(counts.drop(CONTROL_LABEL).sum()) == EXPECTED_TREATED, "treated count changed")
    require(np.isfinite(np.asarray(combined.X)).all(), "combined matrix is non-finite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"refusing to overwrite {destination}")
    combined.write_h5ad(destination, compression=None)
    record = {
        "source": str(source.resolve()), "source_sha256": sha256(source),
        "output": str(destination.resolve()), "output_sha256": sha256(destination),
        "shape": list(combined.shape), "treated_cells": EXPECTED_TREATED,
        "control_cells": 204_536,
        "perturbations_excluding_control": int(len(counts) - 1),
    }
    del combined, control_copy, treated
    return record


def main() -> None:
    args = parse_args()
    test_dir = args.test_dir.expanduser().resolve()
    output = (args.output_dir or test_dir / "celleval_table2").expanduser().resolve()
    metrics_path = test_dir / "metrics.json"
    selection_path = test_dir / "selection_manifest.json"
    require(metrics_path.is_file() and selection_path.is_file(), "missing Tahoe test manifests")
    metrics = json.loads(metrics_path.read_text())
    selection = json.loads(selection_path.read_text())
    require(metrics.get("passed") is True, "test metrics did not pass")
    require(int(metrics["groups"]) == EXPECTED_GROUPS and int(metrics["cells"]) == EXPECTED_TREATED,
            "test coverage changed")
    require(int(selection["selected_groups"]) == EXPECTED_GROUPS,
            "selection does not contain 735 groups")
    pred, truth = Path(metrics["pred"]), Path(metrics["truth"])
    require(sha256(pred) == metrics["pred_sha256"], "prediction checksum changed")
    require(sha256(truth) == metrics["truth_sha256"], "truth checksum changed")
    control_path = args.control.expanduser().resolve()
    require(sha256(control_path) == CONTROL_SHA256, "canonical Tahoe control checksum changed")
    control = ad.read_h5ad(control_path)
    require(control.shape == (204_536, 2000), f"control shape changed: {control.shape}")
    require(set(control.obs["drugname_drugconc"].astype(str)) == {CONTROL_LABEL},
            "canonical control label changed")
    output.mkdir(parents=True, exist_ok=True)
    records = {
        "prediction": build_one(pred, control, output / "prediction_with_canonical_controls.h5ad"),
        "truth": build_one(truth, control, output / "truth_with_canonical_controls.h5ad"),
    }
    record = {
        "schema_version": 1, "passed": True, "kind": "tahoe_u2_full_735_celleval_input",
        "control": str(control_path), "control_sha256": CONTROL_SHA256,
        "control_label": CONTROL_LABEL,
        "gene_alignment": "canonical_tahoe_positional_2000_hvg_then_assign_treated_var_names",
        "records": records,
    }
    manifest = output / "input_manifest.json"
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest), **records}, sort_keys=True))


if __name__ == "__main__":
    main()
