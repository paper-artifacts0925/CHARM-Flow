import importlib.util
from pathlib import Path
import pickle
from typing import Any

import anndata
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "scripts/tahoe/sampling_core.py"
SPEC = importlib.util.spec_from_file_location("_tahoe_u2_screen_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import Tahoe screen base: {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

# The legacy Tahoe output helper expects a mapping asset that is not present in
# the reproduction bundle.  The screen is treated-only, so construct its obs
# directly and source all-cell-line controls from the audited OCOOT artifact.
base.CONTROL = base.FINAL_EXPORT
base.CELL_LINE_MAP = base.FINAL_EXPORT


def install_empty_control(cfg: Any) -> None:
    def empty_control(_cfg: Any):
        with Path(str(cfg.data.selected_gene_file)).open("rb") as handle:
            loaded = pickle.load(handle)
        genes = sorted(loaded) if isinstance(loaded, set) else list(loaded)
        base.require(len(genes) == 2000, "selected-gene asset is not 2000 genes")
        obs = pd.DataFrame({"drugname_drugconc": [], "cell_name": [], "plate": []})
        return anndata.AnnData(
            X=sparse.csr_matrix((0, 2000)),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(genes)),
        ), genes

    def treated_only_obs(
        _cfg: Any,
        all_pert: Any,
        all_celltype: Any,
        all_batch: Any,
        _ctrl: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "drugname_drugconc": np.asarray(all_pert),
                "cell_line": np.asarray(all_celltype),
                "plate": np.asarray(all_batch),
            }
        )

    base.sampling_generation.load_ctrl_adata = empty_control
    base.sampling_generation.build_obs_data = treated_only_obs


def control_means(
    groups: pd.DataFrame,
    genes: list[str],
) -> dict[tuple[str, str], np.ndarray]:
    artifact = base.EXPECTED_ARTIFACTS["parent_residual_artifact_path"][0]
    with np.load(artifact, allow_pickle=False) as payload:
        artifact_genes = list(map(str, payload["gene_names"]))
        lines = list(map(str, payload["cellline_names"]))
        divisor = float(payload["normalization_divisor"])
        means = np.asarray(payload["control_mean"], dtype=np.float64) * divisor
    base.require(artifact_genes == genes, "OCOOT control gene order changed")
    base.require(
        means.shape == (50, 2000) and np.isfinite(means).all(),
        "invalid OCOOT controls",
    )
    by_line = dict(zip(lines, means))
    requested = set(
        zip(groups["cell_line"].astype(str), groups["plate"].astype(str))
    )
    base.require(
        all(line in by_line for line, _ in requested),
        "screen contains an unknown cell line",
    )
    return {(line, plate): by_line[line] for line, plate in requested}


base.install_empty_control = install_empty_control
base.control_means = control_means


if __name__ == "__main__":
    base.main()
