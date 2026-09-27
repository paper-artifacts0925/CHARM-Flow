import importlib.util
from pathlib import Path
from typing import Any

import hydra
import omegaconf


ROOT = Path(__file__).resolve().parents[2]
SCREEN_PATH = ROOT / "scripts/tahoe/checkpoint_contract.py"
SPEC = importlib.util.spec_from_file_location("_tahoe_ctxw256_screen", SCREEN_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import Tahoe ctxw256 screen: {SCREEN_PATH}")
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)
base = screen.base


def audit_cfg(cfg: omegaconf.DictConfig) -> dict[str, Any]:
    """Keep the SHA-pinned screen contract, changing only 28 to 245 groups/dose."""

    base.require(tuple(base.SCENARIOS) == screen.EXPECTED_SCENARIOS, "wrong dose strata")
    base.require(len(base.strings(cfg.data.holdout_celltype)) == 5, "expected 5 held-out lines")
    base.require(int(cfg.sampling.tahoe_groups_per_scenario) == 245, "expected 245 groups per dose")
    base.require(int(cfg.sampling.tahoe_cells_per_group) == 64, "expected 64 cells per group")
    base.require(
        int(cfg.model.parent_residual_context_hidden_dim) == 256,
        "runtime context hidden dim must be 256",
    )

    torch_module = base.torch
    capture = screen._TorchLoadCapture(torch_module)
    base.torch = capture
    try:
        audit = screen._upstream_audit_cfg(cfg)
    finally:
        base.torch = torch_module
    base.require(capture.checkpoint is not None, "upstream audit did not inspect checkpoint")
    model_cfg = capture.checkpoint["hyper_parameters"]["model_cfg"]
    base.require(
        int(model_cfg.get("parent_residual_context_hidden_dim", -1)) == 256,
        "checkpoint context hidden dim must be 256",
    )
    audit.update(
        {
            "entrypoint": "full_test.py",
            "protocol": "official_double_ood_dose_stratified_735_group_test",
            "heldout_cell_lines": 5,
            "dose_scenarios": list(screen.EXPECTED_SCENARIOS),
            "groups_per_dose": 245,
            "expected_total_groups": 735,
            "parent_residual_context_hidden_dim": 256,
        }
    )
    return audit


base.audit_cfg = audit_cfg


@hydra.main(version_base=None, config_path="../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg: omegaconf.DictConfig) -> None:
    base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
