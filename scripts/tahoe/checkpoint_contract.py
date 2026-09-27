from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import hydra
import omegaconf


ROOT = Path(__file__).resolve().parents[2]
V3_PATH = ROOT / "scripts/tahoe/dose_protocol.py"
SPEC = importlib.util.spec_from_file_location("_tahoe_u2_screen_v3_ctxw256", V3_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import Tahoe screen v3: {V3_PATH}")
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)
base = v3.base

CURRENT_EXPORT = (
    ROOT
    / "outputs/crossdataset_ctxw256_seed42_v2/tahoe100m/formal14000"
    / "training/trainer/crossdataset_ctxw256_seed42_v2_tahoe100m_formal14000"
)
CURRENT_EXPORT_SHA256 = (
    "2bd771e3e4feebe709e2d016205bd2174dc00af7b9ea881fbfde325c1e0d8732"
)
EXPECTED_SCENARIOS = ("dose_0.05uM", "dose_0.5uM", "dose_5.0uM")

# v2 uses these two names only as file-presence sentinels after installing its
# treated-only Tahoe adapters.  Keep all checkpoint sentinels on the same pin.
base.FINAL_EXPORT = CURRENT_EXPORT
base.FINAL_EXPORT_SHA256 = CURRENT_EXPORT_SHA256
base.CONTROL = CURRENT_EXPORT
base.CELL_LINE_MAP = CURRENT_EXPORT

_upstream_audit_cfg = base.audit_cfg


class _TorchLoadCapture:
    """Proxy torch while the upstream audit loads its already-required export."""

    def __init__(self, torch_module: Any) -> None:
        self._torch_module = torch_module
        self.checkpoint: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch_module, name)

    def load(self, *args: Any, **kwargs: Any) -> Any:
        checkpoint = self._torch_module.load(*args, **kwargs)
        self.checkpoint = checkpoint
        return checkpoint


def audit_cfg(cfg: omegaconf.DictConfig) -> dict[str, Any]:
    """Extend the official v3 audit without a second 991-MiB checkpoint load."""

    base.require(tuple(base.SCENARIOS) == EXPECTED_SCENARIOS, "wrong dose strata")
    base.require(len(base.strings(cfg.data.holdout_celltype)) == 5, "expected 5 held-out lines")
    base.require(int(cfg.sampling.tahoe_groups_per_scenario) == 28, "expected 28 groups per dose")
    base.require(int(cfg.sampling.tahoe_cells_per_group) == 64, "expected 64 cells per group")
    base.require(
        int(cfg.model.parent_residual_context_hidden_dim) == 256,
        "runtime context hidden dim must be 256",
    )

    torch_module = base.torch
    capture = _TorchLoadCapture(torch_module)
    base.torch = capture
    try:
        audit = _upstream_audit_cfg(cfg)
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
            "entrypoint": "checkpoint_contract.py",
            "protocol": "official_double_ood_dose_stratified_representative_subset",
            "heldout_cell_lines": 5,
            "dose_scenarios": list(EXPECTED_SCENARIOS),
            "parent_residual_context_hidden_dim": 256,
        }
    )
    return audit


base.audit_cfg = audit_cfg


@hydra.main(version_base=None, config_path="../../configs", config_name="rawdata_diffusion_sampling")
def main(cfg: omegaconf.DictConfig) -> None:
    """Resolve Hydra relative to this concrete entrypoint, then run v3 core."""

    base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
