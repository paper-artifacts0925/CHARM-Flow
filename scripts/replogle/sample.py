import hashlib
import os
import sys
from pathlib import Path

# Make the entry point self-contained. Hydra/DataLoader workers may re-import
# this file from a working directory outside the repository and must not rely
# on a caller-provided PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import hydra
import omegaconf
import pytorch_lightning as pl

from src.apps.sampling.sampling_generation import generate_samples
from src.apps.sampling.sampling_setup import (
    build_sampling_datamodule,
    load_sampling_model,
    populate_covariate_cfg,
)
from src.apps.sampling.sampling_utils import setup_device
from src.common.utils import setup_loggings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: omegaconf.DictConfig) -> None:
    omegaconf.OmegaConf.resolve(cfg)
    assert str(cfg.sampling.split) == "test"
    assert list(map(str, cfg.data.evaluation_splits)) == ["test"]
    assert int(cfg.optimization.seed) == 42
    assert int(cfg.optimization.micro_batch_size) == 512
    assert str(cfg.sampling.initial_state) == "parent_residual"
    assert float(cfg.sampling.guidance_strength) == 0.0
    for key in ("num_sampled_batches", "max_perturbations", "fixed_cells_per_perturbation"):
        assert cfg.sampling.get(key) is None, key
    model_cfg = cfg.model
    assert model_cfg.parent_residual_context_hidden_dim == 256
    assert model_cfg.parent_residual_context_parent_depth == 2
    assert model_cfg.parent_residual_strong_condition_memory_enabled is True
    assert model_cfg.parent_residual_sparse_joint_context_enabled is True

    checkpoint = Path(str(cfg.model_checkpoint_path)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_sha = os.environ.get("CELL_DETR_CHECKPOINT_SHA256")
    if expected_sha and sha256(checkpoint) != expected_sha.lower():
        raise AssertionError("checkpoint SHA256 mismatch")

    logger = setup_loggings(cfg)
    seed = int(cfg.optimization.seed)
    pl.seed_everything(seed, workers=True)
    datamodule = build_sampling_datamodule(cfg, logger)
    datamodule.evaluation_split_names = ["test"]
    populate_covariate_cfg(cfg, datamodule)
    model = load_sampling_model(cfg, logger, datamodule)
    expected_encoding = cfg.model.get("parent_residual_child_encoding_mode", "joint")
    actual_encoding = getattr(model.model.context_encoder, "child_encoding_mode", "joint")
    assert actual_encoding == expected_encoding, (actual_encoding, expected_encoding)
    logger.info("CHILD_ENCODING_MODE_RESTORED mode=%s", actual_encoding)
    pl.seed_everything(seed, workers=True)
    logger.info("CELL_DETR_POST_LOAD_RESEED seed=%s", seed)
    device = setup_device(cfg, logger)
    datamodule.setup_dataset()
    _, samples, _, _ = generate_samples(
        model, model.diffusion, cfg, device, logger, datamodule, pca_for_decode=None
    )
    logger.info("CELL_DETR_REPLOGLE_TEST_COMPLETE shape=%s", tuple(samples.shape))


if __name__ == "__main__":
    main()
