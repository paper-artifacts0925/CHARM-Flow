"""Model builders and model-init flows split from rawdata_diffusion_training.py (logic-preserving)."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import hydra
import torch
from pytorch_lightning.utilities import model_summary

from src.apps.training.training_model_checkpoint import (
    allow_omegaconf_checkpoint_unpickling,
    load_plmodel_checkpoint,
    maybe_load_and_patch_checkpoint_model as _maybe_load_and_patch_checkpoint_model,
)
from src.apps.training.training_model_compare import maybe_compare_model_representation as _maybe_compare_model_representation
def _partial_weight_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _partial_weight_allowlist(value):
    if value in (None, "", "null"):
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _format_partial_weight_keys(keys):
    return "[" + ", ".join(repr(str(key)) for key in sorted(keys, key=str)) + "]"


def _preserve_fresh_partial_weight_target(key, target_value):
    """Keep runtime/artifact state from the freshly constructed model.

    EMA checkpoints created with buffer averaging can corrupt integer lookup
    tables by averaging in floating point and truncating on copy-back. Parent
    bank tensors are artifact-derived constants rather than learned weights,
    so the runtime-verified artifact must remain their source of truth even
    when those tensors are floating point.
    """

    if str(key).startswith("model.prior_bank."):
        return True
    return not (target_value.is_floating_point() or target_value.is_complex())


def maybe_load_partial_weight_checkpoint(cfg, model, logger):
    """Load matching model weights without restoring optimizer/trainer state."""
    ckpt_path = getattr(cfg.model, "partial_weight_ckpt_path", None)
    if ckpt_path in (None, "", "null"):
        return model
    strict_coverage = bool(
        getattr(cfg.model, "partial_weight_strict_coverage", False)
    )
    expected_sha256 = getattr(cfg.model, "partial_weight_ckpt_sha256", None)
    if strict_coverage and expected_sha256 in (None, "", "null"):
        raise ValueError(
            "Strict partial-weight coverage requires partial_weight_ckpt_sha256"
        )
    if expected_sha256 not in (None, "", "null"):
        expected_sha256 = str(expected_sha256).lower()
        actual_sha256 = _partial_weight_sha256(ckpt_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Partial-weight checkpoint SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256} ({ckpt_path})"
            )
    allow_omegaconf_checkpoint_unpickling()
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load partial-weight checkpoint with safe weights-only "
            f"deserialization: {ckpt_path}. Unsafe pickle fallback is disabled; "
            "convert the checkpoint to a tensor/primitive-only state dict or "
            "allowlist its trusted globals explicitly."
        ) from exc
    if not isinstance(ckpt, Mapping):
        raise TypeError(
            "Partial-weight checkpoint must deserialize to a mapping, got "
            f"{type(ckpt).__name__}: {ckpt_path}"
        )
    use_ema = bool(getattr(cfg.model, "partial_weight_use_ema", False))
    if use_ema and "state_dict" in ckpt:
        source_state = ckpt["state_dict"]
        source_kind = "ema_state_dict"
    else:
        source_state = ckpt.get("current_model_state", ckpt.get("state_dict", ckpt))
        source_kind = "current_model_state"
    if not isinstance(source_state, Mapping):
        raise TypeError(
            f"Checkpoint {source_kind} must be a mapping, got "
            f"{type(source_state).__name__}: {ckpt_path}"
        )
    target_state = model.state_dict()
    preserved_target = {
        key
        for key, value in target_state.items()
        if torch.is_tensor(value)
        and _preserve_fresh_partial_weight_target(key, value)
    }
    matched = {}
    skipped = []
    skipped_non_tensor = []
    unexpected_source = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if not torch.is_tensor(value):
            # A raw tensor-only export may carry primitive checkpoint metadata
            # (for example epoch/global_step) beside its state keys. Such
            # values are not model-state entries and must not count as
            # unexpected coverage.
            skipped_non_tensor.append(key)
        elif key not in target_state:
            unexpected_source.append(key)
        elif not torch.is_tensor(target_value):
            skipped_non_tensor.append(key)
        elif key in preserved_target:
            # Deliberately ignore checkpoint content (including its shape and
            # dtype). The freshly constructed, hash-verified artifact and
            # runtime vocabularies own these constants and lookup tables.
            continue
        elif tuple(value.shape) == tuple(target_value.shape):
            matched[key] = value
        else:
            skipped.append(key)

    if strict_coverage:
        min_matched = getattr(cfg.model, "partial_weight_min_matched_keys", 0)
        if isinstance(min_matched, bool):
            raise ValueError(
                "partial_weight_min_matched_keys must be a non-negative integer"
            )
        try:
            parsed_min_matched = int(min_matched)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "partial_weight_min_matched_keys must be a non-negative integer"
            ) from exc
        if parsed_min_matched < 0 or str(parsed_min_matched) != str(min_matched):
            raise ValueError(
                "partial_weight_min_matched_keys must be a non-negative integer"
            )
        min_matched = parsed_min_matched
        allowed_missing = set(
            _partial_weight_allowlist(
                getattr(cfg.model, "partial_weight_allowed_missing_keys", ())
            )
        )
        allowed_missing_prefixes = _partial_weight_allowlist(
            getattr(cfg.model, "partial_weight_allowed_missing_prefixes", ())
        )
        allowed_unexpected = set(
            _partial_weight_allowlist(
                getattr(cfg.model, "partial_weight_allowed_unexpected_keys", ())
            )
        )
        missing = [
            key
            for key in target_state
            if key not in matched and key not in preserved_target
        ]
        disallowed_missing = [
            key
            for key in missing
            if key not in allowed_missing
            and not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        disallowed_unexpected = [
            key for key in unexpected_source if key not in allowed_unexpected
        ]
        failures = []
        if len(matched) < min_matched:
            failures.append(
                f"matched={len(matched)} is below min_matched_keys={min_matched}; "
                f"matched_keys={_format_partial_weight_keys(matched)}"
            )
        if skipped:
            failures.append(
                "shape_mismatched_keys=" + _format_partial_weight_keys(skipped)
            )
        if disallowed_missing:
            failures.append(
                "missing_keys=" + _format_partial_weight_keys(disallowed_missing)
            )
        if disallowed_unexpected:
            failures.append(
                "unexpected_keys="
                + _format_partial_weight_keys(disallowed_unexpected)
            )
        if failures:
            metadata_note = ""
            if skipped_non_tensor:
                metadata_note = (
                    "; ignored_non_tensor_metadata_keys="
                    + _format_partial_weight_keys(skipped_non_tensor)
                )
            raise RuntimeError(
                "Strict partial-weight coverage failed for "
                f"{ckpt_path} ({source_kind}): "
                + "; ".join(failures)
                + metadata_note
            )

    missing, unexpected = model.load_state_dict(matched, strict=False)
    if getattr(model, "match_fm_teacher", None) is not None:
        model.match_fm_teacher.load_state_dict(model.model.state_dict(), strict=True)
        model._match_fm_teacher_initialized = True
    if getattr(model, "match_fm_prior_teacher", None) is not None:
        model.match_fm_prior_teacher.load_state_dict(
            model.match_fm_prior.state_dict(), strict=True
        )
    logger.info(
        "Loaded partial weights from %s (%s): matched=%d preserved_target=%d "
        "skipped=%d non_tensor=%d missing=%d unexpected=%d",
        ckpt_path,
        source_kind,
        len(matched),
        len(preserved_target),
        len(skipped),
        len(skipped_non_tensor),
        len(missing),
        len(unexpected_source) if strict_coverage else len(unexpected),
    )
    return model


def build_model(cfg, logger, datamodule):
    """
    Build model.

    :param cfg: Runtime configuration object.
    :param logger: Logger instance.
    :param datamodule: Data module providing datasets and loaders.
    :return: Requested object(s) for downstream use.
    """
    model = hydra.utils.instantiate(
        cfg.lightning.model_module,
        _recursive_=False,
        cov_encoding_cfg=cfg.cov_encoding,
        model_cfg=cfg.model,
        py_logger=logger,
        optimizer_cfg=cfg.optimization,
        trainer_cfg=cfg.trainer,
        all_split_names=getattr(
            datamodule,
            "evaluation_split_names",
            datamodule.all_split_names,
        ),
    )

    summary = model_summary.ModelSummary(model, max_depth=2)
    logger.info(summary)

    model.model.group_mean = None
    model.model.group_mean_ctrl = None

    model = maybe_load_partial_weight_checkpoint(cfg, model, logger)

    return model

def maybe_compare_model_representation(cfg, logger, datamodule):
    """Execute `maybe_compare_model_representation` and return values used by downstream logic."""
    return _maybe_compare_model_representation(
        cfg,
        logger,
        datamodule,
        load_plmodel_checkpoint=load_plmodel_checkpoint,
    )

def maybe_load_and_patch_checkpoint_model(cfg, model, logger):
    """Execute `maybe_load_and_patch_checkpoint_model` and return values used by downstream logic."""
    return _maybe_load_and_patch_checkpoint_model(cfg, model, logger)

def maybe_reinitialize_from_scratch(cfg, model, logger):
    """Execute `maybe_reinitialize_from_scratch` and return values used by downstream logic."""
    if cfg.model.reinitial_all_from_scratch:
        model.model.initialize_weights()
        logger.info("Re-initialize all layers")
