"""DataLoader bridge kept beside the isolated Hungarian-flow sampler."""

from __future__ import annotations

from torch.utils.data import DataLoader

from .sampler import CombinationEpisodeBatchSampler


def build_combination_dataloader(dm):
    if not bool(getattr(dm.data_args, "meta_batch_episode_enabled", False)):
        raise ValueError(
            "Hungarian combination sampling requires meta_batch_episode_enabled"
        )
    if bool(getattr(dm.data_args, "pert_mix_enabled", False)):
        raise ValueError("pert_mix_enabled must be false for combination episodes")
    sampler = CombinationEpisodeBatchSampler(
        dm.train_dataset,
        cells_per_device_batch=int(dm.micro_batch_size),
        batch_pairing=bool(
            getattr(dm.data_args, "hungarian_flow_batch_pairing_enabled", False)
        ),
        min_views=int(getattr(dm.data_args, "meta_batch_min_views", 2)),
        max_views=int(getattr(dm.data_args, "meta_batch_max_views", 5)),
        shuffle=True,
        seed=dm.seed,
        parent_grouping=str(
            getattr(dm.data_args, "meta_batch_parent_grouping", "celltype")
        ),
        perturbation_balance_gamma=float(
            getattr(
                dm.data_args,
                "hungarian_flow_perturbation_balance_gamma",
                0.0,
            )
        ),
    )
    loader = DataLoader(
        dm.train_dataset,
        batch_sampler=sampler,
        collate_fn=dm.train_dataset.collate_fn,
        num_workers=dm.data_args.num_workers,
        persistent_workers=dm.data_args.persistent_workers,
        pin_memory=dm.data_args.pin_memory,
        prefetch_factor=dm.data_args.prefetch_factor,
    )
    dm.py_logger.info(
        "Finished loading Hungarian combination data: %d episodes",
        len(sampler.episodes),
    )
    return loader


__all__ = ["build_combination_dataloader"]
