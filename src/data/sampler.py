"""Unified sampler definitions and exports for cell-set batching."""

import math
import logging
from collections import defaultdict
from typing import Iterator, Optional, Tuple

import numpy as np
import omegaconf
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


Triplet = Tuple[str, int, int]  # (ds_name, local_idx, is_padded_value)
EpisodeIndex = Tuple[str, int, int, int]  # plus stable episode token
RaggedEpisodeIndex = Tuple[
    str, int, int, int, int, int
]  # plus batch-local episode id and view id
logger = logging.getLogger(__name__)


class DistributedMetaBatchEpisodeBatchSampler(Sampler[list[EpisodeIndex]]):
    """Yield complete meta-batch episodes on one DDP rank.

    The default mode preserves the original fixed A/B/C layout. Dynamic mode
    keeps a fixed *set-view* budget per device batch while allowing each
    episode to contain a different number of views. A dynamic view is one
    full cell set from one raw experimental batch. Raw-batch pairs with the
    least prior coverage are selected first, so the episode hypergraph covers
    A+B, A+C, B+C, ... without imposing a permanent triplet partition.

    Only full, without-replacement cell sets are emitted. This intentionally
    avoids replacement padding because Cross-DiT attention has no key-padding
    mask.
    """

    def __init__(
        self,
        dataset: Dataset,
        episodes_per_batch: int = 1,
        num_views: int = 3,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        dynamic_views: bool = False,
        set_views_per_batch: Optional[int] = None,
        min_views: int = 2,
        max_views: int = 5,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"invalid rank={rank} for num_replicas={num_replicas}")
        if int(num_views) < 2:
            raise ValueError("meta-batch episodes require at least two views")
        if int(episodes_per_batch) < 1:
            raise ValueError("episodes_per_batch must be >= 1")
        set_size = dataset.data_args.use_cell_set
        if set_size is None or int(set_size) < 1:
            raise ValueError("meta-batch episodes require data.use_cell_set")

        self.dataset = dataset
        self.episodes_per_batch = int(episodes_per_batch)
        self.num_views = int(num_views)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        # Lightning advances ``batch_sampler.sampler`` at every epoch. This
        # object is already the sampler, so expose that conventional handle
        # instead of silently replaying epoch zero after dataloader reloads.
        self.sampler = self
        self.set_size = int(set_size)
        self.dynamic_views = bool(dynamic_views)
        self.set_views_per_batch = int(
            set_views_per_batch
            if set_views_per_batch is not None
            else self.num_views * self.episodes_per_batch
        )
        self.min_views = int(min_views)
        self.max_views = int(max_views)
        if self.dynamic_views:
            if self.min_views < 2:
                raise ValueError("dynamic meta-batch episodes require min_views >= 2")
            if self.max_views < self.min_views:
                raise ValueError("max_views must be >= min_views")
            if self.set_views_per_batch < self.min_views:
                raise ValueError(
                    "set_views_per_batch must fit at least one dynamic episode"
                )
            self.records = self._build_dynamic_records()
            self._dynamic_device_batches = self._build_dynamic_device_batches()
            usable_device_batches = (
                len(self._dynamic_device_batches) // self.num_replicas
            ) * self.num_replicas
            self._dynamic_device_batches = self._dynamic_device_batches[
                :usable_device_batches
            ]
            episode_index = 0
            for device_batch in self._dynamic_device_batches:
                for episode in device_batch:
                    episode["episode_index"] = episode_index
                    episode_index += 1
            self.num_episodes = episode_index
            self.num_batches = usable_device_batches // self.num_replicas
        else:
            self.records = self._build_records()
            self.num_episodes = sum(
                record["num_chunks"] for record in self.records
            )
            self.num_batches = self.num_episodes // (
                self.num_replicas * self.episodes_per_batch
            )
        if self.num_batches < 1:
            raise ValueError(
                "no complete distributed meta-batch can be formed; reduce set size"
            )
        if self.dynamic_views:
            view_counts = [
                len(episode["views"])
                for device_batch in self._dynamic_device_batches
                for episode in device_batch
            ]
            logger.warning(
                "dynamic meta-batch sampler: episodes=%d batches/rank=%d "
                "set_views/device_batch=%d episode_views=%d..%d set_size=%d "
                "replicas=%d",
                self.num_episodes,
                self.num_batches,
                self.set_views_per_batch,
                min(view_counts),
                max(view_counts),
                self.set_size,
                self.num_replicas,
            )
        else:
            logger.warning(
                "meta-batch sampler: episodes=%d batches/rank=%d views=%d "
                "set_size=%d episodes/device_batch=%d replicas=%d",
                self.num_episodes,
                self.num_batches,
                self.num_views,
                self.set_size,
                self.episodes_per_batch,
                self.num_replicas,
            )

    def _build_dynamic_records(self):
        """Build homogeneous raw-batch view pools for dynamic episodes."""
        records = []
        for ds_index, ds_name in enumerate(self.dataset.dataset_path_map.keys()):
            grouped = defaultdict(list)
            for keys, values in self.dataset.grouped_pert_data_indices[ds_name].items():
                if len(values) == 0:
                    continue
                if self.dataset.control_type[ds_name] is None:
                    raise NotImplementedError(
                        "meta-batch episodes currently support perturbation datasets only"
                    )
                pert_code, cell_code, batch_code = keys
                values = np.asarray(values, dtype=np.int64)
                num_chunks = len(values) // self.set_size
                if num_chunks:
                    grouped[(int(pert_code), int(cell_code))].append(
                        {
                            "batch_code": int(batch_code),
                            "values": values,
                            "num_chunks": int(num_chunks),
                        }
                    )

            for (pert_code, cell_code), batches in sorted(grouped.items()):
                # A single raw batch cannot contribute a batch-pair matching
                # constraint, so leave it to the ordinary cell-set sampler.
                if len(batches) < self.min_views:
                    continue
                batches.sort(key=lambda item: item["batch_code"])
                record_seed = (
                    self.seed
                    + 1000003 * (ds_index + 1)
                    + 9176 * (pert_code + 1)
                    + 131 * (cell_code + 1)
                ) % np.iinfo(np.uint32).max
                records.append(
                    {
                        "ds_name": ds_name,
                        "pert_code": pert_code,
                        "cell_code": cell_code,
                        "batches": batches,
                        "seed": int(record_seed),
                    }
                )
        return records

    def _dynamic_allowed_view_counts(self, active_count: int, remaining: int):
        """Return view counts that leave an exactly packable device budget."""
        upper = min(self.max_views, active_count, remaining)
        allowed = []
        for num_views in range(upper, self.min_views - 1, -1):
            leftover = remaining - num_views
            if leftover == 0 or leftover >= self.min_views:
                allowed.append(num_views)
        return allowed

    @staticmethod
    def _pair_key(left: int, right: int):
        return (left, right) if left < right else (right, left)

    def _select_dynamic_view_batches(self, state, num_views, rng):
        """Select distinct raw batches, prioritizing under-covered pairs."""
        active = [
            index for index, count in enumerate(state["remaining"]) if count > 0
        ]
        if len(active) < num_views:
            raise RuntimeError("dynamic episode requested more views than active batches")

        pair_candidates = []
        for left_offset, left in enumerate(active):
            for right in active[left_offset + 1 :]:
                pair = self._pair_key(left, right)
                pair_candidates.append(
                    (
                        state["pair_coverage"].get(pair, 0),
                        state["batch_usage"][left] + state["batch_usage"][right],
                        float(rng.random()),
                        left,
                        right,
                    )
                )
        _, _, _, left, right = min(pair_candidates)
        selected = [left, right]

        while len(selected) < num_views:
            candidates = []
            for batch_index in active:
                if batch_index in selected:
                    continue
                cross_coverage = sum(
                    state["pair_coverage"].get(
                        self._pair_key(batch_index, prior), 0
                    )
                    for prior in selected
                )
                candidates.append(
                    (
                        cross_coverage,
                        state["batch_usage"][batch_index],
                        -state["remaining"][batch_index],
                        float(rng.random()),
                        batch_index,
                    )
                )
            selected.append(min(candidates)[-1])

        views = []
        for batch_index in selected:
            chunk_index = state["cursor"][batch_index]
            state["cursor"][batch_index] += 1
            state["remaining"][batch_index] -= 1
            state["batch_usage"][batch_index] += 1
            views.append((batch_index, chunk_index))
        for left_offset, left in enumerate(selected):
            for right in selected[left_offset + 1 :]:
                pair = self._pair_key(left, right)
                state["pair_coverage"][pair] = (
                    state["pair_coverage"].get(pair, 0) + 1
                )
        return views

    def _build_dynamic_device_batches(self):
        """Create a deterministic, fixed-budget ragged episode schedule.

        Only cell order changes with the epoch. Keeping this structural plan
        stable makes __len__ exact across epochs and makes DDP resume
        reproduce the same episode boundaries.
        """
        rng = np.random.default_rng(self.seed + 424243)
        states = []
        for record in self.records:
            counts = [batch["num_chunks"] for batch in record["batches"]]
            states.append(
                {
                    "remaining": counts.copy(),
                    "cursor": [0] * len(counts),
                    "batch_usage": [0] * len(counts),
                    "pair_coverage": {},
                }
            )

        device_batches = []
        while True:
            remaining_budget = self.set_views_per_batch
            device_batch = []
            while remaining_budget:
                choices = []
                for record_index, state in enumerate(states):
                    active_count = sum(
                        count > 0 for count in state["remaining"]
                    )
                    allowed = self._dynamic_allowed_view_counts(
                        active_count, remaining_budget
                    )
                    if not allowed:
                        continue
                    # The largest feasible episode uses available raw-batch
                    # diversity; the final slot adapts automatically (e.g.
                    # 5+5+2, 4+4+4, or 3+3+3+3 for a budget of twelve).
                    num_views = allowed[0]
                    choices.append(
                        (
                            -num_views,
                            -sum(state["remaining"]),
                            float(rng.random()),
                            record_index,
                            num_views,
                        )
                    )
                if not choices:
                    break
                _, _, _, record_index, num_views = min(choices)
                views = self._select_dynamic_view_batches(
                    states[record_index], num_views, rng
                )
                device_batch.append(
                    {
                        "record_index": record_index,
                        "views": views,
                    }
                )
                remaining_budget -= num_views
            if remaining_budget:
                # At most one final partial device batch is discarded. Its
                # chunks remain unused rather than replacement padded or split
                # across optimizer steps or DDP ranks.
                break
            device_batches.append(device_batch)

        self.dynamic_pair_coverage = [
            dict(state["pair_coverage"]) for state in states
        ]
        return device_batches

    def _partition_batches(self, batch_arrays, record_seed):
        """Greedily balance cell counts while keeping raw batches disjoint."""
        rng = np.random.default_rng(record_seed)
        order = np.arange(len(batch_arrays))
        rng.shuffle(order)
        order = sorted(order.tolist(), key=lambda idx: -len(batch_arrays[idx][1]))
        view_parts = [[] for _ in range(self.num_views)]
        view_counts = np.zeros(self.num_views, dtype=np.int64)
        for idx in order:
            target = int(np.argmin(view_counts))
            _, values = batch_arrays[idx]
            view_parts[target].append(np.asarray(values, dtype=np.int64))
            view_counts[target] += len(values)
        views = []
        for parts in view_parts:
            if parts:
                views.append(np.concatenate(parts))
            else:
                views.append(np.empty(0, dtype=np.int64))
        return views

    def _build_records(self):
        records = []
        for ds_index, ds_name in enumerate(self.dataset.dataset_path_map.keys()):
            grouped = defaultdict(list)
            for keys, values in self.dataset.grouped_pert_data_indices[ds_name].items():
                if len(values) == 0:
                    continue
                if self.dataset.control_type[ds_name] is None:
                    raise NotImplementedError(
                        "meta-batch episodes currently support perturbation datasets only"
                    )
                pert_code, cell_code, batch_code = keys
                grouped[(int(pert_code), int(cell_code))].append(
                    (int(batch_code), np.asarray(values, dtype=np.int64))
                )

            for (pert_code, cell_code), batch_arrays in sorted(grouped.items()):
                record_seed = (
                    self.seed
                    + 1000003 * (ds_index + 1)
                    + 9176 * (pert_code + 1)
                    + 131 * (cell_code + 1)
                ) % np.iinfo(np.uint32).max
                views = self._partition_batches(batch_arrays, record_seed)
                num_chunks = min(len(view) // self.set_size for view in views)
                if num_chunks < 1:
                    continue
                records.append(
                    {
                        "ds_name": ds_name,
                        "pert_code": pert_code,
                        "cell_code": cell_code,
                        "views": views,
                        "num_chunks": int(num_chunks),
                        "seed": int(record_seed),
                    }
                )
        return records

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self):
        """Return deterministic sampler state for checkpoint integrations."""
        return {
            "epoch": self.epoch,
            "dynamic_views": self.dynamic_views,
            "set_views_per_batch": self.set_views_per_batch,
        }

    def load_state_dict(self, state_dict):
        """Restore the epoch while rejecting an incompatible episode layout."""
        if bool(state_dict.get("dynamic_views", self.dynamic_views)) != self.dynamic_views:
            raise ValueError("sampler checkpoint uses a different dynamic_views mode")
        saved_budget = int(
            state_dict.get("set_views_per_batch", self.set_views_per_batch)
        )
        if saved_budget != self.set_views_per_batch:
            raise ValueError("sampler checkpoint uses a different set-view budget")
        self.set_epoch(int(state_dict.get("epoch", 0)))

    def _iter_dynamic(self) -> Iterator[list[RaggedEpisodeIndex]]:
        epoch_chunks = {}
        for record_index, record in enumerate(self.records):
            for batch_index, batch in enumerate(record["batches"]):
                rng = np.random.default_rng(
                    record["seed"]
                    + 1009 * self.epoch
                    + 53 * (batch_index + 1)
                )
                shuffled = np.asarray(batch["values"], dtype=np.int64).copy()
                rng.shuffle(shuffled)
                usable = batch["num_chunks"] * self.set_size
                epoch_chunks[(record_index, batch_index)] = shuffled[
                    :usable
                ].reshape(batch["num_chunks"], self.set_size)

        device_order = np.arange(len(self._dynamic_device_batches))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + 7919 * self.epoch)
            rng.shuffle(device_order)

        for step in range(self.num_batches):
            plan_index = int(
                device_order[step * self.num_replicas + self.rank]
            )
            device_batch = self._dynamic_device_batches[plan_index]
            flat_batch: list[RaggedEpisodeIndex] = []
            for local_episode_id, episode in enumerate(device_batch):
                record_index = episode["record_index"]
                record = self.records[record_index]
                ds_name = record["ds_name"]
                episode_token = int(
                    self.epoch * self.num_episodes
                    + episode["episode_index"]
                )
                for view_id, (batch_index, chunk_index) in enumerate(
                    episode["views"]
                ):
                    chunk = epoch_chunks[
                        (record_index, batch_index)
                    ][chunk_index]
                    for virtual_index in chunk.tolist():
                        resolved_name, local_index = self.dataset._compute_index(
                            ds_name, virtual_index
                        )
                        if resolved_name != ds_name:
                            raise RuntimeError(
                                "dataset name changed during index resolution"
                            )
                        flat_batch.append(
                            (
                                ds_name,
                                local_index,
                                0,
                                episode_token,
                                local_episode_id,
                                view_id,
                            )
                        )
            expected = self.set_views_per_batch * self.set_size
            if len(flat_batch) != expected:
                raise RuntimeError(
                    f"dynamic meta-batch has {len(flat_batch)} cells, "
                    f"expected {expected}"
                )
            yield flat_batch

    def __iter__(self) -> Iterator[list[EpisodeIndex]]:
        if self.dynamic_views:
            yield from self._iter_dynamic()
            return
        episode_payloads = []
        for record_index, record in enumerate(self.records):
            chunks_by_view = []
            for view_index, values in enumerate(record["views"]):
                rng = np.random.default_rng(
                    record["seed"] + 1009 * self.epoch + 53 * (view_index + 1)
                )
                shuffled = np.asarray(values, dtype=np.int64).copy()
                rng.shuffle(shuffled)
                usable = record["num_chunks"] * self.set_size
                chunks_by_view.append(
                    shuffled[:usable].reshape(record["num_chunks"], self.set_size)
                )
            for chunk_index in range(record["num_chunks"]):
                episode_index = len(episode_payloads)
                episode_payloads.append(
                    (
                        episode_index,
                        record_index,
                        [chunks[chunk_index] for chunks in chunks_by_view],
                    )
                )

        if self.shuffle:
            rng = np.random.default_rng(self.seed + 7919 * self.epoch)
            rng.shuffle(episode_payloads)

        usable = self.num_batches * self.num_replicas * self.episodes_per_batch
        episode_payloads = episode_payloads[:usable]
        cursor = 0
        for _ in range(self.num_batches):
            rank_batches = []
            for _rank in range(self.num_replicas):
                rank_batches.append(
                    episode_payloads[cursor : cursor + self.episodes_per_batch]
                )
                cursor += self.episodes_per_batch
            flat_batch: list[EpisodeIndex] = []
            for episode_index, record_index, view_chunks in rank_batches[self.rank]:
                ds_name = self.records[record_index]["ds_name"]
                episode_token = int(self.epoch * self.num_episodes + episode_index)
                for chunk in view_chunks:
                    for virtual_index in chunk.tolist():
                        resolved_name, local_index = self.dataset._compute_index(
                            ds_name, virtual_index
                        )
                        if resolved_name != ds_name:
                            raise RuntimeError("dataset name changed during index resolution")
                        flat_batch.append((ds_name, local_index, 0, episode_token))
            expected = self.episodes_per_batch * self.num_views * self.set_size
            if len(flat_batch) != expected:
                raise RuntimeError(
                    f"meta-batch has {len(flat_batch)} cells, expected {expected}"
                )
            yield flat_batch

class DistributedCellSetFixPairingBatchSampler(Sampler[Triplet]):
    """Distributedcellsetfixpairingbatchsampler implementation used by the PerturbDiff pipeline."""
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Special method `__init__`."""
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        logger.warning("this sampler is ignoring the drop_last argument")

        self._build_blocks(for_epoch=False)
        self.num_samples = math.ceil(self.total_samples / self.num_replicas)
        logger.warning("after cellset resampling, results in total cells: %s", self.num_samples)

    def __len__(self) -> int:
        """Special method `__len__`."""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Execute `set_epoch` and return values used by downstream logic."""
        self.epoch = int(epoch)
        self._build_blocks(for_epoch=True)

    def _build_blocks(self, for_epoch: bool = False):
        """Execute `_build_blocks` and return values used by downstream logic."""
        if "keep_one_sample" in self.dataset.data_args and self.dataset.data_args.keep_one_sample:
            assert 0, "not supported now"

        rng = np.random.default_rng(self.seed + (self.epoch if for_epoch else 0))

        assert not isinstance(self.dataset.data_args.use_cell_set, omegaconf.dictconfig.DictConfig)
        all_block_indices, all_block_ignored = [], []
        all_block_dsname_idx, dsname_ref = [], []

        self.total_samples = 0
        original_total = 0

        for ds_name in self.dataset.dataset_path_map.keys():
            real_batch_size = self.dataset.data_args.use_cell_set
            if real_batch_size is None:
                real_batch_size = 1

            dsname_ref.append(ds_name)

            groups = self.dataset.grouped_pert_data_indices[ds_name]
            for keys, val in groups.items():
                if len(val) == 0:
                    continue
                original_total += len(val)

                if self.dataset.control_type[ds_name] is None:
                    if "cellxgene" in ds_name.lower():
                        celltype_code = keys
                    else:
                        raise NotImplementedError
                else:
                    pert_code, celltype_code, batch_code = keys

                if self.shuffle and for_epoch:
                    rng.shuffle(val)

                if len(val) % real_batch_size != 0:
                    total_padded = int((len(val) + real_batch_size - 1) // real_batch_size) * real_batch_size - len(val)

                    group_seed = rng.integers(np.iinfo(np.uint32).max, dtype=np.uint32)
                    group_rng = np.random.default_rng(group_seed)
                    pad_samples = group_rng.choice(val, size=total_padded, replace=True)

                    idxs = np.concatenate([val, pad_samples], axis=0)
                    ignored = np.concatenate(
                        [
                            np.zeros(len(val), dtype=np.int32),
                            np.ones(total_padded, dtype=np.int32),
                        ]
                    )
                else:
                    idxs = val
                    ignored = np.zeros(len(val), dtype=np.int32)

                all_block_indices.append(idxs)
                all_block_ignored.append(ignored)
                all_block_dsname_idx.append(np.ones(len(idxs), dtype=np.int32) * len(dsname_ref))

                self.total_samples += len(idxs)

        logger.warning("replicating results in %s from originally %s", self.total_samples, original_total)
        self._all_block_indices = np.concatenate(all_block_indices)
        self._all_block_ignored = np.concatenate(all_block_ignored)
        self._all_block_dsname_idx = np.concatenate(all_block_dsname_idx)
        self.dsname_ref = dsname_ref
        logger.info("dsname_ref: %s", self.dsname_ref)

    def __iter__(self) -> Iterator[Triplet]:
        """Special method `__iter__`."""
        real_batch_size = self.dataset.data_args.use_cell_set
        if real_batch_size is None:
            real_batch_size = 1
        assert len(self._all_block_indices) % real_batch_size == 0
        n_block = len(self._all_block_indices) // real_batch_size
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n_block, generator=g).tolist()
        else:
            logger.warning(
                "still shuffling within batches for validation. Therefore, do not use this sampler for generation"
            )
            indices = list(range(n_block))

        indices = indices[self.rank::self.num_replicas]

        for idx in indices:
            for i in range(idx * real_batch_size, (idx + 1) * real_batch_size):
                gidx, ign, ds_name_idx = self._all_block_indices[i], self._all_block_ignored[i], self._all_block_dsname_idx[i]
                ds_name_idx = ds_name_idx - 1
                ds_name_hint = self.dsname_ref[ds_name_idx]
                ds_name, local_idx = self.dataset._compute_index(ds_name_hint, gidx)
                assert ds_name == ds_name_hint, "ds_name should be consistent"
                yield (ds_name, local_idx, ign)


class CellSetBatchSampler(DistributedCellSetFixPairingBatchSampler):
    """Cellsetbatchsampler implementation used by the PerturbDiff pipeline."""
    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """Special method `__init__`."""
        logger.warning("this sampler should only be used when doing sampling")

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        logger.warning("this sampler is ignoring the drop_last argument")

        self._build_blocks(for_epoch=False)
        assert not dist.is_initialized()
        self.num_samples = self.total_samples

    def __iter__(self) -> Iterator[Triplet]:
        """Special method `__iter__`."""
        real_batch_size = self.dataset.data_args.use_cell_set
        if real_batch_size is None:
            real_batch_size = 1
        assert len(self._all_block_indices) % real_batch_size == 0
        n_block = len(self._all_block_indices) // real_batch_size
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n_block, generator=g).tolist()
        else:
            logger.warning(
                "still shuffling within batches for validation. Therefore, do not use this sampler for generation"
            )
            indices = list(range(n_block))

        for idx in indices:
            for i in range(idx * real_batch_size, (idx + 1) * real_batch_size):
                gidx, ign, ds_name_idx = self._all_block_indices[i], self._all_block_ignored[i], self._all_block_dsname_idx[i]
                ds_name_idx = ds_name_idx - 1
                ds_name_hint = self.dsname_ref[ds_name_idx]
                ds_name, local_idx = self.dataset._compute_index(ds_name_hint, gidx)
                assert ds_name == ds_name_hint, "ds_name should be consistent"
                yield (ds_name, local_idx, ign)

__all__ = [
    "Triplet",
    "EpisodeIndex",
    "RaggedEpisodeIndex",
    "DistributedCellSetFixPairingBatchSampler",
    "CellSetBatchSampler",
    "DistributedMetaBatchEpisodeBatchSampler",
]
