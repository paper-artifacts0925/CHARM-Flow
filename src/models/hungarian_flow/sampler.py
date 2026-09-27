"""Without-replacement cell-set sampling for Hungarian-flow training."""

from __future__ import annotations

import logging
from collections import defaultdict, deque

import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler


logger = logging.getLogger(__name__)


class CombinationEpisodeBatchSampler(Sampler):
    """Build complete set-views from raw-batch combinations.

    In core-matching mode, pooled cell sets are independent one-view episodes.
    Optional raw-batch pairing can form multi-view episodes for later ablations.
    """

    def __init__(
        self,
        dataset,
        cells_per_device_batch,
        min_views=2,
        max_views=5,
        batch_pairing=False,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        perturbation_balance_gamma=0.0,
        parent_grouping="celltype",
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if not 0 <= int(rank) < int(num_replicas):
            raise ValueError("invalid distributed rank")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_pairing = bool(batch_pairing)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.perturbation_balance_gamma = float(
            perturbation_balance_gamma
        )
        if not 0.0 <= self.perturbation_balance_gamma <= 1.0:
            raise ValueError(
                "perturbation_balance_gamma must be in [0, 1]"
            )
        self.parent_grouping = str(parent_grouping).strip().lower()
        if self.parent_grouping not in {"celltype", "donor", "donor_celltype"}:
            raise ValueError(
                "parent_grouping must be 'celltype', 'donor', or 'donor_celltype'"
            )
        self.epoch = 0
        self.sampler = self
        self.set_size = int(dataset.data_args.use_cell_set)
        self.cells_per_device_batch = int(cells_per_device_batch)
        if self.set_size < 1 or self.cells_per_device_batch % self.set_size:
            raise ValueError("device cell count must be divisible by set size")
        self.views_per_device_batch = self.cells_per_device_batch // self.set_size
        self.min_views = int(min_views)
        self.max_views = int(max_views)
        minimum_allowed = 2 if self.batch_pairing else 1
        if self.min_views < minimum_allowed or self.max_views < self.min_views:
            raise ValueError(
                "episode view bounds must satisfy "
                f"{minimum_allowed} <= min <= max"
            )
        if self.views_per_device_batch < self.min_views:
            raise ValueError("one device batch cannot fit a complete episode")

        all_sizes = tuple(range(self.min_views, self.max_views + 1))
        reachable = {0}
        for total in range(1, self.views_per_device_batch + 1):
            if any(total - size in reachable for size in all_sizes):
                reachable.add(total)
        self.packable_episode_sizes = tuple(
            size
            for size in all_sizes
            if self.views_per_device_batch - size in reachable
        )
        if not self.packable_episode_sizes:
            raise ValueError("episode sizes cannot fill one device batch")
        self.records = self._build_records()
        self._apply_perturbation_balance()
        self.episodes = self._build_episodes()
        self.device_batches = self._pack_device_batches(self.episodes)
        usable = (len(self.device_batches) // self.num_replicas) * self.num_replicas
        self.device_batches = self.device_batches[:usable]
        self.num_batches = usable // self.num_replicas
        if self.num_batches < 1:
            raise ValueError("no complete combination episode batch can be formed")
        self.strict_nshot_coverage = None
        strict_gate = getattr(self.dataset, "strict_nshot_gate", None)
        if strict_gate is not None:
            if self.parent_grouping != "celltype":
                raise ValueError(
                    "strict N-shot combination sampling only supports "
                    "parent_grouping='celltype'"
                )
            self.strict_nshot_coverage = strict_gate.validate_combination_sampler(
                self
            )
            logger.warning(
                "Strict-N-shot combination coverage: %s",
                self.strict_nshot_coverage,
            )
        used_views = sum(
            episode["num_views"]
            for batch in self.device_batches
            for episode in batch
        )
        scheduled_views = sum(record["num_sets"] for record in self.records)
        available_views = sum(
            record["available_num_sets"] for record in self.records
        )
        logger.warning(
            "Hungarian set sampler: mode=%s records=%d episodes=%d "
            "batches/rank=%d views/device_batch=%d views_used=%d/%d "
            "views_scheduled=%d balance_gamma=%.3f set_size=%d replicas=%d "
            "parent_grouping=%s",
            (
                "celltype_view_pool"
                if self.parent_grouping == "donor"
                else ("batch_pair" if self.batch_pairing else "pooled_random")
            ),
            len(self.records),
            len(self.episodes),
            self.num_batches,
            self.views_per_device_batch,
            used_views,
            available_views,
            scheduled_views,
            self.perturbation_balance_gamma,
            self.set_size,
            self.num_replicas,
            self.parent_grouping,
        )

    def _build_records(self):
        records = []
        for dataset_offset, ds_name in enumerate(self.dataset.dataset_path_map):
            grouped = defaultdict(list)
            for keys, values in self.dataset.grouped_pert_data_indices[ds_name].items():
                if not len(values):
                    continue
                if self.dataset.control_type[ds_name] is None:
                    raise NotImplementedError(
                        "combination episodes support perturbation datasets only"
                    )
                perturbation, cell_line, raw_batch = keys
                if self.parent_grouping == "donor":
                    group_key = (int(perturbation), int(raw_batch))
                    pool_key = int(cell_line)
                elif self.parent_grouping == "donor_celltype":
                    group_key = (
                        int(perturbation),
                        int(raw_batch),
                        int(cell_line),
                    )
                    pool_key = int(raw_batch)
                else:
                    group_key = (int(perturbation), int(cell_line))
                    pool_key = int(raw_batch)
                grouped[group_key].append(
                    (pool_key, np.asarray(values, dtype=np.int64))
                )
            for group_key, pool_arrays in sorted(grouped.items()):
                perturbation = int(group_key[0])
                pool_arrays.sort(key=lambda item: item[0])
                if self.parent_grouping == "donor":
                    parent_id = int(group_key[1])
                    # A donor Parent may have several cell types. Each view is
                    # drawn wholly from one cell-type pool.
                    num_sets = sum(
                        len(values) // self.set_size
                        for _, values in pool_arrays
                    )
                    cell_line = None
                    raw_batch = parent_id
                    seed_parent_id = parent_id
                    view_pool_kind = "celltype"
                elif self.parent_grouping == "donor_celltype":
                    raw_batch = int(group_key[1])
                    cell_line = int(group_key[2])
                    total_cells = sum(len(values) for _, values in pool_arrays)
                    num_sets = total_cells // self.set_size
                    seed_parent_id = (
                        raw_batch * 1000003 + cell_line
                    )
                    view_pool_kind = "exact_donor_celltype"
                else:
                    parent_id = int(group_key[1])
                    total_cells = sum(len(values) for _, values in pool_arrays)
                    num_sets = total_cells // self.set_size
                    cell_line = parent_id
                    raw_batch = None
                    seed_parent_id = parent_id
                    view_pool_kind = "raw_batch"
                if num_sets < self.min_views:
                    continue
                record_seed = (
                    self.seed
                    + 1000003 * (dataset_offset + 1)
                    + 9176 * (perturbation + 1)
                    + 131 * (int(seed_parent_id) + 1)
                ) % np.iinfo(np.uint32).max
                records.append(
                    {
                        "ds_name": ds_name,
                        "perturbation": perturbation,
                        "cell_line": cell_line,
                        "raw_batch": raw_batch,
                        "batches": pool_arrays,
                        "view_pool_kind": view_pool_kind,
                        "num_sets": int(num_sets),
                        "available_num_sets": int(num_sets),
                        "seed": int(record_seed),
                    }
                )
        return records

    def _apply_perturbation_balance(self):
        """Temper condition quotas without replacement.

        A record with ``n`` available sets receives a quota proportional to
        ``n ** (1 - gamma)``. The smallest eligible record is kept in full,
        which fixes the scale and guarantees that no record is oversampled.
        Gamma zero preserves the historical schedule exactly, while gamma one
        gives every record the same quota.
        """
        gamma = self.perturbation_balance_gamma
        if gamma == 0.0 or not self.records:
            return
        minimum = min(
            record["available_num_sets"] for record in self.records
        )
        scale = float(minimum) ** gamma
        for record in self.records:
            available = record["available_num_sets"]
            quota = int(
                np.floor(
                    scale * (float(available) ** (1.0 - gamma))
                    + 1.0e-12
                )
            )
            record["num_sets"] = min(
                available,
                max(self.min_views, quota),
            )

    def _episode_sizes(self, num_sets):
        best = [None] * (int(num_sets) + 1)
        best[0] = []
        for total in range(1, int(num_sets) + 1):
            for size in sorted(self.packable_episode_sizes, reverse=True):
                if size > total:
                    continue
                previous = best[total - size]
                if previous is None:
                    continue
                candidate = previous + [size]
                if (
                    best[total] is None
                    or len(candidate) < len(best[total])
                ):
                    best[total] = candidate
        for used in range(int(num_sets), -1, -1):
            if best[used] is not None:
                return sorted(best[used], reverse=True)
        return []

    def _build_episodes(self):
        episodes = []
        for record_index, record in enumerate(self.records):
            set_start = 0
            for num_views in self._episode_sizes(record["num_sets"]):
                episodes.append(
                    {
                        "record_index": record_index,
                        "set_start": set_start,
                        "num_views": int(num_views),
                        "episode_index": len(episodes),
                    }
                )
                set_start += int(num_views)
        return episodes

    def _pack_device_batches(self, episodes):
        rng = np.random.default_rng(self.seed + 424243)
        by_size = defaultdict(list)
        order = np.arange(len(episodes))
        rng.shuffle(order)
        for episode_index in order.tolist():
            episode = episodes[episode_index]
            by_size[episode["num_views"]].append(episode)
        queues = {size: deque(items) for size, items in by_size.items()}

        sizes = sorted(self.packable_episode_sizes, reverse=True)

        def find_combination(remaining, counts):
            if remaining == 0:
                return []
            for size in sizes:
                if (
                    size > remaining
                    or counts.get(size, 0) == 0
                ):
                    continue
                counts[size] -= 1
                suffix = find_combination(remaining - size, counts)
                counts[size] += 1
                if suffix is not None:
                    return [size] + suffix
            return None

        batches = []
        while True:
            counts = {size: len(queue) for size, queue in queues.items()}
            combination = find_combination(
                self.views_per_device_batch, counts
            )
            if combination is None:
                break
            batches.append(
                [
                    queues[size].popleft()
                    for size in combination
                ]
            )
        return batches

    @staticmethod
    def _pair_key(left, right):
        return (left, right) if left < right else (right, left)

    def _make_record_sets(self, record):
        rng = np.random.default_rng(record["seed"] + 1009 * self.epoch)
        if self.parent_grouping == "donor":
            pool_sets = []
            for cell_line, values in record["batches"]:
                shuffled = values.copy()
                rng.shuffle(shuffled)
                usable = (len(shuffled) // self.set_size) * self.set_size
                if usable:
                    pool_sets.append(
                        {
                            "cell_line": cell_line,
                            "sets": shuffled[:usable].reshape(-1, self.set_size),
                            "cursor": 0,
                        }
                    )
            order = np.arange(len(pool_sets), dtype=np.int64)
            rng.shuffle(order)
            sets = []
            while len(sets) < record["num_sets"]:
                progressed = False
                for pool_index in order.tolist():
                    pool = pool_sets[pool_index]
                    if pool["cursor"] >= len(pool["sets"]):
                        continue
                    sets.append(pool["sets"][pool["cursor"]])
                    pool["cursor"] += 1
                    progressed = True
                    if len(sets) == record["num_sets"]:
                        break
                if not progressed:
                    raise RuntimeError(
                        "donor record set count exceeded homogeneous "
                        "cell-type pools"
                    )
            return np.stack(sets)
        if not self.batch_pairing:
            pooled = np.concatenate(
                [values for _, values in record["batches"]]
            ).copy()
            rng.shuffle(pooled)
            usable = record["num_sets"] * self.set_size
            return pooled[:usable].reshape(
                record["num_sets"], self.set_size
            )
        pools = []
        for raw_batch, values in record["batches"]:
            shuffled = values.copy()
            rng.shuffle(shuffled)
            pools.append(
                {
                    "raw_batch": raw_batch,
                    "values": shuffled,
                    "cursor": 0,
                }
            )
        pair_coverage = defaultdict(int)

        def remaining(pool_index):
            pool = pools[pool_index]
            return len(pool["values"]) - pool["cursor"]

        def draw(pool_index, count):
            pool = pools[pool_index]
            count = min(int(count), remaining(pool_index))
            start = pool["cursor"]
            pool["cursor"] += count
            return pool["values"][start : start + count].tolist()

        sets = []
        for _ in range(record["num_sets"]):
            active = [index for index in range(len(pools)) if remaining(index)]
            selected = []
            if len(active) >= 2:
                candidates = []
                for offset, left in enumerate(active):
                    for right in active[offset + 1 :]:
                        pair = self._pair_key(left, right)
                        candidates.append(
                            (
                                pair_coverage[pair],
                                -(remaining(left) + remaining(right)),
                                float(rng.random()),
                                left,
                                right,
                            )
                        )
                _, _, _, left, right = min(candidates)
                left_count = (self.set_size + 1) // 2
                selected.extend(draw(left, left_count))
                selected.extend(draw(right, self.set_size - len(selected)))
                pair_coverage[self._pair_key(left, right)] += 1
            elif active:
                selected.extend(draw(active[0], self.set_size))

            while len(selected) < self.set_size:
                active = [index for index in range(len(pools)) if remaining(index)]
                if not active:
                    raise RuntimeError("record set count exceeded available cells")
                fill = max(active, key=lambda index: remaining(index))
                selected.extend(draw(fill, self.set_size - len(selected)))
            rng.shuffle(selected)
            sets.append(np.asarray(selected, dtype=np.int64))
        return np.stack(sets)

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "perturbation_balance_gamma": self.perturbation_balance_gamma,
            "parent_grouping": self.parent_grouping,
        }

    def load_state_dict(self, state_dict):
        saved_parent_grouping = str(
            state_dict.get("parent_grouping", "celltype")
        ).strip().lower()
        if saved_parent_grouping != self.parent_grouping:
            raise ValueError(
                "sampler checkpoint uses a different parent_grouping"
            )
        saved_gamma = float(
            state_dict.get(
                "perturbation_balance_gamma",
                self.perturbation_balance_gamma,
            )
        )
        if not np.isclose(
            saved_gamma,
            self.perturbation_balance_gamma,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "sampler checkpoint uses a different "
                "perturbation_balance_gamma"
            )
        self.set_epoch(int(state_dict.get("epoch", 0)))

    def __iter__(self):
        sets_by_record = {
            index: self._make_record_sets(record)
            for index, record in enumerate(self.records)
        }
        order = np.arange(len(self.device_batches))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + 7919 * self.epoch)
            rng.shuffle(order)

        for step in range(self.num_batches):
            plan_index = int(order[step * self.num_replicas + self.rank])
            device_batch = self.device_batches[plan_index]
            result = []
            for local_episode_id, episode in enumerate(device_batch):
                record_index = episode["record_index"]
                record = self.records[record_index]
                episode_token = self.epoch * len(self.episodes) + episode["episode_index"]
                start = episode["set_start"]
                end = start + episode["num_views"]
                for view_id, cell_set in enumerate(
                    sets_by_record[record_index][start:end]
                ):
                    for virtual_index in cell_set.tolist():
                        resolved_name, local_index = self.dataset._compute_index(
                            record["ds_name"], virtual_index
                        )
                        if resolved_name != record["ds_name"]:
                            raise RuntimeError("dataset name changed during index resolution")
                        result.append(
                            (
                                resolved_name,
                                local_index,
                                0,
                                int(episode_token),
                                local_episode_id,
                                view_id,
                            )
                        )
            if len(result) != self.cells_per_device_batch:
                raise RuntimeError("combination sampler emitted an incomplete device batch")
            yield result


__all__ = ["CombinationEpisodeBatchSampler"]
