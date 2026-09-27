"""Deterministic leave-one-cell-line-out support/query episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SUPPORT_SIZES = (8, 16, 32, 64)


@dataclass(frozen=True)
class LeaveOneLineOutEpisode:
    target_line: int
    donor_lines: np.ndarray
    support_perturbations: np.ndarray
    query_perturbations: np.ndarray
    support_size: int
    seed: int

    def validate(self) -> "LeaveOneLineOutEpisode":
        support = np.asarray(self.support_perturbations, dtype=np.int64)
        query = np.asarray(self.query_perturbations, dtype=np.int64)
        donors = np.asarray(self.donor_lines, dtype=np.int64)
        if len(support) != int(self.support_size):
            raise ValueError("episode support length does not match support_size")
        if len(np.unique(support)) != len(support):
            raise ValueError("support perturbations must be unique")
        if len(np.unique(query)) != len(query):
            raise ValueError("query perturbations must be unique")
        if np.intersect1d(support, query).size:
            raise ValueError("support/query perturbations must be strictly disjoint")
        if int(self.target_line) in donors.tolist():
            raise ValueError("target line cannot appear among donor lines")
        if len(donors) > 3:
            raise ValueError("at most three source lines are supported")
        return self


def deterministic_support_order(
    eligible_mask: np.ndarray,
    *,
    target_line: int,
    seed: int,
) -> np.ndarray:
    """Return a reproducible permutation without consulting expression values."""

    eligible = np.flatnonzero(np.asarray(eligible_mask, dtype=bool)).astype(np.int64)
    generator = np.random.default_rng(int(seed) + 1009 * int(target_line))
    return eligible[generator.permutation(len(eligible))]


def build_leave_one_line_out_episode(
    *,
    train_mask: np.ndarray,
    target_line: int,
    support_size: int,
    seed: int,
    max_donors: int = 3,
    support_order: np.ndarray | None = None,
    exclude_query_ids: np.ndarray | None = None,
) -> LeaveOneLineOutEpisode:
    """Build one episode using train conditions only.

    A query is eligible only when its target-line treated pseudobulk is in the
    training split and at least one other line provides that perturbation.
    Validation/test target expression therefore cannot enter either side.
    """

    mask = np.asarray(train_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("train_mask must have shape [L,P]")
    lines, _ = mask.shape
    target_line = int(target_line)
    if not 0 <= target_line < lines:
        raise ValueError("target_line is out of range")
    support_size = int(support_size)
    if support_size not in SUPPORT_SIZES:
        raise ValueError("support_size must be one of 8, 16, 32, or 64")
    donor_support = mask.copy()
    donor_support[target_line] = False
    eligible = mask[target_line] & donor_support.any(axis=0)
    excluded = np.asarray(
        [] if exclude_query_ids is None else exclude_query_ids,
        dtype=np.int64,
    )
    if len(excluded):
        if len(np.unique(excluded)) != len(excluded):
            raise ValueError("exclude_query_ids must be unique")
        if excluded.min() < 0 or excluded.max() >= mask.shape[1]:
            raise ValueError("exclude_query_ids contains an out-of-range perturbation")
        eligible[excluded] = False
    if support_order is None:
        order = deterministic_support_order(
            eligible, target_line=target_line, seed=seed
        )
    else:
        order = np.asarray(support_order, dtype=np.int64)
        if len(np.unique(order)) != len(order):
            raise ValueError("support_order must contain unique perturbations")
        if len(order) and (order.min() < 0 or order.max() >= mask.shape[1]):
            raise ValueError("support_order contains an out-of-range perturbation")
        if len(order) and not eligible[order].all():
            raise ValueError(
                "support_order contains a non-training or explicitly excluded condition"
            )
    if len(order) <= support_size:
        raise ValueError(
            f"line {target_line} has {len(order)} eligible perturbations; "
            f"need more than support_size={support_size}"
        )
    support = order[:support_size]
    query = order[support_size:]
    donor_counts = mask[:, support].sum(axis=1)
    donor_counts[target_line] = -1
    donors = np.argsort(-donor_counts, kind="stable")
    donors = donors[(donor_counts[donors] > 0) & (donors != target_line)]
    donors = donors[: int(max_donors)].astype(np.int64)
    return LeaveOneLineOutEpisode(
        target_line=target_line,
        donor_lines=donors,
        support_perturbations=support,
        query_perturbations=query,
        support_size=support_size,
        seed=int(seed),
    ).validate()


def enumerate_protocol_episodes(
    train_mask: np.ndarray,
    *,
    seed: int = 42,
    support_sizes: tuple[int, ...] = SUPPORT_SIZES,
    exclude_query_ids: np.ndarray | None = None,
) -> list[LeaveOneLineOutEpisode]:
    episodes = []
    for line in range(np.asarray(train_mask).shape[0]):
        for support_size in support_sizes:
            episodes.append(
                build_leave_one_line_out_episode(
                    train_mask=train_mask,
                    target_line=line,
                    support_size=int(support_size),
                    seed=int(seed),
                    exclude_query_ids=exclude_query_ids,
                )
            )
    return episodes


__all__ = [
    "LeaveOneLineOutEpisode",
    "SUPPORT_SIZES",
    "build_leave_one_line_out_episode",
    "deterministic_support_order",
    "enumerate_protocol_episodes",
]
