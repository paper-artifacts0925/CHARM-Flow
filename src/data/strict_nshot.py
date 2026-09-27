"""Artifact-driven strict-N-shot filtering for few-shot deployment training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Mapping

import numpy as np

from src.models.fewshot_parent_adapter import FewShotParentArtifact


@dataclass(frozen=True)
class StrictNShotFilterReport:
    stage: str
    dataset: str
    artifact_sha256: str
    support_size: int
    rows_before: int
    rows_after: int
    treated_before: int
    treated_after: int
    controls_before: int
    controls_after: int
    frozen_audit_treated_removed: int
    deployment_treated_before: int
    deployment_treated_after: int
    deployment_support_conditions_present: int
    deployment_support_cells_after: int
    retained_fraction: float
    per_line: dict[str, dict[str, int]]

    def to_dict(self) -> dict:
        return asdict(self)


class StrictNShotIndexGate:
    """Filter raw split indices before virtual indexing/grouped sampling."""

    def __init__(
        self,
        artifact: FewShotParentArtifact,
        *,
        support_size: int,
        configured_deployment_line_id: int,
        required_deployment_line_name: str = "hepg2",
    ) -> None:
        self.artifact = artifact
        self.support_size = int(support_size)
        self.deployment_line_id = int(configured_deployment_line_id)
        self.required_deployment_line_name = str(required_deployment_line_name)
        self.enabled = True
        self.reports: dict[tuple[str, str], StrictNShotFilterReport] = {}
        self.sampler_reports: dict[str, dict] = {}

        if self.support_size not in tuple(int(x) for x in artifact.support_sizes):
            raise ValueError(
                "strict-N-shot support_size must be declared by the artifact"
            )
        if self.deployment_line_id != int(artifact.heldout_deployment_line_id):
            raise ValueError(
                "strict-N-shot config/artifact held-out deployment line mismatch"
            )
        if self.deployment_line_id != 0:
            raise ValueError("the current strict-N-shot deployment protocol requires line ID 0")
        deployment_name = str(artifact.cellline_names[self.deployment_line_id])
        if deployment_name.casefold() != self.required_deployment_line_name.casefold():
            raise ValueError(
                "the current strict-N-shot deployment protocol requires HepG2 at artifact line ID 0"
            )

        self.train_condition_mask = np.asarray(
            artifact.train_condition_mask, dtype=bool
        ).copy()
        self.validation_condition_mask = np.asarray(
            artifact.validation_condition_mask, dtype=bool
        ).copy()
        self.test_condition_mask = np.asarray(
            artifact.test_condition_mask, dtype=bool
        ).copy()
        self.adapter_train_condition_mask = np.asarray(
            artifact.adapter_train_condition_mask, dtype=bool
        ).copy()
        self.audit_condition_mask = np.asarray(
            artifact.audit_condition_mask, dtype=bool
        ).copy()
        self.audit_perturbation_mask = np.asarray(
            artifact.audit_perturbation_mask, dtype=bool
        ).copy()
        self.episodic_eligible_mask = np.asarray(
            artifact.episodic_eligible_mask, dtype=bool
        ).copy()

        shape = (len(artifact.cellline_names), len(artifact.perturbation_names))
        named_masks = {
            "train": self.train_condition_mask,
            "validation": self.validation_condition_mask,
            "test": self.test_condition_mask,
            "adapter_train": self.adapter_train_condition_mask,
            "audit": self.audit_condition_mask,
            "episodic_eligible": self.episodic_eligible_mask,
        }
        for name, mask in named_masks.items():
            if mask.shape != shape:
                raise ValueError(f"strict-N-shot {name} mask has wrong [L,P] shape")
        if self.audit_perturbation_mask.shape != (shape[1],):
            raise ValueError("strict-N-shot frozen-audit mask has wrong [P] shape")
        if np.any(
            self.train_condition_mask
            & (self.validation_condition_mask | self.test_condition_mask)
        ):
            raise ValueError("strict-N-shot train mask overlaps validation/test")
        if np.any(self.validation_condition_mask & self.test_condition_mask):
            raise ValueError("strict-N-shot validation and test masks overlap")
        expected_adapter_train = (
            self.train_condition_mask
            & ~self.audit_perturbation_mask[None, :]
        )
        if not np.array_equal(
            self.adapter_train_condition_mask, expected_adapter_train
        ):
            raise ValueError(
                "strict-N-shot adapter-train mask is not train minus frozen audit"
            )
        expected_audit = (
            self.train_condition_mask
            & self.audit_perturbation_mask[None, :]
        )
        if not np.array_equal(self.audit_condition_mask, expected_audit):
            raise ValueError("strict-N-shot audit condition mask is inconsistent")

        order = np.asarray(
            artifact.support_order(self.deployment_line_id), dtype=np.int64
        )
        if len(order) <= self.support_size:
            raise ValueError(
                "strict-N-shot artifact needs at least one deployment query after support"
            )
        self.support_perturbation_ids = order[: self.support_size].copy()
        self.query_perturbation_ids = order[self.support_size :].copy()
        if len(np.unique(self.support_perturbation_ids)) != self.support_size:
            raise ValueError("strict-N-shot support perturbations are not unique")
        if np.intersect1d(
            self.support_perturbation_ids, self.query_perturbation_ids
        ).size:
            raise ValueError("strict-N-shot support/query perturbations overlap")
        support_line_mask = np.zeros(shape[1], dtype=bool)
        support_line_mask[self.support_perturbation_ids] = True
        self.deployment_support_mask = support_line_mask
        if not self.train_condition_mask[
            self.deployment_line_id, self.support_perturbation_ids
        ].all():
            raise ValueError("strict-N-shot support contains a non-train condition")
        if not self.adapter_train_condition_mask[
            self.deployment_line_id, self.support_perturbation_ids
        ].all():
            raise ValueError("strict-N-shot support contains frozen audit")
        if not self.episodic_eligible_mask[
            self.deployment_line_id, self.support_perturbation_ids
        ].all():
            raise ValueError("strict-N-shot support contains an ineligible condition")
        if (
            self.validation_condition_mask[
                self.deployment_line_id, self.support_perturbation_ids
            ].any()
            or self.test_condition_mask[
                self.deployment_line_id, self.support_perturbation_ids
            ].any()
        ):
            raise ValueError("strict-N-shot support overlaps validation/test")

        self._line_name_to_id = {
            str(name): index for index, name in enumerate(artifact.cellline_names)
        }
        self._pert_name_to_id = {
            str(name): index for index, name in enumerate(artifact.perturbation_names)
        }
        if len(self._line_name_to_id) != shape[0] or len(self._pert_name_to_id) != shape[1]:
            raise ValueError("strict-N-shot artifact mappings contain duplicate names")

    @property
    def cache_tag(self) -> str:
        return f"strictn_{self.artifact.sha256[:12]}_n{self.support_size}"

    def summary(self) -> dict:
        return {
            "enabled": True,
            "artifact": self.artifact.path,
            "artifact_sha256": self.artifact.sha256,
            "audit_split_sha256": self.artifact.audit_split_sha256,
            "deployment_line_id": self.deployment_line_id,
            "deployment_line_name": str(
                self.artifact.cellline_names[self.deployment_line_id]
            ),
            "support_size": self.support_size,
            "support_perturbation_names": self.artifact.perturbation_names[
                self.support_perturbation_ids
            ].tolist(),
        }

    def validate_runtime_mappings(
        self,
        *,
        runtime_cell_type_dict: Mapping[str, int],
        runtime_perturbation_dict: Mapping[str, int],
    ) -> None:
        runtime_lines = {str(name) for name in runtime_cell_type_dict}
        artifact_lines = set(self._line_name_to_id)
        if runtime_lines != artifact_lines:
            raise ValueError(
                "strict-N-shot runtime/artifact cell-line mappings differ: "
                f"runtime_only={sorted(runtime_lines - artifact_lines)}, "
                f"artifact_only={sorted(artifact_lines - runtime_lines)}"
            )
        runtime_perts = {str(name) for name in runtime_perturbation_dict}
        artifact_perts = set(self._pert_name_to_id)
        if runtime_perts != artifact_perts:
            raise ValueError(
                "strict-N-shot runtime/artifact perturbation mappings differ: "
                f"runtime_only={sorted(runtime_perts - artifact_perts)[:8]}, "
                f"artifact_only={sorted(artifact_perts - runtime_perts)[:8]}"
            )
        if len(set(runtime_cell_type_dict.values())) != len(runtime_cell_type_dict):
            raise ValueError("strict-N-shot runtime cell-line IDs are not unique")
        if len(set(runtime_perturbation_dict.values())) != len(runtime_perturbation_dict):
            raise ValueError("strict-N-shot runtime perturbation IDs are not unique")

    def _local_artifact_maps(self, cache) -> tuple[np.ndarray, np.ndarray]:
        local_lines = np.full(len(cache.cell_type_categories), -1, dtype=np.int64)
        for local_id, name in enumerate(cache.cell_type_categories):
            local_lines[local_id] = self._line_name_to_id.get(str(name), -1)
        local_perts = np.full(len(cache.pert_categories), -1, dtype=np.int64)
        for local_id, name in enumerate(cache.pert_categories):
            local_perts[local_id] = self._pert_name_to_id.get(str(name), -1)
        if (local_lines < 0).any():
            unknown = np.asarray(cache.cell_type_categories)[local_lines < 0]
            raise ValueError(
                f"strict-N-shot dataset has unknown cell lines: {unknown.tolist()}"
            )
        if (local_perts < 0).any():
            unknown = np.asarray(cache.pert_categories)[local_perts < 0]
            raise ValueError(
                f"strict-N-shot dataset has unknown perturbations: {unknown[:8].tolist()}"
            )
        return local_lines, local_perts

    @staticmethod
    def _observed_mask(line_ids, pert_ids, treated, shape) -> np.ndarray:
        observed = np.zeros(shape, dtype=bool)
        observed[line_ids[treated], pert_ids[treated]] = True
        return observed

    def filter_indices(
        self,
        *,
        stage: str,
        dataset_name: str,
        indices: np.ndarray,
        cache,
        control_perturbation: str,
    ) -> np.ndarray:
        stage = str(stage).lower()
        if stage not in {"train", "validation", "test"}:
            raise ValueError(f"strict-N-shot does not recognize split {stage!r}")
        source = np.asarray(indices, dtype=np.int64)
        if source.ndim != 1:
            raise ValueError("strict-N-shot source indices must be one-dimensional")
        if len(source) and (source.min() < 0 or source.max() >= int(cache.n_cells)):
            raise ValueError("strict-N-shot source index is out of H5 bounds")
        if len(np.unique(source)) != len(source):
            raise ValueError("strict-N-shot source indices contain duplicates")

        control_matches = np.flatnonzero(
            np.asarray(cache.pert_categories).astype(str)
            == str(control_perturbation)
        )
        if len(control_matches) != 1:
            raise ValueError("strict-N-shot control perturbation is missing or ambiguous")
        control_code = int(control_matches[0])
        local_lines, local_perts = self._local_artifact_maps(cache)
        line_ids = local_lines[np.asarray(cache.cell_type_codes[source], dtype=np.int64)]
        local_pert_ids = np.asarray(cache.pert_codes[source], dtype=np.int64)
        pert_ids = local_perts[local_pert_ids]
        controls = local_pert_ids == control_code
        treated = ~controls
        observed = self._observed_mask(
            line_ids,
            pert_ids,
            treated,
            self.train_condition_mask.shape,
        )
        split_mask = {
            "train": self.train_condition_mask,
            "validation": self.validation_condition_mask,
            "test": self.test_condition_mask,
        }[stage]
        unexpected = observed & ~split_mask
        if unexpected.any():
            line, perturbation = np.argwhere(unexpected)[0]
            raise ValueError(
                f"strict-N-shot {stage} indices contain condition outside artifact "
                f"{stage} mask: {self.artifact.cellline_names[line]}/"
                f"{self.artifact.perturbation_names[perturbation]}"
            )

        keep = np.ones(len(source), dtype=bool)
        if stage == "train":
            legal_train = self.adapter_train_condition_mask[line_ids, pert_ids]
            deployment = line_ids == self.deployment_line_id
            legal_train &= (
                ~deployment | self.deployment_support_mask[pert_ids]
            )
            keep = controls | (treated & legal_train)
            support_observed = observed[self.deployment_line_id] & self.deployment_support_mask
            missing_support = self.deployment_support_mask & ~support_observed
            if missing_support.any():
                names = self.artifact.perturbation_names[
                    np.flatnonzero(missing_support)
                ].tolist()
                raise ValueError(
                    "strict-N-shot deployment supports are absent from raw train indices: "
                    + ", ".join(names[:8])
                )

        filtered = source[keep]
        after_line_ids = line_ids[keep]
        after_pert_ids = pert_ids[keep]
        after_controls = controls[keep]
        after_treated = ~after_controls
        frozen_rows = treated & self.audit_perturbation_mask[pert_ids]
        deployment = line_ids == self.deployment_line_id
        after_deployment = after_line_ids == self.deployment_line_id
        after_support = (
            after_treated
            & after_deployment
            & self.deployment_support_mask[after_pert_ids]
        )
        present_support = np.unique(after_pert_ids[after_support])
        per_line = {}
        for line_id, line_name in enumerate(self.artifact.cellline_names):
            before_line = line_ids == line_id
            after_line = after_line_ids == line_id
            per_line[str(line_name)] = {
                "rows_before": int(before_line.sum()),
                "rows_after": int(after_line.sum()),
                "treated_before": int((before_line & treated).sum()),
                "treated_after": int((after_line & after_treated).sum()),
                "controls_before": int((before_line & controls).sum()),
                "controls_after": int((after_line & after_controls).sum()),
            }
        report = StrictNShotFilterReport(
            stage=stage,
            dataset=str(dataset_name),
            artifact_sha256=self.artifact.sha256,
            support_size=self.support_size,
            rows_before=int(len(source)),
            rows_after=int(len(filtered)),
            treated_before=int(treated.sum()),
            treated_after=int(after_treated.sum()),
            controls_before=int(controls.sum()),
            controls_after=int(after_controls.sum()),
            frozen_audit_treated_removed=int((frozen_rows & ~keep).sum()),
            deployment_treated_before=int((deployment & treated).sum()),
            deployment_treated_after=int((after_deployment & after_treated).sum()),
            deployment_support_conditions_present=int(len(present_support)),
            deployment_support_cells_after=int(after_support.sum()),
            retained_fraction=float(len(filtered) / max(len(source), 1)),
            per_line=per_line,
        )
        if report.controls_before != report.controls_after:
            raise ValueError("strict-N-shot filtering removed control cells")
        if stage == "train" and report.deployment_support_conditions_present != self.support_size:
            raise ValueError("strict-N-shot did not retain exactly N deployment supports")
        self.reports[(stage, str(dataset_name))] = report
        return filtered

    def validate_combination_sampler(self, sampler) -> dict:
        """Fail if 2x32-style packing drops any selected deployment support."""

        dataset = sampler.dataset
        support_counts = np.zeros(len(self.artifact.perturbation_names), dtype=np.int64)
        grouped_cells_validated = 0
        for ds_name, groups in dataset.grouped_pert_data_indices.items():
            cache = dataset.meta_cache._cache[dataset.dataset_path_map[ds_name]]
            local_lines, local_perts = self._local_artifact_maps(cache)
            perturb_rows = np.asarray(
                dataset.data_indices[ds_name]["perturb"], dtype=np.int64
            )
            seen = np.zeros(len(perturb_rows), dtype=np.int8)
            for (local_pert, local_line, raw_batch), values in groups.items():
                values = np.asarray(values, dtype=np.int64)
                if len(values) and (
                    values.min() < 0 or values.max() >= len(perturb_rows)
                ):
                    raise ValueError(
                        "strict-N-shot grouped cache contains out-of-range virtual indices"
                    )
                np.add.at(seen, values, 1)
                local_rows = perturb_rows[values]
                if len(local_rows) and (
                    not np.all(cache.pert_codes[local_rows] == int(local_pert))
                    or not np.all(cache.cell_type_codes[local_rows] == int(local_line))
                    or not np.all(cache.batch_codes[local_rows] == int(raw_batch))
                ):
                    raise ValueError(
                        "strict-N-shot grouped cache key/value membership mismatch"
                    )
                grouped_cells_validated += len(values)
                line = int(local_lines[int(local_line)])
                perturbation = int(local_perts[int(local_pert)])
                if line == self.deployment_line_id and self.deployment_support_mask[perturbation]:
                    support_counts[perturbation] += len(values)
                if self.audit_perturbation_mask[perturbation]:
                    raise ValueError("frozen-audit record entered strict-N-shot sampler")
                if line == self.deployment_line_id and not self.deployment_support_mask[perturbation]:
                    raise ValueError("deployment non-support record entered strict-N-shot sampler")
            if len(seen) and not np.all(seen == 1):
                raise ValueError(
                    "strict-N-shot grouped cache does not cover each filtered treated row exactly once"
                )

        minimum_cells = int(sampler.min_views) * int(sampler.set_size)
        too_small = self.support_perturbation_ids[
            support_counts[self.support_perturbation_ids] < minimum_cells
        ]
        if len(too_small):
            details = [
                f"{self.artifact.perturbation_names[i]}={support_counts[i]}"
                for i in too_small[:8]
            ]
            raise ValueError(
                f"strict-N-shot support cannot form min_views={sampler.min_views} "
                f"at set_size={sampler.set_size}: " + ", ".join(details)
            )

        packed_record_indices = {
            int(episode["record_index"])
            for batch in sampler.device_batches
            for episode in batch
        }
        packed_support = set()
        for record_index in packed_record_indices:
            record = sampler.records[record_index]
            cache = dataset.meta_cache._cache[
                dataset.dataset_path_map[record["ds_name"]]
            ]
            local_lines, local_perts = self._local_artifact_maps(cache)
            line = int(local_lines[int(record["cell_line"])])
            perturbation = int(local_perts[int(record["perturbation"])])
            if line == self.deployment_line_id and self.deployment_support_mask[perturbation]:
                packed_support.add(perturbation)
        missing_packed = sorted(set(self.support_perturbation_ids.tolist()) - packed_support)
        if missing_packed:
            raise ValueError(
                "strict-N-shot combination packing dropped deployment supports: "
                + ", ".join(
                    str(self.artifact.perturbation_names[i])
                    for i in missing_packed[:8]
                )
            )
        report = {
            "artifact_sha256": self.artifact.sha256,
            "support_size": self.support_size,
            "set_size": int(sampler.set_size),
            "min_views": int(sampler.min_views),
            "max_views": int(sampler.max_views),
            "cells_per_device_batch": int(sampler.cells_per_device_batch),
            "views_per_device_batch": int(sampler.views_per_device_batch),
            "device_batches_per_rank": int(sampler.num_batches),
            "support_conditions_with_raw_cells": int(
                np.sum(support_counts[self.support_perturbation_ids] > 0)
            ),
            "support_conditions_in_packed_batches": int(len(packed_support)),
            "support_cells_available": int(
                support_counts[self.support_perturbation_ids].sum()
            ),
            "grouped_treated_cells_validated": int(grouped_cells_validated),
            "support_min_cells": int(
                support_counts[self.support_perturbation_ids].min()
            ),
            "support_max_cells": int(
                support_counts[self.support_perturbation_ids].max()
            ),
        }
        self.sampler_reports["combination"] = report
        return report


def maybe_build_strict_nshot_gate(
    model_cfg,
    *,
    runtime_cell_type_dict: Mapping[str, int],
    runtime_perturbation_dict: Mapping[str, int],
) -> StrictNShotIndexGate | None:
    """Build the gate only for the enabled few-shot Parent protocol."""

    enabled = bool(
        getattr(
            model_cfg,
            "parent_residual_fewshot_adapter_enabled",
            getattr(model_cfg, "parent_residual_fewshot_enabled", False),
        )
    )
    if not enabled:
        return None
    artifact_path = getattr(
        model_cfg, "parent_residual_fewshot_adapter_artifact_path", None
    )
    artifact_sha256 = getattr(
        model_cfg, "parent_residual_fewshot_adapter_artifact_sha256", None
    )
    deployment_line_id = getattr(
        model_cfg,
        "parent_residual_fewshot_adapter_heldout_deployment_line_id",
        None,
    )
    if artifact_path in (None, "", "null"):
        raise ValueError("few-shot enabled requires an artifact path")
    if artifact_sha256 in (None, "", "null"):
        raise ValueError("few-shot enabled requires an artifact SHA256")
    if deployment_line_id is None:
        raise ValueError("few-shot enabled requires a held-out deployment line ID")
    artifact = FewShotParentArtifact(
        artifact_path,
        expected_sha256=str(artifact_sha256),
        expected_parent_sha256=getattr(
            model_cfg, "parent_residual_artifact_sha256", None
        ),
        expected_normalization_divisor=float(
            getattr(model_cfg, "parent_residual_normalization_divisor", 10.0)
        ),
    )
    gate = StrictNShotIndexGate(
        artifact,
        support_size=int(
            getattr(model_cfg, "parent_residual_fewshot_adapter_support_size")
        ),
        configured_deployment_line_id=int(deployment_line_id),
    )
    gate.validate_runtime_mappings(
        runtime_cell_type_dict=runtime_cell_type_dict,
        runtime_perturbation_dict=runtime_perturbation_dict,
    )
    return gate


def reports_as_json(gate: StrictNShotIndexGate) -> str:
    payload = {
        "protocol": gate.summary(),
        "index_reports": {
            f"{stage}/{dataset}": report.to_dict()
            for (stage, dataset), report in sorted(gate.reports.items())
        },
        "sampler_reports": gate.sampler_reports,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "StrictNShotFilterReport",
    "StrictNShotIndexGate",
    "maybe_build_strict_nshot_gate",
    "reports_as_json",
]
