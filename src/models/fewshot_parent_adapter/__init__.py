"""Leakage-safe few-shot target-cell-line Parent adaptation."""

from .artifact import (
    DEFAULT_AUDIT_SALT,
    FewShotParentArtifact,
    SCHEMA_VERSION,
    build_fewshot_parent_artifact,
    sha256_file,
)
from .config import FewShotParentAdapterConfig
from .episodes import (
    LeaveOneLineOutEpisode,
    SUPPORT_SIZES,
    build_leave_one_line_out_episode,
    enumerate_protocol_episodes,
)
from .module import (
    FewShotParentAdapter,
    FewShotParentOutput,
    ZeroInitReliabilityGate,
)
from .ridge import (
    FewShotCandidate,
    FewShotRidgeFit,
    build_leakage_safe_equal_line_prior,
    fit_fewshot_ridge_candidate,
)

__all__ = [
    "FewShotCandidate",
    "DEFAULT_AUDIT_SALT",
    "FewShotParentAdapter",
    "FewShotParentAdapterConfig",
    "FewShotParentArtifact",
    "FewShotParentOutput",
    "FewShotRidgeFit",
    "LeaveOneLineOutEpisode",
    "SCHEMA_VERSION",
    "SUPPORT_SIZES",
    "ZeroInitReliabilityGate",
    "build_fewshot_parent_artifact",
    "build_leakage_safe_equal_line_prior",
    "build_leave_one_line_out_episode",
    "enumerate_protocol_episodes",
    "fit_fewshot_ridge_candidate",
    "sha256_file",
]
