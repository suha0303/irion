"""User-operated MAIN/LR input preparation only. No jobs, models, training or deletion."""

from .aggregate import aggregate, audit_aggregate, load_shared_scaler, whole_manifest_paths
from .cases import audit_case
from .contract import check_scope, prepare_policy
from .rejection import check_quality_rejection, publish_quality_rejection

CRITICAL = tuple(
    "ai/learning/collection19/" + name
    for name in (
        "__init__.py",
        "contract.py",
        "cases.py",
        "streams.py",
        "aggregate.py",
        "rejection.py",
        "recovery.py",
    )
) + (
    "ai/learning/admission19.py",
    "ai/learning/wind_metadata.py",
    "ai/learning/history19/data.py",
    "ai/learning/history19/preprocessing.py",
    "ai/learning/history19/preprocessing.json",
    "ai/learning/history19/pilot.py",
    "ai/learning/history19/pilot_connector.py",
    "ai/learning/history19/joint_native.py",
    "ai/learning/history19/duration_profiles.py",
    "ai/learning/run_health19/data.py",
    "ai/src/capstone_ai/mapping.py",
    "ai/src/capstone_ai/source.py",
    "ai/src/capstone_ai/dataset.py",
    "ai/execution/common.py",
    "ai/execution/prepare.py",
)

__all__ = [
    "CRITICAL",
    "aggregate",
    "audit_aggregate",
    "audit_case",
    "check_scope",
    "check_quality_rejection",
    "load_shared_scaler",
    "prepare_policy",
    "publish_quality_rejection",
    "whole_manifest_paths",
]
