"""
DeepGuard Temporal Verification & Image Provenance Module
"""

from .temporal_extractor import extract_temporal_info
from .provenance_service import ImageProvenanceService, MockProvenanceProvider
from .temporal_engine import (
    TemporalConsistencyEngine,
    TEMPORALLY_CONSISTENT,
    POSSIBLE_TEMPORAL_MISMATCH,
    TEMPORAL_STATUS_UNVERIFIED,
    INSUFFICIENT_CONTEXT,
    CONTRADICTORY_DATE_EVIDENCE
)

__all__ = [
    "extract_temporal_info",
    "ImageProvenanceService",
    "MockProvenanceProvider",
    "TemporalConsistencyEngine",
    "TEMPORALLY_CONSISTENT",
    "POSSIBLE_TEMPORAL_MISMATCH",
    "TEMPORAL_STATUS_UNVERIFIED",
    "INSUFFICIENT_CONTEXT",
    "CONTRADICTORY_DATE_EVIDENCE"
]
