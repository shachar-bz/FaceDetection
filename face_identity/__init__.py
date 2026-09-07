"""Face detection, embedding and closed-set identification against a database of known people."""
from face_identity.configuration import (
    PIPELINE_A_RESNET50_WEBFACE600K,
    PIPELINE_B_SFACE,
    PIPELINE_CONFIGURATIONS,
    UNKNOWN_PERSON_LABEL,
    PipelineConfiguration,
    resolve_pipeline_configuration,
)

__all__ = [
    "PIPELINE_A_RESNET50_WEBFACE600K",
    "PIPELINE_B_SFACE",
    "PIPELINE_CONFIGURATIONS",
    "UNKNOWN_PERSON_LABEL",
    "PipelineConfiguration",
    "resolve_pipeline_configuration",
]
