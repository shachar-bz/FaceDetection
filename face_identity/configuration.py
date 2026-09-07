"""Every tunable setting for face detection, embedding and identification, in one place.

A `PipelineConfiguration` names one complete, validated identification setup: which embedding
model, which matching strategy, and which similarity threshold separates a named person from
`unknown`. The configurations chosen by the study in `experiments/` are defined here as
constants, so the code, the CLIs and the documentation can never disagree about them.
"""
import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Model weights cache
# ---------------------------------------------------------------------------
# Every model file and the InsightFace pack are cached in one directory, shared by the
# pipelines and the experiments. Override it with this environment variable to keep the
# weights outside the repository.
MODELS_DIRECTORY_ENVIRONMENT_VARIABLE = "FACE_IDENTITY_MODELS_DIR"
DEFAULT_MODELS_DIRECTORY = Path(__file__).resolve().parent.parent / "models"

# ---------------------------------------------------------------------------
# Detection defaults, matching the values the committed results were produced with
# ---------------------------------------------------------------------------
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_DETECTION_INPUT_SIZE = 640
DEFAULT_SCRFD_NMS_THRESHOLD = 0.4
# Crop margin per side, as a fraction of the longer box side, used when drawing annotated boxes.
DEFAULT_CROP_MARGIN_RATIO = 0.5

# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
UNKNOWN_PERSON_LABEL = "unknown"

# Filenames written by the database builder and read back by the identifier.
FACE_DATABASE_FILENAME = "face_database.npz"
FACE_DATABASE_MANIFEST_FILENAME = "face_database_manifest.csv"


@dataclass(frozen=True)
class PipelineConfiguration:
    """One complete identification setup: the embedding model, matching strategy and threshold."""

    name: str
    description: str
    embedding_model_name: str
    matching_strategy: str
    identification_threshold: float
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE
    detection_input_size: int = DEFAULT_DETECTION_INPUT_SIZE


# The configuration selected by the identification study: highest F1 with zero false accepts.
PIPELINE_A_RESNET50_WEBFACE600K = PipelineConfiguration(
    name="pipeline_a_resnet50_webface600k",
    description="ResNet50@WebFace600K (InsightFace buffalo_l), TOP3 @ threshold 0.30",
    embedding_model_name="resnet_webface600k",
    matching_strategy="top3",
    identification_threshold=0.30,
)

# The runner-up configuration, kept for comparison.
PIPELINE_B_SFACE = PipelineConfiguration(
    name="pipeline_b_sface",
    description="OpenCV SFace, TOP2 @ threshold 0.45",
    embedding_model_name="sface",
    matching_strategy="top2",
    identification_threshold=0.45,
)

PIPELINE_CONFIGURATIONS = {
    "a": PIPELINE_A_RESNET50_WEBFACE600K,
    "b": PIPELINE_B_SFACE,
}
DEFAULT_PIPELINE_KEY = "a"


def resolve_pipeline_configuration(pipeline_key: str) -> PipelineConfiguration:
    """Looks up one of the named pipeline configurations, rejecting unknown names."""
    try:
        return PIPELINE_CONFIGURATIONS[pipeline_key]
    except KeyError:
        raise ValueError(
            f"Unknown pipeline {pipeline_key!r}; choose one of {sorted(PIPELINE_CONFIGURATIONS)}"
        ) from None


def resolve_models_directory(models_directory: Path | None = None) -> Path:
    """Names the directory model weights are cached in, creating it if it does not exist."""
    if models_directory is None:
        environment_override = os.environ.get(MODELS_DIRECTORY_ENVIRONMENT_VARIABLE)
        models_directory = Path(environment_override) if environment_override else DEFAULT_MODELS_DIRECTORY
    models_directory = Path(models_directory).expanduser().resolve()
    models_directory.mkdir(parents=True, exist_ok=True)
    return models_directory
