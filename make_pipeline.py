"""Regenerates the ready-to-run pipeline folders under pipelines/, one per configuration.

Each folder holds the library, the two CLIs, a requirements.txt and a README, and depends on
nothing else in this repository: someone taking pipeline A or B to production copies that one
folder out, installs its requirements, and runs it. The folders are committed, so they are
there for anyone who clones the repository -- no generation step on their side.

They are generated, never hand-edited. Re-run this after changing anything in face_identity/
or the two CLIs, and commit the result, so a folder in production is never running code the
study did not validate.
"""
import argparse
import shutil
from pathlib import Path

from face_identity.configuration import PIPELINE_CONFIGURATIONS, resolve_pipeline_configuration

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE_ROOT = REPOSITORY_ROOT / "pipelines"

# What a pipeline folder contains, copied verbatim from the repository root.
BUNDLED_PACKAGE_DIRECTORY = "face_identity"
BUNDLED_SCRIPTS = ["identify_faces.py", "build_face_database.py"]
EXCLUDED_FROM_BUNDLE = shutil.ignore_patterns("__pycache__", "*.pyc")
# The whole library ships, including the BlazeFace detector only the benchmark uses. It stays
# because the model registry references it, and it costs the bundle nothing: it imports
# mediapipe inside its constructor, so a bundle that never builds it never needs that package
# -- which is why mediapipe is absent from the generated requirements.txt.

# The line in the copied configuration.py that decides which pipeline the CLIs use by default.
DEFAULT_PIPELINE_KEY_PREFIX = "DEFAULT_PIPELINE_KEY = "
PIPELINE_CONFIGURATIONS_START = "PIPELINE_CONFIGURATIONS = {"
PIPELINE_CONSTANT_NAMES = {
    "a": "PIPELINE_A_RESNET50_WEBFACE600K",
    "b": "PIPELINE_B_SFACE",
}
PIPELINE_DOWNLOADABLE_MODEL_NAMES = {
    "a": [],
    "b": ["sface"],
}
PIPELINE_MODEL_NAMES_PREFIX = "PIPELINE_MODEL_NAMES = "

REQUIREMENTS_TEMPLATE = """\
# Dependencies for the {pipeline_name} standalone bundle.
insightface==1.0.1               # SCRFD-10GF detection and its 5 facial keypoints
opencv-python==5.0.0.93          # image IO, annotation drawing, and the SFace recognizer
numpy==2.5.2
pandas==3.0.5                    # database manifest CSV

# Indirect: pulled in by insightface, pinned here so a resolver change cannot silently
# swap the ONNX runtime or the transform implementation underneath it.
onnxruntime==1.29.0
scikit-image==0.26.0
"""

README_TEMPLATE = """\
# {pipeline_name}

Standalone face-identification pipeline: {description}.

Point it at a folder of labelled photos of the people you care about, then hand it any new
image and it tells you which of those people appear in it. Anyone not in your folder is
reported as `unknown`.

Generated from the FaceDetection project. See that repository for how this configuration was
chosen; this folder is a copy, so edit it there and regenerate rather than editing here.

## Setup

```bash
pip install -r requirements.txt
python -m face_identity.model_downloads --group pipelines
```

## Build the database of people you want recognised

Lay your images out with one folder per person:

```
my_people/
  Person One/  photo1.jpg  photo2.jpg  photo3.jpg
  Person Two/  photo1.jpg  photo2.jpg  photo3.jpg
```

Three to five clear, varied photos per person works well.
Each reference image must contain exactly one detected face: the person named by its folder.
Images with zero faces or multiple faces are skipped with a warning and recorded in the
database manifest, preventing another person's face from being stored under the wrong name.

```bash
python build_face_database.py --people-images-root my_people
```

## Identify people in a new image

```bash
python identify_faces.py path/to/photo.jpg
python identify_faces.py path/to/photo.jpg --annotated-output labelled.jpg
```

To process your own folder of evaluation images recursively, pass the folder instead of one
file. This writes one row per detected face to `identification_results.csv`; unreadable and
no-face images are retained as rows so none disappear silently:

```bash
python identify_faces.py evaluation_images
python identify_faces.py evaluation_images --results-csv my_results.csv \\
  --annotated-output annotated_images
```

## Configuration

Everything tunable -- the embedding model, the matching strategy and the decision threshold --
lives in `face_identity/configuration.py`. This bundle defaults to `{pipeline_key}`:

| Setting | Value |
|---|---|
| Embedding model | `{embedding_model_name}` |
| Matching strategy | `{matching_strategy}` |
| Decision threshold | {identification_threshold} |
| Detection confidence | {min_detection_confidence} |

Raising the threshold means fewer wrong names but more people reported `unknown`; lowering it
does the reverse.
"""


def pin_pipeline_configuration(configuration_path: Path, pipeline_key: str) -> None:
    """Limits a bundled configuration module to its one selected pipeline."""
    lines = configuration_path.read_text(encoding="utf-8").splitlines(keepends=True)
    default_key_was_replaced = False
    for line_index, line in enumerate(lines):
        if line.startswith(DEFAULT_PIPELINE_KEY_PREFIX):
            lines[line_index] = f'{DEFAULT_PIPELINE_KEY_PREFIX}"{pipeline_key}"\n'
            default_key_was_replaced = True
            break
    if not default_key_was_replaced:
        raise ValueError(f"No {DEFAULT_PIPELINE_KEY_PREFIX!r} line found in {configuration_path}")

    try:
        configurations_start_index = next(
            index for index, line in enumerate(lines) if line.rstrip() == PIPELINE_CONFIGURATIONS_START
        )
        configurations_end_index = next(
            index
            for index in range(configurations_start_index + 1, len(lines))
            if lines[index].rstrip() == "}"
        )
    except StopIteration:
        raise ValueError(f"Could not find the pipeline configuration mapping in {configuration_path}") from None

    selected_constant_name = PIPELINE_CONSTANT_NAMES[pipeline_key]
    lines[configurations_start_index:configurations_end_index + 1] = [
        f"{PIPELINE_CONFIGURATIONS_START}\n",
        f'    "{pipeline_key}": {selected_constant_name},\n',
        "}\n",
    ]
    configuration_path.write_text("".join(lines), encoding="utf-8")


def pin_pipeline_model_downloads(model_downloads_path: Path, pipeline_key: str) -> None:
    """Limits a bundle's explicit model downloads to the weights its selected pipeline uses."""
    lines = model_downloads_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_index, line in enumerate(lines):
        if line.startswith(PIPELINE_MODEL_NAMES_PREFIX):
            model_names = PIPELINE_DOWNLOADABLE_MODEL_NAMES[pipeline_key]
            lines[line_index] = f"{PIPELINE_MODEL_NAMES_PREFIX}{model_names!r}\n"
            model_downloads_path.write_text("".join(lines), encoding="utf-8")
            return
    raise ValueError(f"No {PIPELINE_MODEL_NAMES_PREFIX!r} line found in {model_downloads_path}")


def bundle_pipeline(pipeline_key: str, bundle_root: Path) -> Path:
    """Writes one pipeline's standalone folder, replacing any previous bundle of the same name."""
    configuration = resolve_pipeline_configuration(pipeline_key)
    bundle_directory = bundle_root / configuration.name
    if bundle_directory.exists():
        shutil.rmtree(bundle_directory)
    bundle_directory.mkdir(parents=True)

    shutil.copytree(
        REPOSITORY_ROOT / BUNDLED_PACKAGE_DIRECTORY,
        bundle_directory / BUNDLED_PACKAGE_DIRECTORY,
        ignore=EXCLUDED_FROM_BUNDLE,
    )
    for script_name in BUNDLED_SCRIPTS:
        shutil.copy2(REPOSITORY_ROOT / script_name, bundle_directory / script_name)

    bundled_package_directory = bundle_directory / BUNDLED_PACKAGE_DIRECTORY
    pin_pipeline_configuration(bundled_package_directory / "configuration.py", pipeline_key)
    pin_pipeline_model_downloads(bundled_package_directory / "model_downloads.py", pipeline_key)

    (bundle_directory / "requirements.txt").write_text(
        REQUIREMENTS_TEMPLATE.format(pipeline_name=configuration.name), encoding="utf-8")
    (bundle_directory / "README.md").write_text(
        README_TEMPLATE.format(
            pipeline_name=configuration.name,
            pipeline_key=pipeline_key,
            description=configuration.description,
            embedding_model_name=configuration.embedding_model_name,
            matching_strategy=configuration.matching_strategy,
            identification_threshold=configuration.identification_threshold,
            min_detection_confidence=configuration.min_detection_confidence,
        ),
        encoding="utf-8",
    )
    return bundle_directory


def main() -> None:
    """Generates a standalone folder for each requested pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    # No argparse `choices` here: it would validate the default list as a single value.
    # resolve_pipeline_configuration rejects an unknown name with a clearer message anyway.
    parser.add_argument("pipelines", nargs="*", default=None,
                        help=f"Which pipelines to build, from {sorted(PIPELINE_CONFIGURATIONS)} "
                             "(default: all of them)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_ROOT,
                        help="Where the bundle folders are written")
    arguments = parser.parse_args()

    for pipeline_key in arguments.pipelines or sorted(PIPELINE_CONFIGURATIONS):
        bundle_directory = bundle_pipeline(pipeline_key, arguments.output_dir)
        print(f"Bundled pipeline {pipeline_key} -> {bundle_directory}")


if __name__ == "__main__":
    main()
