"""Evaluates how accurately each embedding model identifies people from the known-people database.

Scores every human-labelled face in the evaluation set against the reference database under
both embedding models, all four matching strategies, and a sweep of similarity thresholds
below which a face is called unknown. Writes one metrics row per model, strategy, threshold
and scope, and prints the best-F1 threshold per model and strategy.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.identification_study.study_face_embedding import (
    STUDY_EMBEDDING_MODEL_NAMES,
    load_embeddings_by_model,
)
from face_identity.configuration import UNKNOWN_PERSON_LABEL
from face_identity.detection.box_geometry import match_boxes_by_overlap
from face_identity.matching.face_database import FaceDatabase
from face_identity.matching.matching_strategies import MATCHING_STRATEGIES, score_faces_against_people

DEFAULT_REFERENCE_MANIFEST = Path("results/embeddings/embeddings_manifest.csv")
DEFAULT_EVAL_MANIFEST = Path("results/eval_embeddings/eval_embeddings_manifest.csv")
DEFAULT_GROUND_TRUTH_TABLE = Path("real_data/final_human_reviewed_image_identification_table.csv")
DEFAULT_OUTPUT_DIR = Path("results/identification")
METRICS_FILENAME = "identification_metrics.csv"

# Shared across both models and all four strategies; a face scoring below the threshold is unknown.
SIMILARITY_THRESHOLDS = [round(0.15 + 0.05 * step, 2) for step in range(15)]
# A detected face is considered to be the labelled face when their boxes overlap at least this much.
MIN_IOU_FOR_FACE_MATCH = 0.5
# Ground-truth faces the detector never found have no embedding to score. When True they are
# counted as unknown predictions (a miss for a known person, a correct rejection otherwise);
# the undetected_* columns report how many there were so they can be excluded instead.
COUNT_UNDETECTED_GROUND_TRUTH_FACES = True

EVALUATION_GROUPS = ["one_person", "few_people"]
EVALUATION_SCOPES = ["pooled", *EVALUATION_GROUPS]

METRICS_COLUMNS = [
    "model",
    "strategy",
    "threshold",
    "scope",
    "total_faces",
    "known_faces",
    "unknown_faces",
    "correct_identifications",
    "correct_rejections",
    "incorrect_identifications",
    "wrong_identity",
    "known_as_unknown",
    "unknown_as_known",
    "undetected_known",
    "undetected_unknown",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


@dataclass
class EvaluationFace:
    """One human-labelled face, paired with the detected face it matched (if any)."""

    image_id: str
    group: str
    ground_truth_person: str
    embedding_row: int | None


def resolve_group(image_id: str) -> str:
    """Names the evaluation group an image belongs to; every image is named after its group folder."""
    for group in EVALUATION_GROUPS:
        if image_id.startswith(group):
            return group
    raise ValueError(f"Image {image_id} does not belong to any of {EVALUATION_GROUPS}")


def load_reference_databases(manifest_path: Path) -> dict[str, FaceDatabase]:
    """Loads one known-people database per embedding model from the reference manifest.

    The manifest is authoritative: stale embedding files left on disk by earlier builds are
    ignored, as are rows recorded for unreadable images.
    """
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["error"].fillna("") == ""].copy()
    manifest = manifest.sort_values(["person", "embedding_path"], kind="stable").reset_index(drop=True)

    embeddings_by_model = load_embeddings_by_model(list(manifest["embedding_path"]))
    person_name_per_row = manifest["person"].to_numpy()
    return {
        model_name: FaceDatabase.from_labelled_embeddings(person_name_per_row, embeddings)
        for model_name, embeddings in embeddings_by_model.items()
    }


def load_ground_truth_faces(table_path: Path) -> dict[str, list[dict]]:
    """Reads the human-reviewed table into per-image lists of labelled faces with their boxes."""
    table = pd.read_csv(table_path)
    return {row.image_id: json.loads(row.people) for row in table.itertuples()}


def build_evaluation_faces(
    ground_truth_by_image: dict[str, list[dict]],
    detections_manifest: pd.DataFrame,
) -> tuple[list[EvaluationFace], dict[str, np.ndarray], int]:
    """Pairs every labelled face with its detected face, returning the faces to score,
    the detected embeddings they point into, and the count of detections no label claimed."""
    detections_by_image = {image_id: rows for image_id, rows in detections_manifest.groupby("image_id")}

    evaluation_faces = []
    embedding_paths: list[str] = []
    matched_detection_count = 0
    total_detection_count = 0

    for image_id, labelled_faces in ground_truth_by_image.items():
        image_detections = detections_by_image.get(image_id)
        detected_boxes = np.zeros((0, 4), dtype=np.float32)
        if image_detections is not None:
            detected_boxes = image_detections[
                ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
            ].to_numpy(dtype=np.float32)
            total_detection_count += len(detected_boxes)

        labelled_boxes = np.array([face["bbox_xyxy"] for face in labelled_faces], dtype=np.float32)
        matches = match_boxes_by_overlap(labelled_boxes, detected_boxes, MIN_IOU_FOR_FACE_MATCH)
        matched_detection_count += len(matches)

        for labelled_index, labelled_face in enumerate(labelled_faces):
            detected_index = matches.get(labelled_index)
            embedding_row = None
            if detected_index is not None:
                embedding_row = len(embedding_paths)
                embedding_paths.append(image_detections.iloc[detected_index]["embedding_path"])
            evaluation_faces.append(EvaluationFace(
                image_id=image_id,
                group=resolve_group(image_id),
                ground_truth_person=labelled_face["person_name"],
                embedding_row=embedding_row,
            ))

    face_embeddings_by_model = load_embeddings_by_model(embedding_paths)
    unclaimed_detections = total_detection_count - matched_detection_count
    return evaluation_faces, face_embeddings_by_model, unclaimed_detections


def compute_metrics(ground_truth: np.ndarray, predictions: np.ndarray, undetected: np.ndarray) -> dict:
    """Counts every identification outcome and derives accuracy, precision, recall and F1.

    A known face matched to the wrong known person counts against both precision (a wrong
    identity was asserted) and recall (the right person was never found).
    """
    truth_is_known = ground_truth != UNKNOWN_PERSON_LABEL
    prediction_is_known = predictions != UNKNOWN_PERSON_LABEL

    correct_identifications = int((truth_is_known & prediction_is_known & (ground_truth == predictions)).sum())
    wrong_identity = int((truth_is_known & prediction_is_known & (ground_truth != predictions)).sum())
    known_as_unknown = int((truth_is_known & ~prediction_is_known).sum())
    unknown_as_known = int((~truth_is_known & prediction_is_known).sum())
    correct_rejections = int((~truth_is_known & ~prediction_is_known).sum())

    incorrect_identifications = wrong_identity + unknown_as_known
    total_faces = len(ground_truth)
    known_faces = int(truth_is_known.sum())

    accuracy = (correct_identifications + correct_rejections) / total_faces if total_faces else 0.0
    identified_count = correct_identifications + incorrect_identifications
    precision = correct_identifications / identified_count if identified_count else 0.0
    recall = correct_identifications / known_faces if known_faces else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "total_faces": total_faces,
        "known_faces": known_faces,
        "unknown_faces": total_faces - known_faces,
        "correct_identifications": correct_identifications,
        "correct_rejections": correct_rejections,
        "incorrect_identifications": incorrect_identifications,
        "wrong_identity": wrong_identity,
        "known_as_unknown": known_as_unknown,
        "unknown_as_known": unknown_as_known,
        "undetected_known": int((undetected & truth_is_known).sum()),
        "undetected_unknown": int((undetected & ~truth_is_known).sum()),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def score_every_configuration(
    evaluation_faces: list[EvaluationFace],
    face_embeddings_by_model: dict[str, np.ndarray],
    reference_databases: dict[str, FaceDatabase],
    ground_truth_names: np.ndarray,
    face_groups: np.ndarray,
    embedding_rows: np.ndarray,
    undetected: np.ndarray,
) -> list[dict]:
    """Scores every model, strategy, threshold and scope combination into one metrics row each."""
    metrics_rows = []
    for model_name in STUDY_EMBEDDING_MODEL_NAMES:
        database = reference_databases[model_name]
        for strategy in MATCHING_STRATEGIES:
            person_scores = score_faces_against_people(
                face_embeddings_by_model[model_name], database, strategy)
            best_person_index = person_scores.argmax(axis=1)
            best_score = person_scores.max(axis=1)

            # Faces with no embedding score below every threshold, so they are always unknown.
            face_best_person = np.full(len(evaluation_faces), UNKNOWN_PERSON_LABEL, dtype=object)
            face_best_score = np.full(len(evaluation_faces), -np.inf, dtype=np.float64)
            detected = ~undetected
            face_best_person[detected] = database.person_names[best_person_index[embedding_rows[detected]]]
            face_best_score[detected] = best_score[embedding_rows[detected]]

            for threshold in SIMILARITY_THRESHOLDS:
                predictions = np.where(face_best_score >= threshold, face_best_person, UNKNOWN_PERSON_LABEL)
                for scope in EVALUATION_SCOPES:
                    in_scope = (
                        np.ones(len(evaluation_faces), dtype=bool) if scope == "pooled"
                        else face_groups == scope
                    )
                    if not COUNT_UNDETECTED_GROUND_TRUTH_FACES:
                        in_scope &= ~undetected
                    metrics_rows.append({
                        "model": model_name,
                        "strategy": strategy,
                        "threshold": threshold,
                        "scope": scope,
                        **compute_metrics(
                            ground_truth_names[in_scope], predictions[in_scope], undetected[in_scope]
                        ),
                    })
    return metrics_rows


def print_best_thresholds(metrics: pd.DataFrame) -> None:
    """Reports the highest-F1 threshold for each model and strategy, flagging any that
    land on the edge of the swept range (where the real optimum may lie outside it)."""
    pooled = metrics[metrics["scope"] == "pooled"]
    lowest, highest = min(SIMILARITY_THRESHOLDS), max(SIMILARITY_THRESHOLDS)
    print("\nBest threshold per model and strategy (pooled, by F1):")
    for (model, strategy), rows in pooled.groupby(["model", "strategy"], sort=False):
        best = rows.loc[rows["f1"].idxmax()]
        edge_flag = "  <- at grid edge, true optimum may lie beyond it" if best["threshold"] in (lowest, highest) else ""
        print(f"  {model:<20} {strategy:<9} threshold={best['threshold']:.2f}  "
              f"F1={best['f1']:.4f}  precision={best['precision']:.4f}  recall={best['recall']:.4f}  "
              f"correct={best['correct_identifications']}  incorrect={best['incorrect_identifications']}  "
              f"known->unknown={best['known_as_unknown']}  unknown->known={best['unknown_as_known']}{edge_flag}")


def main() -> None:
    """Sweeps every model, strategy and threshold, writing the metrics table and the best points."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--ground-truth-table", type=Path, default=DEFAULT_GROUND_TRUTH_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()

    reference_databases = load_reference_databases(arguments.reference_manifest)
    any_database = reference_databases[STUDY_EMBEDDING_MODEL_NAMES[0]]
    print(f"Reference database: {any_database.person_count} people, "
          f"{len(any_database.embeddings)} face embeddings")

    ground_truth_by_image = load_ground_truth_faces(arguments.ground_truth_table)
    detections_manifest = pd.read_csv(arguments.eval_manifest)
    detections_manifest = detections_manifest[detections_manifest["error"].fillna("") == ""]

    evaluation_faces, face_embeddings_by_model, unclaimed_detections = build_evaluation_faces(
        ground_truth_by_image, detections_manifest)

    ground_truth_names = np.array([face.ground_truth_person for face in evaluation_faces])
    face_groups = np.array([face.group for face in evaluation_faces])
    embedding_rows = np.array(
        [-1 if face.embedding_row is None else face.embedding_row for face in evaluation_faces])
    undetected = embedding_rows < 0

    known_names = set(any_database.person_names)
    unresolved = {
        name for name in ground_truth_names if name != UNKNOWN_PERSON_LABEL and name not in known_names
    }
    if unresolved:
        raise ValueError(f"Ground-truth names missing from the reference database: {sorted(unresolved)}")

    print(f"Labelled faces: {len(evaluation_faces)} "
          f"({int((ground_truth_names != UNKNOWN_PERSON_LABEL).sum())} known, "
          f"{int((ground_truth_names == UNKNOWN_PERSON_LABEL).sum())} unknown)")
    print(f"Labelled faces the detector never found: {int(undetected.sum())} "
          f"({int((undetected & (ground_truth_names != UNKNOWN_PERSON_LABEL)).sum())} of them known people)")
    print(f"Detected faces no label claimed (excluded): {unclaimed_detections}")

    metrics_rows = score_every_configuration(
        evaluation_faces, face_embeddings_by_model, reference_databases,
        ground_truth_names, face_groups, embedding_rows, undetected)

    metrics = pd.DataFrame(metrics_rows, columns=METRICS_COLUMNS)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.output_dir / METRICS_FILENAME
    metrics.to_csv(metrics_path, index=False)
    print(f"\nWrote {len(metrics)} metric rows -> {metrics_path}")

    print_best_thresholds(metrics)


if __name__ == "__main__":
    main()
