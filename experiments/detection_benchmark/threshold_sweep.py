"""Sweep min_detection_confidence over detections captured at --min-confidence 0.1.

Re-thresholding the saved scores avoids re-running the model for every
candidate confidence value.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from face_identity.detection.detector_registry import BENCHMARK_DETECTOR_NAMES


def counts_at_threshold(scores_json: str, threshold: float) -> int:
    scores = json.loads(scores_json)
    return sum(1 for s in scores if s >= threshold)


def summarize(df: pd.DataFrame, threshold: float) -> dict:
    df = df[df["error"].fillna("") == ""].copy()
    df["count_at_t"] = df["scores"].apply(lambda s: counts_at_threshold(s, threshold))

    no_person = df[df["group"] == "no_person"]
    one_person = df[df["group"] == "one_person"]
    multi = df[df["group"] == "multiple_people"]
    has_face = pd.concat([one_person, multi])
    recall = (has_face["count_at_t"] / has_face["verified_person_count"]).clip(upper=1.0)

    return {
        "threshold": threshold,
        "no_person_fp_rate": (no_person["count_at_t"] > 0).mean(),
        "one_person_exact": (one_person["count_at_t"] == 1).mean(),
        "one_person_missed": (one_person["count_at_t"] == 0).mean(),
        "overall_recall": recall.mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--detectors", nargs="+", default=BENCHMARK_DETECTOR_NAMES,
                        choices=BENCHMARK_DETECTOR_NAMES,
                        help="Which detectors' detections CSVs to score")
    args = parser.parse_args()

    for detector_name in args.detectors:
        csv_path = args.output_dir / f"detections_{detector_name}.csv"
        df = pd.read_csv(csv_path)
        print(f"=== {detector_name} ===")
        rows = [summarize(df, t) for t in args.thresholds]
        report = pd.DataFrame(rows).set_index("threshold")
        report["no_person_fp_rate"] = report["no_person_fp_rate"].map("{:.1%}".format)
        report["one_person_exact"] = report["one_person_exact"].map("{:.1%}".format)
        report["one_person_missed"] = report["one_person_missed"].map("{:.1%}".format)
        report["overall_recall"] = report["overall_recall"].map("{:.1%}".format)
        print(report.to_string())
        print()


if __name__ == "__main__":
    main()
