"""Turn face_detection/detect.py's raw detection CSVs into accuracy stats.

Reports, per model variant:
- no_person: false-positive rate (any detection on an image with 0 real faces)
- one_person: exact/miss/over-detect rates
- multiple_people: how detected_count compares to verified_person_count
- breakdowns by brightness_bucket and framing_proxy, to see where the
  model struggles
"""
import argparse
from pathlib import Path

import pandas as pd


def summarize_variant(df: pd.DataFrame) -> None:
    df = df[df["error"].fillna("") == ""].copy()
    df["detected_count"] = df["detected_count"].astype(int)

    no_person = df[df["group"] == "no_person"]
    one_person = df[df["group"] == "one_person"]
    multi = df[df["group"] == "multiple_people"]

    print(f"  no_person (n={len(no_person)}): "
          f"false-positive rate = {(no_person['detected_count'] > 0).mean():.1%}")

    print(f"  one_person (n={len(one_person)}): "
          f"exact (1 face) = {(one_person['detected_count'] == 1).mean():.1%}, "
          f"missed (0 faces) = {(one_person['detected_count'] == 0).mean():.1%}, "
          f"over-detected (>1) = {(one_person['detected_count'] > 1).mean():.1%}")

    multi_recall = (multi["detected_count"] / multi["verified_person_count"]).clip(upper=1.0)
    print(f"  multiple_people (n={len(multi)}): "
          f"mean per-image recall = {multi_recall.mean():.1%}, "
          f"exact count match = {(multi['detected_count'] == multi['verified_person_count']).mean():.1%}, "
          f"zero detected = {(multi['detected_count'] == 0).mean():.1%}")

    has_face = pd.concat([one_person, multi])
    has_face_recall = (has_face["detected_count"] / has_face["verified_person_count"]).clip(upper=1.0)
    print(f"  overall (one_person + multiple_people, n={len(has_face)}): "
          f"mean per-image recall = {has_face_recall.mean():.1%}")

    print("\n  recall by brightness_bucket (one_person + multiple_people):")
    has_face = has_face.copy()
    has_face["recall"] = has_face_recall
    print(has_face.groupby("brightness_bucket")["recall"].mean().to_string())

    print("\n  recall by framing_proxy (one_person + multiple_people):")
    print(has_face.groupby("framing_proxy")["recall"].mean().to_string())

    print("\n  recall by orientation (one_person + multiple_people):")
    print(has_face.groupby("orientation")["recall"].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["short_range", "full_range"])
    args = parser.parse_args()

    for variant_name in args.variants:
        csv_path = args.output_dir / f"detections_{variant_name}.csv"
        print(f"=== {variant_name} ({csv_path.name}) ===")
        summarize_variant(pd.read_csv(csv_path))
        print()


if __name__ == "__main__":
    main()
