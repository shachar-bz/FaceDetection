"""Score a detections CSV against per-group success rules and trace TP/FP/TN/FN.

Ground truth is the image's group (folder): no_person is a true negative
(no face present), one_person and multiple_people are true positives (a
face is present). Per-image success/failure is decided by group-specific
rules on detected_count:

- no_person:       success (TN) if detected_count <= 1, else fail (FP)
- one_person:       success (TP) if detected_count == 1, else fail (FN)
- multiple_people:  success (TP) if detected_count > 1,  else fail (FN)

(The no_person tolerance for exactly 1 detection matches the known MIAP
"person" ground-truth noise documented in README.md — a single stray box
isn't treated as a real false positive.)

Writes a per-image CSV with an `outcome` column (TP/FP/TN/FN) and prints
the resulting confusion matrix and accuracy per variant.
"""
import argparse
from pathlib import Path

import pandas as pd


def classify(row: pd.Series) -> str:
    group = row["group"]
    count = row["detected_count"]
    if group == "no_person":
        return "TN" if count <= 1 else "FP"
    if group == "one_person":
        return "TP" if count == 1 else "FN"
    if group == "multiple_people":
        return "TP" if count > 1 else "FN"
    raise ValueError(f"unknown group: {group}")


def score_variant(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["error"].fillna("") == ""].copy()
    df["detected_count"] = df["detected_count"].astype(int)
    df["outcome"] = df.apply(classify, axis=1)
    return df


def metrics(outcomes: pd.Series) -> dict:
    counts = outcomes.value_counts().reindex(["TP", "TN", "FP", "FN"], fill_value=0)
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    total = tp + tn + fp + fn
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
    }


def per_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    per_group = df.groupby("group")["outcome"].apply(metrics).unstack()
    return per_group.reindex(["no_person", "one_person", "multiple_people"])


def report(df: pd.DataFrame, per_group: pd.DataFrame, variant_name: str) -> None:
    overall = metrics(df["outcome"])

    print(f"=== {variant_name} ===")
    print(f"  TP={overall['TP']}  TN={overall['TN']}  FP={overall['FP']}  FN={overall['FN']}  (n={len(df)})")
    print(f"  accuracy  = {overall['accuracy']:.1%}")
    print(f"  precision = {overall['precision']:.1%}")
    print(f"  recall    = {overall['recall']:.1%}")

    print("\n  by group:")
    display = per_group.copy()
    for col in ("accuracy", "precision", "recall"):
        display[col] = display[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a")
    print(display.to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                         help="Directory containing detections_<variant>.csv from detect.py")
    parser.add_argument("--variants", nargs="+", default=["short_range", "full_range"])
    args = parser.parse_args()

    for variant_name in args.variants:
        csv_path = args.results_dir / f"detections_{variant_name}.csv"
        df = score_variant(pd.read_csv(csv_path))

        out_csv = args.results_dir / f"accuracy_{variant_name}.csv"
        df.to_csv(out_csv, index=False)

        per_group = per_group_metrics(df)
        per_group_csv = args.results_dir / f"accuracy_by_group_{variant_name}.csv"
        per_group.to_csv(per_group_csv)

        report(df, per_group, variant_name)
        print(f"  wrote per-image outcomes -> {out_csv}")
        print(f"  wrote per-group metrics -> {per_group_csv}\n")


if __name__ == "__main__":
    main()
