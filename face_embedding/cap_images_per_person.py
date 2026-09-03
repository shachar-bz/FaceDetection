"""Keeps at most MAX_IMAGES_PER_PERSON images for each person under a people-root directory.

Walks --people-root/<group>/<person>/*, and for any person with more than
MAX_IMAGES_PER_PERSON images, deletes the excess (sorted by filename, keeping
the first MAX_IMAGES_PER_PERSON). Writes a CSV log of every deleted file.
Defaults to a dry run (--apply actually deletes).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_face_database import discover_person_images

MAX_IMAGES_PER_PERSON = 5

LOG_COLUMNS = ["group", "person", "relative_path"]


def find_excess_images(people_root: Path, max_images_per_person: int) -> list[dict]:
    """Groups images by (group, person) and returns a log row for every image beyond max_images_per_person, keeping the first ones by filename."""
    person_images = discover_person_images(people_root)
    print(f"Found {len(person_images)} images under {people_root}")

    by_person: dict[tuple[str, str], list] = {}
    for person_image in person_images:
        by_person.setdefault((person_image.group, person_image.person), []).append(person_image)

    rows = []
    for (group, person), images in by_person.items():
        images.sort(key=lambda pi: pi.relative_path.name)
        for person_image in images[max_images_per_person:]:
            rows.append({
                "group": group,
                "person": person,
                "relative_path": str(person_image.relative_path),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-root", type=Path, required=True,
                         help="Directory laid out as <group>/<person>/<image files>")
    parser.add_argument("--max-images-per-person", type=int, default=MAX_IMAGES_PER_PERSON)
    parser.add_argument("--log-path", type=Path, default=Path("results_scrfd/removed_excess_person_images.csv"))
    parser.add_argument("--apply", action="store_true",
                         help="Actually delete the excess images. Without this flag, only logs them.")
    args = parser.parse_args()

    rows = find_excess_images(args.people_root, args.max_images_per_person)

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(args.log_path, index=False)
    print(f"{len(rows)} excess images flagged (beyond {args.max_images_per_person} per person) -> {args.log_path}")

    if args.apply:
        deleted = 0
        for row in rows:
            image_path = args.people_root / row["relative_path"]
            image_path.unlink(missing_ok=True)
            deleted += 1
        print(f"Deleted {deleted} images.")
    else:
        print("Dry run only (no files deleted). Re-run with --apply to delete the flagged images.")


if __name__ == "__main__":
    main()
