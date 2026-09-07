"""Reading images from disk and discovering the labelled people images a database is built from."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PersonImage:
    """One reference image of a known person, with the folder names it was labelled by."""

    group: str
    person: str
    path: Path
    relative_path: Path


def read_image_bgr(image_path: Path) -> np.ndarray | None:
    """Reads an image as BGR, returning None when the file is missing or undecodable."""
    return cv2.imread(str(image_path))


def is_image_file(path: Path) -> bool:
    """Says whether a path points at a file with one of the supported image extensions."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def discover_person_images(
    people_images_root: Path,
    include_groups: set[str] | None = None,
) -> list[PersonImage]:
    """Finds every labelled reference image under people_images_root.

    The folder directly containing an image names the person; the folder above that, when there
    is one, names the group. Both `<root>/<person>/<images>` and `<root>/<group>/<person>/<images>`
    layouts therefore work. When include_groups is given, only those groups are returned.
    """
    person_images = []
    for image_path in sorted(people_images_root.rglob("*")):
        if not is_image_file(image_path):
            continue
        relative_path = image_path.relative_to(people_images_root)
        path_parts = relative_path.parts
        person = path_parts[-2] if len(path_parts) >= 2 else ""
        group = path_parts[-3] if len(path_parts) >= 3 else ""
        if include_groups is not None and group not in include_groups:
            continue
        person_images.append(
            PersonImage(group=group, person=person, path=image_path, relative_path=relative_path)
        )
    return person_images


def discover_images_in_groups(images_root: Path, group_names: list[str]) -> list[tuple[str, str, Path]]:
    """Finds every image directly inside each named group folder, as (image_id, group, path).

    The image_id is the file stem, which is the key evaluation ground-truth tables join on.
    """
    images = []
    for group_name in group_names:
        group_directory = images_root / group_name
        for image_path in sorted(group_directory.iterdir()):
            if is_image_file(image_path):
                images.append((image_path.stem, group_name, image_path))
    return images
