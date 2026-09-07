"""The known-people reference database: every person's embeddings, and how it is stored on disk.

Rows are held sorted by person so each person owns one contiguous slice, which lets a single
matrix product score a batch of query faces against everyone at once.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from face_identity.configuration import FACE_DATABASE_FILENAME


@dataclass
class FaceDatabase:
    """Every known person's reference embeddings, grouped into contiguous per-person row slices."""

    person_names: np.ndarray
    person_slices: list[tuple[int, int]]
    embeddings: np.ndarray
    _person_centroids: np.ndarray | None = field(default=None, repr=False)

    @property
    def person_count(self) -> int:
        """How many distinct people the database holds."""
        return len(self.person_names)

    @property
    def person_centroids(self) -> np.ndarray:
        """Each person's L2-renormalized mean embedding, computed once on first use."""
        if self._person_centroids is None:
            centroids = np.stack(
                [self.embeddings[start:end].mean(axis=0) for start, end in self.person_slices]
            )
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
            self._person_centroids = centroids.astype(np.float32)
        return self._person_centroids

    @classmethod
    def from_labelled_embeddings(
        cls, person_name_per_row: np.ndarray, embeddings: np.ndarray
    ) -> "FaceDatabase":
        """Builds a database from one embedding per row plus the person each row belongs to."""
        person_name_per_row = np.asarray(person_name_per_row)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        sort_order = np.argsort(person_name_per_row, kind="stable")
        person_name_per_row = person_name_per_row[sort_order]
        embeddings = embeddings[sort_order]

        person_names, start_indices, counts = np.unique(
            person_name_per_row, return_index=True, return_counts=True
        )
        person_slices = [(int(start), int(start + count)) for start, count in zip(start_indices, counts)]
        return cls(person_names=person_names, person_slices=person_slices, embeddings=embeddings)


def save_face_database(
    output_directory: Path, person_name_per_row: list[str], embeddings: np.ndarray
) -> Path:
    """Writes one .npz holding every reference embedding and the person each one belongs to."""
    output_directory.mkdir(parents=True, exist_ok=True)
    database_path = output_directory / FACE_DATABASE_FILENAME
    np.savez(
        database_path,
        person_names=np.array(person_name_per_row),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    return database_path


def load_face_database(database_path: Path) -> FaceDatabase:
    """Reads a face database back from the .npz written by save_face_database."""
    with np.load(database_path, allow_pickle=False) as stored:
        return FaceDatabase.from_labelled_embeddings(stored["person_names"], stored["embeddings"])
