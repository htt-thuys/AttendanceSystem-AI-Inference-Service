"""Value objects returned by `FaceEmbedder` implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.constants import RECOGNIZER_EMBEDDING_DIM


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """A single face's identity vector.

    The embedding is L2-normalized so cosine similarity reduces to a dot
    product — which is exactly how Qdrant compares the vectors on the BE
    side.
    """

    vector: np.ndarray  # shape (512,), dtype float32, ||v||_2 == 1

    def __post_init__(self) -> None:
        if self.vector.shape != (RECOGNIZER_EMBEDDING_DIM,):
            raise ValueError(
                f"FaceEmbedding must be shape ({RECOGNIZER_EMBEDDING_DIM},), "
                f"got {self.vector.shape}"
            )

    def to_list(self) -> list[float]:
        """JSON-serializable form (used in API response)."""
        return self.vector.astype(float).tolist()
