"""Face recognition (embedding) module."""

from app.ml.recognizer.interface import FaceEmbedder
from app.ml.recognizer.types import FaceEmbedding
from app.ml.recognizer.arcface import ArcFaceEmbedder

__all__ = ["FaceEmbedder", "FaceEmbedding", "ArcFaceEmbedder"]
