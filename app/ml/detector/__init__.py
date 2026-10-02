"""Face detection module."""

from app.ml.detector.interface import FaceDetector
from app.ml.detector.types import DetectedFace
from app.ml.detector.scrfd import SCRFDFaceDetector

__all__ = ["FaceDetector", "DetectedFace", "SCRFDFaceDetector"]
