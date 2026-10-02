"""
Image IO helpers — decode raw bytes into a BGR ndarray for OpenCV.

Decoding is the one operation that can fail on user input, so we wrap it
in a single function with explicit error handling instead of letting cv2
raise silently.
"""

from __future__ import annotations

import cv2
import numpy as np


class InvalidImageError(ValueError):
    """Raised when input bytes cannot be decoded as an image."""


def decode_image_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to an HxWx3 BGR uint8 ndarray.

    Args:
        image_bytes: raw bytes of a JPEG / PNG / WEBP / BMP image.

    Returns:
        ndarray of shape (H, W, 3), dtype=uint8, in BGR order (OpenCV native).

    Raises:
        InvalidImageError: if the buffer is empty, corrupt, or grayscale-only
            and cannot be coerced into a 3-channel BGR image.
    """
    if not image_bytes:
        raise InvalidImageError("Empty image payload.")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise InvalidImageError("Could not decode image bytes (unsupported format or corrupt data).")

    if img.ndim != 3 or img.shape[2] != 3:
        raise InvalidImageError(f"Decoded image has unexpected shape {img.shape}; expected HxWx3.")

    return img
