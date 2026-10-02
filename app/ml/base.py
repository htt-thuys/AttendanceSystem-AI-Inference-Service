"""
Common ONNX Runtime session bootstrap shared by every ML module.

Both `detector` and `recognizer` build their `InferenceSession` through
`OnnxModel` so we keep provider selection, session options, and warm-up
logic in one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.core.runtime import build_session_options, resolve_providers

logger = get_logger(__name__)


class OnnxModel:
    """Thin wrapper around `onnxruntime.InferenceSession`.

    Subclasses are expected to handle preprocessing and postprocessing;
    this class only owns session lifecycle and the raw `forward` call.
    """

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        import onnxruntime as ort  # noqa: WPS433 — local import is intentional

        providers = resolve_providers()
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=build_session_options(),
            providers=providers,
        )
        self._providers_in_use: list[str] = list(self._session.get_providers())
        self._input_names: list[str] = [inp.name for inp in self._session.get_inputs()]
        self._output_names: list[str] = [out.name for out in self._session.get_outputs()]

        logger.info(
            "Loaded ONNX model %s | providers=%s | inputs=%s | outputs=%s",
            model_path.name,
            self._providers_in_use,
            self._input_names,
            self._output_names,
        )

    @property
    def providers(self) -> list[str]:
        return list(self._providers_in_use)

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def forward(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run a single inference pass and return raw outputs in declared order."""
        return self._session.run(self._output_names, feeds)
