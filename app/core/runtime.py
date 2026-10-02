"""
Runtime helpers — ONNX Runtime provider selection and Windows DLL bootstrap.

Two responsibilities:
1. Decide which ONNX Runtime execution providers to try, based on:
   - `settings.USE_GPU`
   - what the installed `onnxruntime` build actually advertises.
2. On Windows, expose pip-installed NVIDIA CUDA wheels (cuBLAS, cuDNN, ...)
   to the loader BEFORE `onnxruntime` is imported, so the GPU provider can
   pick up the DLLs without a system-wide CUDA install.

This module must be imported as early as possible (before any
`import onnxruntime`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from app.core.config import settings

_CUDA_BOOTSTRAPPED: bool = False


def _collect_nvidia_bin_dirs() -> list[str]:
    """Return pip-installed NVIDIA wheel `bin` folders (cuBLAS, cuDNN, ...)."""
    try:
        import site
    except ImportError:
        return []

    seen: set[str] = set()
    bin_dirs: list[str] = []
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_root = Path(site_dir) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for child in nvidia_root.iterdir():
            bin_dir = child / "bin"
            if not bin_dir.is_dir():
                continue
            key = str(bin_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            bin_dirs.append(key)
    return bin_dirs


def bootstrap_windows_cuda_runtime() -> None:
    """Expose pip CUDA wheels to ONNX Runtime on Windows.

    `os.add_dll_directory` alone is not enough for `onnxruntime_providers_cuda.dll`
    to resolve `cublasLt64_12.dll`; prepending NVIDIA `bin` folders to PATH is
    required (verified against onnxruntime-gpu 1.25 + nvidia-*-cu12 wheels).

    Safe to call multiple times; runs once per process.
  """
    global _CUDA_BOOTSTRAPPED
    if _CUDA_BOOTSTRAPPED or sys.platform != "win32":
        return

    bin_dirs = _collect_nvidia_bin_dirs()
    if not bin_dirs:
        return

    # PATH must be updated before the first `import onnxruntime`.
    current_path = os.environ.get("PATH", "")
    prefix = os.pathsep.join(bin_dirs)
    if not current_path.startswith(prefix):
        os.environ["PATH"] = prefix + os.pathsep + current_path

    if hasattr(os, "add_dll_directory"):
        for directory in bin_dirs:
            os.add_dll_directory(directory)

    _CUDA_BOOTSTRAPPED = True


# Run as early as possible when this module is imported on Windows.
bootstrap_windows_cuda_runtime()


def resolve_providers() -> list[str]:
    """Return the ONNX Runtime providers list to pass to InferenceSession.

    The list is ordered by preference. ONNX Runtime tries each one and falls
    back to the next if unavailable, so CPU is always the last resort.
    """
    bootstrap_windows_cuda_runtime()

    # Defer the import until after CUDA runtime paths are registered.
    import onnxruntime as ort  # noqa: WPS433 — intentional local import

    available = set(ort.get_available_providers())
    providers: list[str] = []

    if settings.USE_GPU and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def build_session_options():
    """Build ONNX Runtime SessionOptions with sane defaults."""
    import onnxruntime as ort  # noqa: WPS433

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if settings.ORT_INTRA_OP_THREADS > 0:
        opts.intra_op_num_threads = settings.ORT_INTRA_OP_THREADS
    opts.log_severity_level = 3  # warning+
    return opts
