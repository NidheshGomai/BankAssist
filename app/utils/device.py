"""
BankAssist RAG — Device Management
=====================================
Detects available compute devices (CUDA / MPS / CPU),
manages device selection, VRAM estimation, and CPU fallback.
All model-loading code should use this module for device resolution.
"""

from __future__ import annotations

import gc
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Device Enum
# ---------------------------------------------------------------------------
class DeviceType(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of compute device availability."""

    device_type: DeviceType
    device_str: str           # e.g. "cuda:0", "mps", "cpu"
    cuda_available: bool
    mps_available: bool
    gpu_name: str             # Empty if CPU
    total_vram_gb: float      # 0.0 if CPU
    free_vram_gb: float       # 0.0 if CPU
    compute_capability: tuple[int, int] | None  # CUDA only
    python_version: str
    platform: str

    @property
    def is_gpu(self) -> bool:
        return self.device_type in (DeviceType.CUDA, DeviceType.MPS)

    @property
    def supports_bfloat16(self) -> bool:
        """BF16 requires Ampere (SM 8.0+) or newer."""
        if self.compute_capability is None:
            return False
        return self.compute_capability >= (8, 0)

    @property
    def supports_4bit_quantization(self) -> bool:
        """BitsAndBytes 4-bit requires CUDA."""
        return self.device_type == DeviceType.CUDA

    def __str__(self) -> str:
        if self.is_gpu:
            return (
                f"DeviceInfo({self.device_str}, GPU={self.gpu_name!r}, "
                f"VRAM={self.free_vram_gb:.1f}/{self.total_vram_gb:.1f} GB free)"
            )
        return f"DeviceInfo(cpu, platform={self.platform!r})"


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------
def detect_device() -> DeviceInfo:
    """
    Probe the system and return a DeviceInfo snapshot.

    This function never raises — it falls back to CPU on any error.
    """
    import sys

    cuda_available = False
    mps_available = False
    gpu_name = ""
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    compute_capability: tuple[int, int] | None = None

    # --- CUDA ---
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / (1024**3)
            free_vram_gb = (
                props.total_memory - torch.cuda.memory_reserved(0)
            ) / (1024**3)
            compute_capability = (props.major, props.minor)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cuda_detection_failed", error=str(exc))

    # --- MPS (Apple Silicon) ---
    if not cuda_available:
        try:
            import torch

            mps_available = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("mps_detection_failed", error=str(exc))

    # Determine device type
    if cuda_available:
        device_type = DeviceType.CUDA
        device_str = "cuda:0"
    elif mps_available:
        device_type = DeviceType.MPS
        device_str = "mps"
    else:
        device_type = DeviceType.CPU
        device_str = "cpu"

    info = DeviceInfo(
        device_type=device_type,
        device_str=device_str,
        cuda_available=cuda_available,
        mps_available=mps_available,
        gpu_name=gpu_name,
        total_vram_gb=round(total_vram_gb, 2),
        free_vram_gb=round(free_vram_gb, 2),
        compute_capability=compute_capability,
        python_version=sys.version,
        platform=platform.platform(),
    )

    logger.info(
        "device_detected",
        device=info.device_str,
        gpu=info.gpu_name or "none",
        vram_free_gb=info.free_vram_gb,
        supports_4bit=info.supports_4bit_quantization,
        supports_bf16=info.supports_bfloat16,
    )

    return info


# ---------------------------------------------------------------------------
# Device Resolution
# ---------------------------------------------------------------------------
def resolve_device(preference: str = "auto") -> str:
    """
    Resolve a device string from a preference setting.

    Args:
        preference: "auto" | "cuda" | "cpu" | "mps"

    Returns:
        Resolved device string, e.g. "cuda:0", "cpu".
    """
    if preference == "auto":
        info = detect_device()
        return info.device_str

    if preference == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            logger.warning(
                "cuda_requested_but_unavailable",
                fallback="cpu",
            )
        except ImportError:
            logger.warning("torch_not_installed", fallback="cpu")
        return "cpu"

    if preference == "mps":
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning(
                "mps_requested_but_unavailable",
                fallback="cpu",
            )
        except ImportError:
            pass
        return "cpu"

    return "cpu"


# ---------------------------------------------------------------------------
# VRAM Estimation
# ---------------------------------------------------------------------------
def estimate_model_vram_gb(
    num_parameters: int,
    dtype_bytes: int = 2,  # bfloat16 default
    quantization: str = "none",
) -> float:
    """
    Rough estimate of VRAM needed to load a model.

    Args:
        num_parameters: Total parameter count.
        dtype_bytes: Bytes per parameter (2=bf16/fp16, 4=fp32).
        quantization: "none" | "8bit" | "4bit"

    Returns:
        Estimated VRAM in GB (includes ~20% overhead buffer).
    """
    if quantization == "4bit":
        bytes_per_param = 0.5  # 4-bit = 0.5 bytes
    elif quantization == "8bit":
        bytes_per_param = 1.0
    else:
        bytes_per_param = dtype_bytes

    raw_gb = (num_parameters * bytes_per_param) / (1024**3)
    return raw_gb * 1.2  # 20% overhead for activations, KV-cache, etc.


def check_sufficient_vram(required_gb: float) -> bool:
    """Return True if the GPU has sufficient free VRAM."""
    info = detect_device()
    if not info.is_gpu:
        return False
    sufficient = info.free_vram_gb >= required_gb
    if not sufficient:
        logger.warning(
            "insufficient_vram",
            required_gb=required_gb,
            free_gb=info.free_vram_gb,
        )
    return sufficient


# ---------------------------------------------------------------------------
# Dtype Resolution
# ---------------------------------------------------------------------------
def get_torch_dtype(device_info: DeviceInfo) -> Any:
    """
    Select the best float dtype for the given device.

    - Ampere+ CUDA → bfloat16
    - Older CUDA → float16
    - MPS → float16
    - CPU → float32
    """
    try:
        import torch

        if device_info.device_type == DeviceType.CUDA:
            if device_info.supports_bfloat16:
                return torch.bfloat16
            return torch.float16
        if device_info.device_type == DeviceType.MPS:
            return torch.float16
        return torch.float32
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Memory Cleanup
# ---------------------------------------------------------------------------
def clear_gpu_cache() -> None:
    """Free unused GPU memory cache."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("gpu_cache_cleared")
    except ImportError:
        pass
