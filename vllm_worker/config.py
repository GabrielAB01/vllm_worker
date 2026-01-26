"""Environment-backed settings for the lightweight vLLM service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _read_float(env_key: str, default: float) -> float:
    value = os.getenv(env_key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {env_key}: {value}") from exc


def _read_int(env_key: str, default: int) -> int:
    value = os.getenv(env_key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {env_key}: {value}") from exc


def _read_bool(env_key: str, default: bool) -> bool:
    value = os.getenv(env_key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class VLLMSettings:
    """Configuration for the vLLM inference service."""

    model_path: Path
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    idle_ttl_seconds: int = 1800
    trust_remote_code: bool = True

    # Default generation parameters
    default_temperature: float = 0.7
    default_top_p: float = 0.95
    default_top_k: int = -1  # -1 means disabled
    default_max_tokens: int = 2048
    default_repetition_penalty: float = 1.0

    # Remote mode settings
    vllm_mode: str = "local"  # "local" or "remote"
    vllm_api_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "VLLMSettings":
        """Create settings from environment variables."""
        model_value = os.getenv("VLLM_MODEL_PATH")
        if not model_value:
            raise ValueError("VLLM_MODEL_PATH environment variable must be set")
        model_path = Path(model_value)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        vllm_mode = os.getenv("VLLM_MODE", "local").lower()
        if vllm_mode not in {"local", "remote"}:
            raise ValueError(f"VLLM_MODE must be 'local' or 'remote', got: {vllm_mode}")

        vllm_api_url = os.getenv("VLLM_API_URL")
        if vllm_mode == "remote" and not vllm_api_url:
            raise ValueError("VLLM_API_URL must be set when VLLM_MODE is 'remote'")

        return cls(
            model_path=model_path,
            gpu_memory_utilization=_read_float("VLLM_GPU_MEMORY_UTILIZATION", 0.9),
            max_model_len=_read_int("VLLM_MAX_MODEL_LEN", 32768),
            tensor_parallel_size=_read_int("VLLM_TENSOR_PARALLEL_SIZE", 1),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            idle_ttl_seconds=_read_int("VLLM_IDLE_TTL_SECONDS", 1800),
            trust_remote_code=_read_bool("VLLM_TRUST_REMOTE_CODE", True),
            default_temperature=_read_float("VLLM_DEFAULT_TEMPERATURE", 0.7),
            default_top_p=_read_float("VLLM_DEFAULT_TOP_P", 0.95),
            default_top_k=_read_int("VLLM_DEFAULT_TOP_K", -1),
            default_max_tokens=_read_int("VLLM_DEFAULT_MAX_TOKENS", 2048),
            default_repetition_penalty=_read_float("VLLM_DEFAULT_REPETITION_PENALTY", 1.0),
            vllm_mode=vllm_mode,
            vllm_api_url=vllm_api_url,
        )


@dataclass
class GenerationParams:
    """Parameters for text generation."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def merge_with_defaults(self, settings: VLLMSettings) -> "GenerationParams":
        """Return a new GenerationParams with defaults filled in from settings."""
        return GenerationParams(
            temperature=self.temperature if self.temperature is not None else settings.default_temperature,
            top_p=self.top_p if self.top_p is not None else settings.default_top_p,
            top_k=self.top_k if self.top_k is not None else settings.default_top_k,
            max_tokens=self.max_tokens if self.max_tokens is not None else settings.default_max_tokens,
            repetition_penalty=self.repetition_penalty if self.repetition_penalty is not None else settings.default_repetition_penalty,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            seed=self.seed,
        )
