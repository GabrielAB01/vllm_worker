"""vllm_worker - Service Python léger pour vLLM avec worker persistant.

Usage:
    from vllm_worker import VLLMInferenceService

    service = VLLMInferenceService()
    result = service.generate("Hello!", temperature=0.7)
    print(result.text)
"""

from .config import GenerationParams, VLLMSettings
from .inference import GenerationResult, VLLMInferenceEngine
from .service import VLLMInferenceService
from .worker_pool import VLLMWorkerPool

__all__ = [
    "VLLMInferenceService",
    "VLLMSettings",
    "GenerationParams",
    "GenerationResult",
    "VLLMInferenceEngine",
    "VLLMWorkerPool",
]

__version__ = "0.1.0"
