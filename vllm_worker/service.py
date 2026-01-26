"""Main service interface for vLLM inference."""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from typing import List, Optional, Union

from .config import GenerationParams, VLLMSettings
from .inference import GenerationResult
from .worker_pool import VLLMWorkerPool


class VLLMInferenceService:
    """Service wrapper for vLLM inference with persistent worker management.

    The service maintains a worker process that loads the model once and handles
    multiple inference requests. The worker automatically shuts down after being
    idle for the configured TTL.

    Example usage:
        service = VLLMInferenceService()

        # Simple text generation
        result = service.generate("What is the capital of France?")
        print(result.text)

        # With custom parameters
        result = service.generate(
            "Write a poem about the sea",
            temperature=0.9,
            max_tokens=500,
        )

        # Batch generation
        prompts = ["Hello", "How are you?", "What's up?"]
        results = service.generate_batch(prompts, temperature=0.7)

        # Chat completion
        messages = [
            {"role": "user", "content": "Hello!"},
        ]
        result = service.chat(messages)
        print(result.text)

        # Warmup (pre-load model)
        service.warmup()

        # Shutdown when done
        service.shutdown()
    """

    def __init__(
        self,
        settings: Optional[VLLMSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.settings = settings or VLLMSettings.from_env()

        self.logger = logger or logging.getLogger("vllm_service")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self._mp_ctx = mp.get_context("spawn")
        self._worker_lock = threading.Lock()
        self._last_used = 0.0

        self._worker_pool = VLLMWorkerPool(
            settings=self.settings,
            logger=self.logger,
            mp_ctx=self._mp_ctx,
            idle_ttl_seconds=self.settings.idle_ttl_seconds,
        )

    def _update_last_used(self) -> None:
        self._last_used = time.monotonic()

    def warmup(self) -> None:
        """Pre-load the model by starting the worker process.

        Call this method to ensure the model is loaded before the first
        inference request. This avoids latency on the first actual request.
        """
        with self._worker_lock:
            self._worker_pool.ensure_worker()
            self._update_last_used()
        self.logger.info("vLLM service warmed up and ready")

    def shutdown(self) -> None:
        """Shutdown the worker process gracefully."""
        with self._worker_lock:
            self._worker_pool.shutdown()
        self.logger.info("vLLM service shut down")

    def is_ready(self) -> bool:
        """Check if the worker is running and ready."""
        return self._worker_pool.is_alive()

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """Generate text from a single prompt.

        Args:
            prompt: The input text prompt.
            temperature: Sampling temperature (0.0 = deterministic, higher = more random).
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter (-1 to disable).
            max_tokens: Maximum number of tokens to generate.
            repetition_penalty: Penalty for repeating tokens.
            stop: List of stop sequences.
            presence_penalty: Penalty for token presence.
            frequency_penalty: Penalty for token frequency.
            seed: Random seed for reproducibility.

        Returns:
            GenerationResult containing the generated text and metadata.
        """
        params = GenerationParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )
        results = self.generate_batch([prompt], params=params)
        return results[0]

    def generate_batch(
        self,
        prompts: List[str],
        *,
        params: Optional[GenerationParams] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[GenerationResult]:
        """Generate text from multiple prompts in a batch.

        Args:
            prompts: List of input text prompts.
            params: GenerationParams object (alternative to individual params).
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.
            max_tokens: Maximum number of tokens to generate.
            repetition_penalty: Penalty for repeating tokens.
            stop: List of stop sequences.
            presence_penalty: Penalty for token presence.
            frequency_penalty: Penalty for token frequency.
            seed: Random seed for reproducibility.

        Returns:
            List of GenerationResult objects.
        """
        if not prompts:
            return []

        if params is None:
            params = GenerationParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
            )

        with self._worker_lock:
            results = self._worker_pool.generate_batch(prompts, params)
            self._update_last_used()
            return results

    def chat(
        self,
        messages: List[dict],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        """Generate a chat completion from a conversation.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                      Roles: "system", "user", "assistant"
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.
            max_tokens: Maximum number of tokens to generate.
            repetition_penalty: Penalty for repeating tokens.
            stop: List of stop sequences.
            presence_penalty: Penalty for token presence.
            frequency_penalty: Penalty for token frequency.
            seed: Random seed for reproducibility.

        Returns:
            GenerationResult containing the assistant's response.

        Example:
            result = service.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ])
        """
        params = GenerationParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )
        results = self.chat_batch([messages], params=params)
        return results[0]

    def chat_batch(
        self,
        conversations: List[List[dict]],
        *,
        params: Optional[GenerationParams] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[GenerationResult]:
        """Generate chat completions for multiple conversations.

        Args:
            conversations: List of conversation message lists.
            params: GenerationParams object (alternative to individual params).
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.
            max_tokens: Maximum number of tokens to generate.
            repetition_penalty: Penalty for repeating tokens.
            stop: List of stop sequences.
            presence_penalty: Penalty for token presence.
            frequency_penalty: Penalty for token frequency.
            seed: Random seed for reproducibility.

        Returns:
            List of GenerationResult objects.
        """
        if not conversations:
            return []

        if params is None:
            params = GenerationParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                seed=seed,
            )

        with self._worker_lock:
            results = self._worker_pool.chat_batch(conversations, params)
            self._update_last_used()
            return results

    def __enter__(self) -> "VLLMInferenceService":
        """Context manager entry."""
        self.warmup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
