"""Persistent worker pool for vLLM inference with idle TTL."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import traceback
import uuid
from typing import List, Optional

from .config import GenerationParams, VLLMSettings
from .inference import GenerationResult


def _worker_process_loop(
    settings: VLLMSettings,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    log_level: int,
    idle_ttl_seconds: int,
) -> None:
    """Main loop for the worker process."""
    logger = logging.getLogger("vllm_service.worker")
    logging.basicConfig(
        level=log_level or logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger.setLevel(log_level or logging.INFO)

    engine = None
    last_used = time.monotonic()

    try:
        from .inference import VLLMInferenceEngine

        engine = VLLMInferenceEngine(settings=settings, logger=logger)

        while True:
            try:
                job = request_queue.get(timeout=1)
            except Exception:
                if (
                    idle_ttl_seconds > 0
                    and (time.monotonic() - last_used) >= idle_ttl_seconds
                ):
                    logger.info(
                        "vLLM worker idle for %ds, shutting down.",
                        idle_ttl_seconds,
                    )
                    break
                continue

            if job is None:
                logger.info("Received shutdown signal")
                break

            job_id = job.get("job_id")
            job_type = job.get("type", "generate")

            try:
                if job_type == "generate":
                    prompts = job.get("prompts", [])
                    params_dict = job.get("params", {})
                    params = (
                        GenerationParams(**params_dict)
                        if params_dict
                        else GenerationParams()
                    )

                    results = engine.generate_batch(prompts, params)
                    response_queue.put(
                        {
                            "job_id": job_id,
                            "status": "ok",
                            "results": [_result_to_dict(r) for r in results],
                        }
                    )

                elif job_type == "chat":
                    conversations = job.get("conversations", [])
                    params_dict = job.get("params", {})
                    params = (
                        GenerationParams(**params_dict)
                        if params_dict
                        else GenerationParams()
                    )

                    results = engine.chat_batch(conversations, params)
                    response_queue.put(
                        {
                            "job_id": job_id,
                            "status": "ok",
                            "results": [_result_to_dict(r) for r in results],
                        }
                    )

                else:
                    response_queue.put(
                        {
                            "job_id": job_id,
                            "status": "error",
                            "message": f"Unknown job type: {job_type}",
                        }
                    )

            except Exception as exc:
                response_queue.put(
                    {
                        "job_id": job_id,
                        "status": "error",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

            last_used = time.monotonic()

    finally:
        if engine is not None:
            try:
                import torch

                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.destroy_process_group()
            except Exception:
                pass


def _result_to_dict(result: GenerationResult) -> dict:
    """Convert GenerationResult to dictionary for IPC."""
    return {
        "text": result.text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
    }


def _dict_to_result(data: dict) -> GenerationResult:
    """Convert dictionary back to GenerationResult."""
    return GenerationResult(
        text=data.get("text", ""),
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        finish_reason=data.get("finish_reason", "stop"),
    )


class VLLMWorkerPool:
    """Manages a persistent worker process for vLLM inference."""

    def __init__(
        self,
        settings: VLLMSettings,
        *,
        logger: logging.Logger,
        mp_ctx: mp.context.BaseContext,
        idle_ttl_seconds: int,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self._mp_ctx = mp_ctx
        self._idle_ttl_seconds = idle_ttl_seconds
        self._request_queue: Optional[mp.Queue] = None
        self._response_queue: Optional[mp.Queue] = None
        self._worker_process: Optional[mp.Process] = None

    def ensure_worker(self) -> None:
        """Ensure the worker process is running."""
        if self._worker_process is not None and self._worker_process.is_alive():
            return

        self.logger.info("Starting vLLM worker process...")
        self._request_queue = self._mp_ctx.Queue()
        self._response_queue = self._mp_ctx.Queue()
        self._worker_process = self._mp_ctx.Process(
            target=_worker_process_loop,
            args=(
                self.settings,
                self._request_queue,
                self._response_queue,
                self.logger.getEffectiveLevel(),
                self._idle_ttl_seconds,
            ),
            daemon=False,
        )
        self._worker_process.start()

    def shutdown(self) -> None:
        """Shutdown the worker process gracefully."""
        if self._worker_process is None or not self._worker_process.is_alive():
            return

        self.logger.info("Shutting down vLLM worker...")
        if self._request_queue is not None:
            self._request_queue.put(None)

        self._worker_process.join(timeout=30)
        if self._worker_process.is_alive():
            self.logger.warning("Worker did not shutdown gracefully, terminating...")
            self._worker_process.terminate()
            self._worker_process.join(timeout=5)

    def is_alive(self) -> bool:
        """Check if the worker process is running."""
        return self._worker_process is not None and self._worker_process.is_alive()

    def generate_batch(
        self,
        prompts: List[str],
        params: Optional[GenerationParams] = None,
    ) -> List[GenerationResult]:
        """Run batch generation through the worker."""
        if not prompts:
            return []

        self.ensure_worker()
        if self._request_queue is None or self._response_queue is None:
            raise RuntimeError("vLLM worker is not available")

        job_id = uuid.uuid4().hex
        self._request_queue.put(
            {
                "job_id": job_id,
                "type": "generate",
                "prompts": prompts,
                "params": params.to_dict() if params else {},
            }
        )

        payload = self._response_queue.get()
        return self._process_response(job_id, payload)

    def chat_batch(
        self,
        conversations: List[List[dict]],
        params: Optional[GenerationParams] = None,
    ) -> List[GenerationResult]:
        """Run batch chat through the worker."""
        if not conversations:
            return []

        self.ensure_worker()
        if self._request_queue is None or self._response_queue is None:
            raise RuntimeError("vLLM worker is not available")

        job_id = uuid.uuid4().hex
        self._request_queue.put(
            {
                "job_id": job_id,
                "type": "chat",
                "conversations": conversations,
                "params": params.to_dict() if params else {},
            }
        )

        payload = self._response_queue.get()
        return self._process_response(job_id, payload)

    def _process_response(self, job_id: str, payload: dict) -> List[GenerationResult]:
        """Process the response from the worker."""
        if payload.get("job_id") != job_id:
            raise RuntimeError("vLLM worker returned unexpected job result")

        status = payload.get("status")
        if status != "ok":
            message = payload.get("message", "Unknown error")
            tb = payload.get("traceback")
            self.logger.error("Worker error: %s", message)
            if tb:
                self.logger.debug(tb)
            raise RuntimeError(message)

        results_data = payload.get("results", [])
        return [_dict_to_result(r) for r in results_data]
