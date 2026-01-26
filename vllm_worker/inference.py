"""vLLM wrapper for text generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .config import GenerationParams, VLLMSettings


@dataclass
class GenerationResult:
    """Result of a single generation."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class VLLMInferenceEngine:
    """Minimal vLLM wrapper for text generation."""

    def __init__(
        self,
        settings: VLLMSettings,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("vllm_service.engine")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

        self.llm = None
        self.client = None
        self.tokenizer = None

        if self.settings.vllm_mode == "local":
            self._init_local_vllm()
        elif self.settings.vllm_mode == "remote":
            self._init_remote_vllm()
        else:
            raise ValueError(f"Invalid vllm_mode: {self.settings.vllm_mode}")

    def _init_local_vllm(self) -> None:
        """Initialize local vLLM instance."""
        from vllm import LLM

        self.logger.info("Loading local vLLM engine from %s", self.settings.model_path)
        self.llm = LLM(
            model=str(self.settings.model_path),
            trust_remote_code=self.settings.trust_remote_code,
            dtype=self.settings.dtype,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            max_model_len=self.settings.max_model_len,
            tensor_parallel_size=self.settings.tensor_parallel_size,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.logger.info("Local vLLM engine ready")

    def _init_remote_vllm(self) -> None:
        """Initialize remote vLLM client via OpenAI API."""
        from openai import OpenAI

        if not self.settings.vllm_api_url:
            raise ValueError("vllm_api_url must be provided for remote mode")

        self.logger.info("Connecting to remote vLLM server at %s", self.settings.vllm_api_url)
        self.client = OpenAI(
            base_url=self.settings.vllm_api_url,
            api_key="EMPTY",
        )
        self.logger.info("Remote vLLM client ready")

    def _build_sampling_params(self, params: GenerationParams):
        """Build vLLM SamplingParams from GenerationParams."""
        from vllm import SamplingParams

        merged = params.merge_with_defaults(self.settings)
        kwargs = {
            "temperature": merged.temperature,
            "top_p": merged.top_p,
            "max_tokens": merged.max_tokens,
            "repetition_penalty": merged.repetition_penalty,
        }

        if merged.top_k is not None and merged.top_k > 0:
            kwargs["top_k"] = merged.top_k

        if merged.stop:
            kwargs["stop"] = merged.stop

        if merged.presence_penalty is not None:
            kwargs["presence_penalty"] = merged.presence_penalty

        if merged.frequency_penalty is not None:
            kwargs["frequency_penalty"] = merged.frequency_penalty

        if merged.seed is not None:
            kwargs["seed"] = merged.seed

        return SamplingParams(**kwargs)

    def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
    ) -> GenerationResult:
        """Generate text from a single prompt."""
        results = self.generate_batch([prompt], params)
        return results[0]

    def generate_batch(
        self,
        prompts: List[str],
        params: Optional[GenerationParams] = None,
    ) -> List[GenerationResult]:
        """Generate text from multiple prompts."""
        if not prompts:
            return []

        params = params or GenerationParams()

        if self.settings.vllm_mode == "local":
            return self._generate_batch_local(prompts, params)
        else:
            return self._generate_batch_remote(prompts, params)

    def _generate_batch_local(
        self,
        prompts: List[str],
        params: GenerationParams,
    ) -> List[GenerationResult]:
        """Generate using local vLLM instance."""
        sampling_params = self._build_sampling_params(params)
        self.logger.info("Processing %d prompt(s) with local vLLM", len(prompts))

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        results: List[GenerationResult] = []
        for output in outputs:
            if not output.outputs:
                results.append(GenerationResult(text="", finish_reason="error"))
                continue

            generated = output.outputs[0]
            results.append(
                GenerationResult(
                    text=generated.text,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(generated.token_ids),
                    finish_reason=generated.finish_reason or "stop",
                )
            )
        return results

    def _generate_batch_remote(
        self,
        prompts: List[str],
        params: GenerationParams,
    ) -> List[GenerationResult]:
        """Generate using remote vLLM server via OpenAI API."""
        self.logger.info("Processing %d prompt(s) with remote vLLM", len(prompts))

        merged = params.merge_with_defaults(self.settings)
        results: List[GenerationResult] = []

        for prompt in prompts:
            try:
                response = self.client.completions.create(
                    model=str(self.settings.model_path),
                    prompt=prompt,
                    temperature=merged.temperature,
                    top_p=merged.top_p,
                    max_tokens=merged.max_tokens,
                    stop=merged.stop,
                    presence_penalty=merged.presence_penalty,
                    frequency_penalty=merged.frequency_penalty,
                    seed=merged.seed,
                )

                choice = response.choices[0]
                results.append(
                    GenerationResult(
                        text=choice.text,
                        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                        completion_tokens=response.usage.completion_tokens if response.usage else 0,
                        finish_reason=choice.finish_reason or "stop",
                    )
                )
            except Exception as exc:
                self.logger.error("Failed to process prompt: %s", exc)
                results.append(GenerationResult(text="", finish_reason="error"))

        return results

    def chat(
        self,
        messages: List[dict],
        params: Optional[GenerationParams] = None,
    ) -> GenerationResult:
        """Generate a chat completion from a list of messages.

        Messages should be in the format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        results = self.chat_batch([messages], params)
        return results[0]

    def chat_batch(
        self,
        conversations: List[List[dict]],
        params: Optional[GenerationParams] = None,
    ) -> List[GenerationResult]:
        """Generate chat completions for multiple conversations."""
        if not conversations:
            return []

        params = params or GenerationParams()

        if self.settings.vllm_mode == "local":
            return self._chat_batch_local(conversations, params)
        else:
            return self._chat_batch_remote(conversations, params)

    def _chat_batch_local(
        self,
        conversations: List[List[dict]],
        params: GenerationParams,
    ) -> List[GenerationResult]:
        """Process chat via local vLLM by applying chat template."""
        # Apply chat template to convert messages to prompts
        prompts = []
        for messages in conversations:
            if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Fallback: simple concatenation
                prompt = self._simple_chat_format(messages)
            prompts.append(prompt)

        return self._generate_batch_local(prompts, params)

    def _chat_batch_remote(
        self,
        conversations: List[List[dict]],
        params: GenerationParams,
    ) -> List[GenerationResult]:
        """Process chat via remote vLLM using OpenAI chat API."""
        self.logger.info("Processing %d conversation(s) with remote vLLM", len(conversations))

        merged = params.merge_with_defaults(self.settings)
        results: List[GenerationResult] = []

        for messages in conversations:
            try:
                response = self.client.chat.completions.create(
                    model=str(self.settings.model_path),
                    messages=messages,
                    temperature=merged.temperature,
                    top_p=merged.top_p,
                    max_tokens=merged.max_tokens,
                    stop=merged.stop,
                    presence_penalty=merged.presence_penalty,
                    frequency_penalty=merged.frequency_penalty,
                    seed=merged.seed,
                )

                choice = response.choices[0]
                results.append(
                    GenerationResult(
                        text=choice.message.content or "",
                        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                        completion_tokens=response.usage.completion_tokens if response.usage else 0,
                        finish_reason=choice.finish_reason or "stop",
                    )
                )
            except Exception as exc:
                self.logger.error("Failed to process conversation: %s", exc)
                results.append(GenerationResult(text="", finish_reason="error"))

        return results

    @staticmethod
    def _simple_chat_format(messages: List[dict]) -> str:
        """Simple fallback format for chat messages."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)
