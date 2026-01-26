# vllm_worker

Lightweight Python service for vLLM with a persistent worker and automatic VRAM management.

## Environment variables

```bash
export VLLM_MODEL_PATH=/path/to/model    # Required
export VLLM_GPU_MEMORY_UTILIZATION=0.9   # Optional (default: 0.9)
export VLLM_MAX_MODEL_LEN=32768          # Optional (default: 32768)
export VLLM_IDLE_TTL_SECONDS=1800        # Optional (default: 1800)
export VLLM_DEFAULT_TEMPERATURE=0.7      # Optional (default: 0.7)
export VLLM_DEFAULT_MAX_TOKENS=2048      # Optional (default: 2048)

# Remote mode (optional)
export VLLM_MODE=remote
export VLLM_API_URL=http://localhost:8000/v1
```

## Usage

```python
from vllm_worker import VLLMInferenceService

service = VLLMInferenceService()

# Simple generation
result = service.generate("What is Python?", temperature=0.7, max_tokens=500)
print(result.text)

# Chat
result = service.chat([
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "Hello!"},
])

# Batch
results = service.generate_batch(["Q1?", "Q2?"], temperature=0.3)

# Explicit warmup
service.warmup()

# Context manager
with VLLMInferenceService() as service:
    result = service.generate("Hello!")
```

## Structure

```
vllm_worker/
├── __init__.py      # Public exports
├── config.py        # VLLMSettings, GenerationParams
├── inference.py     # VLLMInferenceEngine (local/remote)
├── worker_pool.py   # VLLMWorkerPool (subprocess management)
└── service.py       # VLLMInferenceService (main API)
```

**Flow:** `VLLMInferenceService` → `VLLMWorkerPool` → subprocess with `VLLMInferenceEngine`

The worker is spawned on the first call, stays active for subsequent requests, and automatically stops after `VLLM_IDLE_TTL_SECONDS` of inactivity (VRAM is released).

## API

### Main methods

| Method                                | Description                          |
| ------------------------------------- | ------------------------------------ |
| `generate(prompt, **params)`          | Simple text generation               |
| `generate_batch(prompts, **params)`   | Batch generation                     |
| `chat(messages, **params)`            | Chat completion                      |
| `chat_batch(conversations, **params)` | Batch chat                           |
| `warmup()`                            | Preloads the model                   |
| `shutdown()`                          | Stops the worker                     |
| `is_ready()`                          | Checks whether the worker is running |

### Generation parameters

`temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty`, `stop`, `presence_penalty`, `frequency_penalty`, `seed`

### GenerationResult

```python
result.text              # Generated text
result.prompt_tokens     # Prompt tokens
result.completion_tokens # Generated tokens
result.finish_reason     # "stop", "length", "error"
```
