# vLLM Inference Service

A lightweight Python service to run vLLM inference with persistent worker management,
automatic idle shutdown, and support for both local and remote modes.

## Features

- **Persistent worker process**: Model loaded once, reused for multiple requests
- **Automatic idle shutdown**: Worker shuts down after configurable idle period
- **Local and remote modes**: Use local vLLM or connect to existing vLLM server
- **Batch processing**: Efficient batch generation for multiple prompts
- **Chat support**: OpenAI-compatible chat completion interface
- **Thread-safe**: Safe to use from multiple threads

## Environment Variables

Set the following environment variables before creating the service:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VLLM_MODEL_PATH` | Yes | - | Path to the model directory |
| `VLLM_GPU_MEMORY_UTILIZATION` | No | 0.9 | GPU memory fraction to use |
| `VLLM_MAX_MODEL_LEN` | No | 32768 | Maximum sequence length |
| `VLLM_TENSOR_PARALLEL_SIZE` | No | 1 | Number of GPUs for tensor parallelism |
| `VLLM_DTYPE` | No | "auto" | Model dtype (auto, float16, bfloat16) |
| `VLLM_IDLE_TTL_SECONDS` | No | 1800 | Worker idle timeout in seconds |
| `VLLM_TRUST_REMOTE_CODE` | No | true | Trust remote code in model |
| `VLLM_DEFAULT_TEMPERATURE` | No | 0.7 | Default sampling temperature |
| `VLLM_DEFAULT_TOP_P` | No | 0.95 | Default nucleus sampling probability |
| `VLLM_DEFAULT_TOP_K` | No | -1 | Default top-k sampling (-1 = disabled) |
| `VLLM_DEFAULT_MAX_TOKENS` | No | 2048 | Default maximum tokens to generate |
| `VLLM_DEFAULT_REPETITION_PENALTY` | No | 1.0 | Default repetition penalty |
| `VLLM_MODE` | No | "local" | Mode: "local" or "remote" |
| `VLLM_API_URL` | When remote | - | URL of remote vLLM server |

## Example Usage

### Basic Generation

```python
from vllm_service import VLLMInferenceService

service = VLLMInferenceService()

# Simple generation
result = service.generate("What is the capital of France?")
print(result.text)
print(f"Tokens: {result.prompt_tokens} + {result.completion_tokens}")

# With custom parameters
result = service.generate(
    "Write a poem about the ocean",
    temperature=0.9,
    max_tokens=500,
    top_p=0.95,
)
print(result.text)
```

### Batch Generation

```python
prompts = [
    "What is 1+1?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
]

results = service.generate_batch(prompts, temperature=0.3)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

### Chat Completion

```python
# Single conversation
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
]

result = service.chat(messages, temperature=0.7)
print(result.text)

# Multi-turn conversation
messages.append({"role": "assistant", "content": result.text})
messages.append({"role": "user", "content": "How about writing to a file?"})

result = service.chat(messages)
print(result.text)
```

### Batch Chat

```python
conversations = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "user", "content": "What's 2+2?"}],
    [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Tell me about the sea."},
    ],
]

results = service.chat_batch(conversations)
for conv, result in zip(conversations, results):
    print(f"User: {conv[-1]['content']}")
    print(f"Assistant: {result.text}\n")
```

### Using GenerationParams

```python
from vllm_service import VLLMInferenceService, GenerationParams

service = VLLMInferenceService()

# Create reusable parameter object
creative_params = GenerationParams(
    temperature=0.9,
    top_p=0.95,
    max_tokens=1000,
    repetition_penalty=1.1,
)

deterministic_params = GenerationParams(
    temperature=0.0,
    max_tokens=100,
)

# Use with batch methods
results = service.generate_batch(
    ["Write a story", "Write another story"],
    params=creative_params,
)
```

### Warmup and Lifecycle

```python
# Pre-load model before first request
service = VLLMInferenceService()
service.warmup()  # Model loaded here

# Check if worker is running
if service.is_ready():
    print("Worker is ready!")

# Explicit shutdown
service.shutdown()

# Or use context manager (recommended)
with VLLMInferenceService() as service:
    result = service.generate("Hello!")
    print(result.text)
# Worker automatically shut down
```

### Custom Settings

```python
from vllm_service import VLLMInferenceService, VLLMSettings
from pathlib import Path

# Create settings programmatically
settings = VLLMSettings(
    model_path=Path("/models/Qwen2.5-7B-Instruct"),
    gpu_memory_utilization=0.8,
    max_model_len=16384,
    idle_ttl_seconds=600,  # 10 minutes
    default_temperature=0.5,
)

service = VLLMInferenceService(settings=settings)
```

### Using Remote vLLM Server

```bash
# Start vLLM server separately
python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen2.5-7B-Instruct \
    --port 8000

# Set environment variables
export VLLM_MODE=remote
export VLLM_API_URL=http://localhost:8000/v1
export VLLM_MODEL_PATH=/models/Qwen2.5-7B-Instruct
```

```python
# Service will connect to remote server
service = VLLMInferenceService()
result = service.generate("Hello!")
```

## Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | Sampling temperature (0 = deterministic) |
| `top_p` | float | 0.95 | Nucleus sampling probability |
| `top_k` | int | -1 | Top-k sampling (-1 = disabled) |
| `max_tokens` | int | 2048 | Maximum tokens to generate |
| `repetition_penalty` | float | 1.0 | Penalty for repeating tokens |
| `stop` | list[str] | None | Stop sequences |
| `presence_penalty` | float | None | Penalty for token presence |
| `frequency_penalty` | float | None | Penalty for token frequency |
| `seed` | int | None | Random seed for reproducibility |

## GenerationResult

The result object contains:

```python
@dataclass
class GenerationResult:
    text: str              # Generated text
    prompt_tokens: int     # Number of prompt tokens
    completion_tokens: int # Number of generated tokens
    finish_reason: str     # "stop", "length", or "error"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
```

## Architecture

```
VLLMInferenceService
    │
    ├── VLLMSettings (configuration)
    │
    └── VLLMWorkerPool
            │
            └── Worker Process (spawned)
                    │
                    └── VLLMInferenceEngine
                            │
                            ├── Local: vLLM LLM instance
                            └── Remote: OpenAI client
```

The worker process is spawned using `multiprocessing` with the "spawn" context
to ensure clean GPU memory management. The worker automatically shuts down
after being idle for the configured TTL.

## Thread Safety

The service is thread-safe. Multiple threads can call `generate()`, `chat()`,
etc. concurrently. Requests are serialized through the worker process.

## Error Handling

```python
try:
    result = service.generate("Hello")
    if result.finish_reason == "error":
        print("Generation failed")
    elif result.finish_reason == "length":
        print("Generation stopped due to max_tokens limit")
except RuntimeError as e:
    print(f"Service error: {e}")
```