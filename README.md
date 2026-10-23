# Call Me Maybe

An educational Python pipeline that explores local LLM function calling through token filtering and function-name prefix constraints.

It translates prompts into JSON records describing a proposed function call. **It does not execute functions, and it is not a complete JSON Schema decoder.**

## How it works

1. Load function definitions and prompts through Pydantic models.
2. Build a prompt describing the available functions.
3. Load a local causal language model through the included `llm_sdk`.
4. Start generation with `{"name": "`, then select the highest-logit token from a filtered vocabulary.
5. Apply additional prefix filtering while generating the function name.
6. Extract and parse a complete JSON object, rename `args` to `parameters`, and attempt type conversion for recognized parameters.
7. Save the prompt, selected name, and parameters as JSON.

The default model is `Qwen/Qwen3-0.6B`. The SDK uses PyTorch and Hugging Face Transformers, selecting MPS, CUDA, or CPU according to availability.

## Run locally

Requirements: Python 3.10 or later, [uv](https://docs.astral.sh/uv/), and access to download the model and tokenizer on the first run. Model inference requires more memory and setup than the decoder utilities alone.

```bash
git clone https://github.com/ahmedaitelaaoud/Call_me_maybe.git
cd Call_me_maybe
uv sync
uv run python -m src
```

The local SDK is an editable path dependency configured in `pyproject.toml`.

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --model Qwen/Qwen3-0.6B
```

Use an output path containing a directory, as in the example; the current writer expects one.

Input prompts are JSON objects with a `prompt` field. The included fixtures cover arithmetic, greetings, string reversal, square roots, and regex substitution. Output records contain `prompt`, `name`, and `parameters`; model predictions are not reference answers.

## Code map

| File | Responsibility |
| --- | --- |
| `src/__main__.py` | CLI, generation loop, and output writing |
| `src/constrained_decoding.py` | Token masks, name-prefix cache, JSON extraction, and coercion |
| `src/json_loader.py` | Input loading |
| `src/models/` | Input models |
| `llm_sdk/` | Local model and tokenizer wrapper |
| `data/input/` | Function definitions and example prompts |

## Engineering decisions and limits

Filtering logits makes the selection mechanism inspectable. The name-prefix cache avoids rebuilding its candidate sets on every step. However, filtering characters is weaker than tracking a JSON grammar.

- The decoder does **not** enforce all parameter names, required fields, or value types during generation.
- A closing quote is permitted at every cached name prefix, so function-name filtering is not a strict membership guarantee.
- Type conversion happens after parsing; failed conversions retain the original value. It is not schema validation.
- Generation is capped at 50 steps. Failure to parse a complete object returns `{"name": "none", "parameters": {}}`.
- The SDK runs a forward pass over the growing sequence for each token; the loop does not reuse a KV cache.
- Input fixtures are examples, not an automated accuracy evaluation.
- The SDK enables `trust_remote_code`; use model sources you trust.

## Development

```bash
uv run make lint
uv run make lint-strict
```

These run lint and type checks, not model-quality tests. The next useful steps are a deterministic decoder test suite, explicit output-schema validation, and an evaluation set separating syntax validity from correct function selection.

## License

No repository-level license is currently included.

