Call Me Maybe — Function Calling with Constrained Decoding

A minimal, reliable pipeline that turns natural language prompts into **valid, schema‑compliant JSON function calls**. It uses constrained decoding to keep the output parseable and aligned with your function definitions.

## Highlights
- **Strict JSON output** with a fixed schema.
- **Token‑level constraints** prevent invalid keys, types, or structure.
- **Deterministic schema compliance** without post‑hoc repair.
- **Simple CLI** for quick experiments and grading.

## How it works
1. Load function definitions and prompts from data/input/.
2. Build a system prompt listing available functions.
3. Generate the function name using token‑level constrained decoding.
4. Enforce a strict JSON schema while generating arguments:
   - Fixed JSON structure and key order
   - Parameter names restricted to the provided schema
   - Value types constrained by JSON type (string, number, integer, boolean)
5. Coerce parameters to schema types and write results to data/output/.

## Constrained decoding strategy
The decoder enforces the template:

{"name": "<fn>", "args": {<params>}}

It uses:
- A prefix cache to restrict function name tokens to valid functions
- Literal‑only emission for fixed JSON segments and parameter keys
- Type‑aware value generation with token filters that allow only valid prefixes

This guarantees **parseable JSON** and prevents schema drift or extra keys.

## Project layout
- data/input/ — function definitions and test prompts
- data/output/ — generated function calls
- src/ — constrained decoding pipeline
- src/models/ — Pydantic models for schemas and prompts

## Quickstart
Run the pipeline:

uv run python -m src

Optional arguments:
--functions_definition data/input/functions_definition.json
--input data/input/function_calling_tests.json
--output data/output/function_calling_results.json

## Development
Common Makefile targets:
- install: uv sync
- run: uv run python -m src
- debug: uv run python -m pdb -m src
- clean: remove caches
- lint: flake8 + mypy with required flags
- lint-strict: flake8 + mypy --strict

## Notes
- The function name is selected by the LLM (no heuristics).
- Schema validity is enforced during generation, not repaired afterward.
- Errors on invalid inputs are surfaced with clear messages.
