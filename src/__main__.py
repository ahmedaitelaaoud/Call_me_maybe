import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from llm_sdk import Small_LLM_Model

from src.constrained_decoding import (
    build_json_valid_ids,
    build_schema_valid_ids,
    build_system_prompt,
    extract_complete_json,
    get_best_valid_token,
    load_vocabulary,
    coerce_parameters,
    precompute_name_valid_ids,
)
from src.json_loader import load_function_definition, load_prompts
from src.models.functions_definition import FunctionDef


DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns:
        The parsed command-line arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language into function calls."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=DEFAULT_FUNCTIONS_DEFINITION,
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    return parser.parse_args()


def load_model(model_name: str) -> Small_LLM_Model:
    """
    Load the small LLM model explicitly by valid model path.

    Args:
        model_name: The name or path of the model.

    Returns:
        The initialized model instance.

    Raises:
        RuntimeError: if the model is not found.
    """
    try:
        return Small_LLM_Model(model_name=model_name)
    except OSError:
        raise RuntimeError(f"Model '{model_name}' not found.")


def generate_function_call(
    model: Small_LLM_Model,
    base_input_ids: List[int],
    base_valid_ids: np.ndarray,
    name_cache: Dict[str, np.ndarray],
    max_steps: int = 50,
) -> Dict[str, Any]:
    """
    Generate a function call step-by-step using constrained decoding.

    Args:
        model: The language model.
        base_input_ids: The input IDs of the prompt.
        base_valid_ids: The initial valid IDs for JSON.
        name_cache: The precomputed valid IDs for function names.
        max_steps: The maximum number of generation steps.

    Returns:
        A dictionary containing the structured function call.
    """
    newly_generated_ids: List[int] = list(
        model.encode('{"name": "')[0].tolist()
    )
    decoded: str = model.decode(newly_generated_ids)

    for _ in range(max_steps):
        valid_ids = build_schema_valid_ids(base_valid_ids, name_cache, decoded)

        logits = model.get_logits_from_input_ids(
            base_input_ids + newly_generated_ids
        )
        next_id = get_best_valid_token(logits, valid_ids)
        newly_generated_ids.append(next_id)

        decoded = model.decode(newly_generated_ids)

        extracted = extract_complete_json(decoded)
        if extracted:
            try:
                response = json.loads(extracted)
                if "args" in response:
                    response["parameters"] = response.pop("args")
                return response
            except (json.JSONDecodeError, ValueError):
                continue

    return {"name": "none", "parameters": {}}


def save_results(path: str, results: List[Dict[str, Any]]) -> None:
    """
    Save the function calling results to a JSON file.

    Args:
        path: The path where the results should be saved.
        results: A list of dictionaries representing function calls.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main() -> None:
    """
    The main pipeline function that loads the model,
    builds valid token configurations, and iterates through tests.
    """

    print("🚀 Starting function calling pipeline...")
    args = parse_args()

    print("📂 Loading functions and prompts...")
    functions: List[FunctionDef] = load_function_definition(
        args.functions_definition
    )
    if not functions:
        raise RuntimeError(
            "No function definitions found. Please provide at least one."
        )
    prompts = load_prompts(args.input)
    if not prompts:
        raise RuntimeError(
            "No prompts found. Please provide at least one."
        )

    print("🧠 Building a system prompt...")
    system = build_system_prompt(functions)

    print(f"🤖 Loading model: {args.model}")
    model = load_model(args.model)

    print("⚙️  Building valid token IDs...")
    vocab = load_vocabulary(model)
    base_valid_ids = build_json_valid_ids(vocab)
    fn_names = {fn.name for fn in functions}
    name_cache = precompute_name_valid_ids(vocab, fn_names)

    fn_lookup: Dict[str, FunctionDef] = {fn.name: fn for fn in functions}

    all_results: List[Dict[str, Any]] = []
    start_time = time.time()

    print("⏱️  Processing prompts...")
    for test_case in prompts:
        prompt = test_case.prompt
        full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
        base_input_ids: List[int] = model.encode(full_prompt)[0].tolist()

        response = generate_function_call(
            model=model,
            base_input_ids=base_input_ids,
            base_valid_ids=base_valid_ids,
            name_cache=name_cache,
        )

        fn_name: str = response.get("name", "none")
        raw_params: Any = response.get("parameters", {})
        fn_def: Optional[FunctionDef] = fn_lookup.get(fn_name)
        parameters = (
            coerce_parameters(raw_params, fn_def)
            if fn_def is not None
            else raw_params
        )
        if fn_name != "none":
            print(
                f"  ✅ Generated call -> {response['name']}"
                f"({response.get('parameters', {})})"
            )
        else:
            print("  ❌ [ERROR] Could not generate function call.")

        all_results.append({
                "prompt": prompt,
                "name": fn_name,
                "parameters": parameters,
            })
    total_time = time.time() - start_time

    save_results(args.output, all_results)

    print(f"\n💾 Saved results to: {args.output}")
    print(f"🎉 Completed in {total_time:.2f} seconds")
    print(
        f"⚡ Average time per prompt: "
        f"{total_time/len(prompts):.2f} seconds"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
