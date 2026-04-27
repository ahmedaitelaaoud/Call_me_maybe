import json
from argparse import ArgumentParser, Namespace
from json import JSONDecodeError
from pathlib import Path
from sys import stderr
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from llm_sdk import Small_LLM_Model
from numpy import intp
from pydantic import ValidationError

from src.models import Functions, Prompts
from src.vocab import VocabManager

DEFAULT_FUNCTIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calls.json"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"

PARAMS_PROMPT_TEMPLATE = """Task: Complete the JSON function call.

Function: {function_name}

RULES:
1. You must write the SHORTEST possible pattern.
2. For numbers, you must output EXACTLY [0-9]+ and immediately stop.
3. For vowels, you must output EXACTLY aeiouAEIOU and immediately stop.
4. DO NOT repeat patterns. ALWAYS close the string with a double quote (")!
5. ALWAYS escape double quotes in parameters with (\\)!
6. when generating a path, ALWAYS generate the full path not just the file name
8. NEVER include 'database' for database parameter only the name of database

--- EXAMPLES ---
Input: "Replace all vowels in 'this is a test' with asterisks"
JSON: {{"name": "fn_substitute_string_with_regex", "parameters": \
    {{"source_string": "this is a test", \
        "regex": "[aeiouAEIOU]", "replacement": "*"}}}}

Input: "Replace all numbers in 'Phone 555-1234' with NUMBERS"
JSON: {{"name": "fn_substitute_string_with_regex", "parameters": \
    {{"source_string": "Phone 555-1234", "regex": "[0-9]+", \
        "replacement": "NUMBERS"}}}}

Input: "Run the query 'INSERT INTO logs VALUES (1, 2, 3)' \
    on the system database"
JSON: {{"name": "fn_execute_sql_query", "parameters":
    {{"query": "INSERT INTO logs VALUES (1, 2, 3)", "database": "system"}}}}

Input: "Format template: Say "hello" to {{name}}" \
JSON: {{"name": "fn_format_template", "parameters":
    {{"template": "Say \\"hello\\" to {{name}}"}}}}

Input: "{prompt}"
JSON:
{json_head}"""


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS_PATH)
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def load_input_data(args: Namespace) -> Tuple[List[Functions], List[Prompts]]:
    try:
        with open(args.functions_definition, "r", encoding="utf-8") as file:
            function_defs_data: Any = json.load(file)
            functions: List[Functions] = [
                Functions(**function_def) for function_def in function_defs_data
            ]

        with open(args.input, "r", encoding="utf-8") as file:
            prompt_data: Any = json.load(file)
            prompts: List[Prompts] = [Prompts(**prompt_item) for prompt_item in prompt_data]
    except ValidationError:
        print("CMM: invalid input data", file=stderr)
        raise SystemExit(1)
    except JSONDecodeError:
        print("CMM: Invalid JSON format", file=stderr)
        raise SystemExit(1)
    except FileNotFoundError:
        print("CMM: input file not found", file=stderr)
        raise SystemExit(1)
    except PermissionError:
        print("CMM: permission denied", file=stderr)
        raise SystemExit(1)
    except Exception as err:
        print(f"CMM: Unexpected error: {err}", file=stderr)
        raise SystemExit(1)

    return functions, prompts


def pick_token_with_mask(
    model: Small_LLM_Model,
    prompt_text: str,
    mask: np.ndarray,
) -> str:
    input_ids = model.encode(prompt_text)[0].tolist()
    token_id: intp = np.argmax(model.get_logits_from_input_ids(input_ids) + mask)
    return model.decode([int(token_id)])


def choose_function_name(
    model: Small_LLM_Model,
    vocab: VocabManager,
    name_prompt: str,
) -> str:
    selected_name = ""
    while True:
        token = pick_token_with_mask(model, name_prompt, vocab.function_name_mask)
        if '"' in token:
            break
        name_prompt += token
        selected_name += token
        print(token, end="", flush=True)
    return selected_name


def build_parameters_prompt(
    selected_function: Functions,
    prompt_text: str,
    json_head: str,
) -> str:
    return PARAMS_PROMPT_TEMPLATE.format(
        function_name=selected_function.name,
        prompt=prompt_text,
        json_head=json_head,
    )


def find_function_by_name(
    functions: List[Functions],
    selected_name: str,
) -> Optional[Functions]:
    for function_item in functions:
        if function_item.name == selected_name:
            return function_item
    return None


def generate_string_parameter(
    model: Small_LLM_Model,
    vocab: VocabManager,
    params_prompt: str,
) -> Tuple[str, str]:
    params_prompt += '"'
    string_value = ""
    print('"', end="", flush=True)
    while True:
        str_token = pick_token_with_mask(model, params_prompt, vocab.string_mask)
        if '"' in str_token and '\\"' not in str_token:
            if '\\"' in (string_value + str_token):
                continue
            str_token = str_token.split('"', 1)[0] + '"'
            string_value += str_token.split('"', 1)[0]
            params_prompt += str_token
            print(str_token, end="", flush=True)
            break

        params_prompt += str_token
        string_value += str_token
        print(str_token, end="", flush=True)

        if '\\"' in string_value:
            string_value = string_value.replace("\\", "")

    return params_prompt, string_value


def generate_number_parameter(
    model: Small_LLM_Model,
    vocab: VocabManager,
    params_prompt: str,
) -> Tuple[str, str]:
    number_value = ""
    while True:
        num_token = pick_token_with_mask(model, params_prompt, vocab.number_mask)
        if "," in num_token:
            break
        params_prompt += num_token
        number_value += num_token
        print(num_token, end="", flush=True)
    return params_prompt, number_value


def append_parameter_separator(
    params_prompt: str,
    param_index: int,
    param_count: int,
) -> str:
    if param_index < param_count - 1:
        params_prompt += ", "
        print(", ", end="", flush=True)
    return params_prompt


def generate_entry_for_prompt(
    model: Small_LLM_Model,
    vocab: VocabManager,
    functions: List[Functions],
    function_summaries: List[str],
    prompt_item: Prompts,
) -> Dict[str, Any]:
    name_prompt = (
        "choose a function name from the following functions"
        f"\n\n{''.join(function_summaries)}"
        f"\nfor the following prompt \"{prompt_item.prompt}\""
        "\nchosen name: \""
    )
    json_head = f'{{\n    "prompt": "{prompt_item.prompt}",\n    "name": "'
    print(json_head, end="", flush=True)

    selected_name = choose_function_name(model, vocab, name_prompt)
    json_head += selected_name + '",\n    "parameters": {'
    print('",\n    "parameters": {', end="", flush=True)

    selected_function = find_function_by_name(functions, selected_name)
    if selected_function is None:
        raise ValueError("CMM: unknown chosen function")

    entry: Dict[str, Any] = {
        "prompt": prompt_item.prompt,
        "name": selected_name,
        "parameters": {},
    }

    params_prompt = build_parameters_prompt(selected_function, prompt_item.prompt, json_head)
    param_count = len(selected_function.parameters)

    for param_index, (param_name, param_spec) in enumerate(
        selected_function.parameters.items()
    ):
        params_prompt += f'"{param_name}": '
        print(f'"{param_name}": ', end="", flush=True)

        if param_spec.type == "string":
            params_prompt, string_value = generate_string_parameter(
                model, vocab, params_prompt
            )
            entry["parameters"][param_name] = string_value
        elif param_spec.type in {"number", "integer"}:
            params_prompt, number_value = generate_number_parameter(
                model, vocab, params_prompt
            )
            if param_spec.type == "number":
                entry["parameters"][param_name] = float(number_value)
            else:
                entry["parameters"][param_name] = int(number_value)
        else:
            raise ValueError(f"CMM: unsupported parameter type '{param_spec.type}'")

        params_prompt = append_parameter_separator(params_prompt, param_index, param_count)

    print("}\n}")
    return entry


def write_output(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    except Exception as err:
        print(f"CMM: error writing output file: {err}", file=stderr)


def main() -> None:
    args = parse_args()
    functions, prompts = load_input_data(args)

    model = Small_LLM_Model(model_name=args.model)
    vocab = VocabManager(model, functions)

    function_summaries = [
        f"name: {function_item.name} - description: {function_item.description}\n"
        for function_item in functions
    ]

    print("\n====================json generation start=====================")
    results: List[Dict[str, Any]] = []
    for prompt_index, prompt_item in enumerate(prompts, start=1):
        print("\n")
        print(f"- json output for prompt {prompt_index}:\n")
        entry = generate_entry_for_prompt(
            model=model,
            vocab=vocab,
            functions=functions,
            function_summaries=function_summaries,
            prompt_item=prompt_item,
        )
        results.append(entry)

    write_output(Path(args.output), results)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"CMM: an error happened: {err}", file=stderr)
