import json
from typing import Any, Dict, List, Optional, Set

import numpy as np
from llm_sdk import Small_LLM_Model

from src.models.functions_definition import FunctionDef


_JSON_SAFE: frozenset = frozenset(
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789*.,_:-+/\'!?()[]{}"ĠĊ\\'
)


TYPE_MAP: Dict[str, type] = {
    "number": float,
    "string": str,
    "boolean": bool,
    "integer": int,
}


def coerce_parameters(
    parameters: Any,
    fn_def: FunctionDef,
) -> Dict[str, Any]:
    """
    Coerce the parameters to match the function definition.

    Args:
        parameters: The input mapped parameters.
        fn_def: The definition of the function specifying types.

    Returns:
        A dictionary with properly typed parameters.
    """

    if not isinstance(parameters, dict):
        return {}
    coerced: Dict[str, Any] = {}
    for key, value in parameters.items():
        param_def = fn_def.parameters.get(key)
        if param_def is not None:
            target = TYPE_MAP.get(param_def.type)
            try:
                coerced[key] = target(value) if target is not None else value
            except (ValueError, TypeError):
                coerced[key] = value
        else:
            coerced[key] = value
    return coerced


def extract_complete_json(text: str) -> Optional[str]:
    """
    Extract the first complete JSON object from text.

    Args:
        text: The input text containing a potential JSON object.

    Returns:
        The valid JSON string if found, otherwise None.
    """

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return None


def get_best_valid_token(logits: Any, valid_ids: np.ndarray) -> int:
    """
    Get the best valid token based on its logits.

    Args:
        logits: The logits for the vocabulary.
        valid_ids: A numpy array of valid token IDs.

    Returns:
        The token ID with the highest logit out of the valid set.
    """
    logits_np: np.ndarray = np.asarray(logits)
    valid_logits: np.ndarray = logits_np[valid_ids]
    return int(valid_ids[np.argmax(valid_logits)])


def build_schema_valid_ids(
    base_valid_ids: np.ndarray,
    name_cache: Dict[str, np.ndarray],
    decoded_so_far: str,
) -> np.ndarray:
    """
    Filter valid IDs dynamically as the generation progresses.

    Args:
        base_valid_ids: The initial base valid IDs.
        name_cache: A dictionary with precomputed valid token sets by prefix.
        decoded_so_far: The string reconstructed up to this point.

    Returns:
        The refined valid token array.
    """

    name_prefix = '"name": "'
    idx = decoded_so_far.find(name_prefix)
    if idx == -1:
        return base_valid_ids

    after_prefix = decoded_so_far[idx + len(name_prefix):]
    if '"' in after_prefix:
        return base_valid_ids

    return name_cache.get(after_prefix, base_valid_ids)


def precompute_name_valid_ids(
        vocab: Dict[str, int],
        fn_names: Set[str]
) -> Dict[str, np.ndarray]:
    """
    Precompute valid next tokens for function names based on their prefixes.

    Args:
        vocab: The vocabulary mapping string tokens to IDs.
        fn_names: The set of available function names.

    Returns:
        A mapping of current prefix to a numpy array of valid next token IDs.
    """

    all_prefixes: Set[str] = {""}
    for name in fn_names:
        for i in range(1, len(name) + 1):
            all_prefixes.add(name[:i])

    cache: Dict[str, np.ndarray] = {}
    for prefix in all_prefixes:
        valid: Set[int] = set()
        for token_str, token_id in vocab.items():
            if token_str == '"':
                valid.add(token_id)
                continue
            candidate = prefix + token_str
            if any(
                name.startswith(candidate) or candidate.startswith(name)
                for name in fn_names
            ):
                valid.add(token_id)
        cache[prefix] = np.array(sorted(valid), dtype=np.int64)

    return cache


def build_json_valid_ids(vocab: Dict[str, int]) -> np.ndarray:
    """
    Build an array of token IDs that contain only JSON-safe characters.

    Args:
        vocab: The vocabulary dictionary.

    Returns:
        A numpy array with valid token IDs.
    """
    valid: Set[int] = {
        token_id
        for token_str, token_id in vocab.items()
        if token_str and _JSON_SAFE.issuperset(token_str)
    }
    return np.array(sorted(valid), dtype=np.int64)


def load_vocabulary(model: Small_LLM_Model) -> Dict[str, int]:
    """
    Load the vocabulary dictionary from the model tokenizer.

    Args:
        model: The small LLM model instance.

    Returns:
        A dictionary mapping strings to token IDs.
    """
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, 'r', encoding="utf-8") as f:
        tok_data = json.load(f)
    raw_vocab: Dict[str, int] = tok_data.get("model", {}).get("vocab", {})
    return raw_vocab


def build_system_prompt(functions: List[FunctionDef]) -> str:
    """
    Build the system prompt detailing available functions constraints.

    Args:
        functions: The list of function definitions.

    Returns:
        The generated system prompt text formatting instructions.
    """
    lines = [
        "STRICT SYSTEM RULES: use ONLY a matching function "
        "from the list below",
        "If No function matches the user's intent (even if "
        "types match), set name:\"none\".",
        "Never use an unrelated function for a differnet task.",
        "",
        "Available functions:",
    ]
    for fn in functions:
        params = ", ".join(
            f"{name}: {info.type}"
            for name, info in fn.parameters.items()
        )
        lines.append(f"  -{fn.name}({params}): {fn.description}")
    lines.append(
        '\nOutput ONLY valid JSON: '
        '{"name": "<fn>", "args": {<args>}}'
    )
    return "\n".join(lines)
