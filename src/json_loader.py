import json
from pydantic import ValidationError
from typing import List

from src.models.functions_definition import FunctionDef
from src.models.prompts import Prompt


def load_function_definition(path: str) -> List[FunctionDef]:
    """
    Load function definitions from a JSON file.

    Args:
        path: Path to the JSON setup file containing function definitions.

    Returns:
        A list of function definition objects.
    """
    try:
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return [FunctionDef(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid json format")
    except ValidationError as e:
        raise RuntimeError(f"Invalid function definition data: {e}")


def load_prompts(path: str) -> List[Prompt]:
    """
    Load natural language prompts from a JSON file.

    Args:
        path: Path to the JSON file containing prompts.

    Returns:
        A list of prompt objects.
    """
    try:
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return [Prompt(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid json format")
    except ValidationError as e:
        raise RuntimeError(f"Invalid prompt data: {e}")
