from src.models.functions_definition import FunctionDef
from src.models.prompts import Prompt
import json


def load_function_definition(path: str):
    try:
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return [FunctionDef(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid json format")


def load_prompts(path: str):
    try:
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return [Prompt(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid json format")
