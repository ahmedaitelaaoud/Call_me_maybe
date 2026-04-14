from pydantic import BaseModel
from typing import Dict, Any


class ParameterDef(BaseModel):
    """
    Represents the type definition of a single parameter or return value.
    Example: {"type": "number"}
    """
    type: str


class FunctionDef(BaseModel):
    """
    Represents a single function definition.
    Validates the structure provided in data/input/function_definitions.json.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class FunctionCallOutput(BaseModel):
    """
    Represents a single generated object for our final output file.
    Validates that the LLM successfully generated the required structure.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]
