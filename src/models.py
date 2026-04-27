from typing import Dict

from pydantic import BaseModel

ParameterMap = Dict[str, "Parameters"]
ReturnTypeMap = Dict[str, str]


class Parameters(BaseModel):
    """Describes one function parameter in the input schema."""

    type: str


class Functions(BaseModel):
    """Schema for a callable function exposed to the model."""

    name: str
    description: str
    parameters: ParameterMap
    returns: ReturnTypeMap


class Prompts(BaseModel):
    """Single natural-language prompt to transform into a function call."""

    prompt: str
