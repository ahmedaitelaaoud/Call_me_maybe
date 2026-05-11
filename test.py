import json
from pydantic import BaseModel
from typing import Dict
class Parameter(BaseModel):
    type: str


class ReturnType:
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType
def load_function_definition(path: str):

    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return [FunctionDef(**item) for item in data]


load_function_definition("data/input/functions_definition.json")
