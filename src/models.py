from pydantic import BaseModel
from typing import Dict


class Parameters(BaseModel):
    type: str


class Functions(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameters]
    returns: Dict[str, str]


class Prompts(BaseModel):
    prompt: str
