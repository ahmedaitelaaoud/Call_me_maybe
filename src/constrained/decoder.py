from llm_sdk import Small_LLM_Model
from src.constrained.state_machine import StateMachine, JsonStates
from src.constrained.token_filter import TokenFilter
from src.llm.vocabulary import Vocabulary
import numpy as np

class Decoder:
    def __init__(self, model: Small_LLM_Model ,state_machine: StateMachine, vocab: Vocabulary, token_filter: TokenFilter):
        self.model = model
        self.state_machine = state_machine
        self.vocab = vocab
        self.token_filter = token_filter

    def generate(self, initial_prompt: str) -> str:
        generated_json = ""
        current_text = initial_prompt
        while self.state_machine.current_state != JsonStates.DONE:
