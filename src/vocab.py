import numpy as np
from llm_sdk import Small_LLM_Model
from src.models import Functions
from typing import Dict, List, Set, Any
from numpy.typing import npt


class VocabManager:
    def __init__(self, model: Small_LLM_Model, functions: List[Functions]) \
            -> None:
        self.model: Small_LLM_Model = model
        self.functions: List[Functions] = functions

        # Build an id->token map once so later checks stay simple.
        sample_input_ids = model.encode("hello").tolist()[0]
        sample_logits: List[float] = model.get_logits_from_input_ids(
            sample_input_ids
        )
        self.id_to_token: Dict[int, str] = {}
        for token_id, _ in enumerate(sample_logits):
            self.id_to_token = model.decode([token_id])

        self.vocab_size: int = len(sample_logits)
        # Start with everything blocked (-inf), then open valid tokens
        # with 0.0.
        self.number_mask: npt.NDArray[np.float32] = np.full(
            self.vocab_size, -np.inf, dtype=np.float32
        )
        self.string_mask: npt.NDArray[np.float32] = np.full(
            self.vocab_size, -np.inf, dtype=np.float32
        )
        self.bool_mask: npt.NDArray[np.float32] = np.full(
            self.vocab_size, -np.inf, dtype=np.float32
        )
        self.function_name_mask: npt.NDArray[np.float32] = np.full(
            self.vocab_size, -np.inf, dtype=np.float32
        )

        # Allow token ids that appear in known function names
        function_name_token_ids: Set[int] = set()
        encoded_function_names: List[Any] = [
            self.model.encode(function_item.name)
            for function_item in self.functions
        ]
        # apply the mathematical "whitelist"
        for encoded_name in encoded_function_names:
            function_name_token_ids.update(encoded_name[0].tolist())
        self.function_name_mask[list(function_name_token_ids)] = 0.0

        for token_id, token_text in self.id_to_token.items:
            # GPT-style markers: Ġ is space, Ċ is newline.
            token_with_whitespace: str = token_text.replace(
                "Ġ", " "
            ).replace("Ċ", "\n")
            compact_token: str = token_text.replace("Ġ", "").replace("Ċ", "")
            if not token_with_whitespace:
                continue

            if "\n" not in token_with_whitespace:
                self.string_mask[token_id] = 0.0

            if not compact_token:
                continue

            if compact_token == '"':
                self.function_name_mask[token_id] = 0.0

            # Accept tokens that look like a number
            if all(char in "0123456789.-," for char in compact_token):
                if sum([1 for char in compact_token if char in ",.-"]) > 1:
                    continue
                self.number_mask[token_id] = 0.0
