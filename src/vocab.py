import numpy as np
import numpy.typing as npt
from llm_sdk import Small_LLM_Model

from src.models import Functions

from typing import Any, Dict, List, Set


class VocabManager:
    """Precomputes token masks used for constrained decoding."""

    def __init__(self, model: Small_LLM_Model, functions: List[Functions]) -> None:
        self.model = model
        self.functions = functions

        # Build an id -> token map once so mask construction stays simple.
        sample_input_ids = model.encode("hello").tolist()[0]
        sample_logits: List[float] = model.get_logits_from_input_ids(sample_input_ids)
        self.id_to_token: Dict[int, str] = {
            token_id: model.decode([token_id])
            for token_id, _ in enumerate(sample_logits)
        }

        self.vocab_size: int = len(sample_logits)
        # Start with everything blocked (-inf), then unlock valid tokens with 0.0.
        self.number_mask: npt.NDArray[np.float32] = self._new_mask()
        self.string_mask: npt.NDArray[np.float32] = self._new_mask()
        self.bool_mask: npt.NDArray[np.float32] = self._new_mask()
        self.function_name_mask: npt.NDArray[np.float32] = self._new_mask()

        # Allow token ids that appear in known function names.
        function_name_token_ids: Set[int] = set()
        encoded_function_names: List[Any] = [
            self.model.encode(function_item.name)
            for function_item in self.functions
        ]
        for encoded_name in encoded_function_names:
            function_name_token_ids.update(encoded_name[0].tolist())
        self.function_name_mask[list(function_name_token_ids)] = 0.0

        for token_id, token_text in self.id_to_token.items():
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
                if sum(1 for char in compact_token if char in ",.-") > 1:
                    continue
                self.number_mask[token_id] = 0.0

        # Backward-compatible aliases used in existing call sites.
        self.M_numbers = self.number_mask
        self.M_chars = self.string_mask
        self.M_fun_name = self.function_name_mask

    def _new_mask(self) -> npt.NDArray[np.float32]:
        return np.full(self.vocab_size, -np.inf, dtype=np.float32)
