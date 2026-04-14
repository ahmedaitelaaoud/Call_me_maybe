import sys
from typing import Dict
from src.io.file_handler import load_json


class Vocabulary:
    def __init__(self, vocab_path: str):
        """
        Initializes the Vocabulary object by loading the mapping from disk.
        """
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int]  = {}
        self._load_vocabulary(vocab_path)

    def _load_vocabulary(self, path: str) -> None:
        """
        Loads the vocabulary JSON and builds two-way lookup tables.
        The expected JSON format is usually {"token_string": token_id, ...}
        or vice versa depending on the tokenizer.
        """
        raw_vocab = load_json(path)
        for token_str, token_id in raw_vocab.items():
            t_id = int(token_id)
            t_str = str(token_str)

            self.token_to_id[t_str] = t_id
            self.id_to_token[t_id] = t_str

    def get_string(self, token_id: int) -> str:
        """
        Given a token ID, returns its exact string representation.
        """
        return self.id_to_token.get(token_id, "")

    def get_all_valid_ids(self) -> list[int]:
        """
        Returns a list of all available token IDs.
        Useful for iterating over the entire vocabulary during filtering.
        """
        return list(self.id_to_token.keys())
