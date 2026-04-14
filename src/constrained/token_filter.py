from src.constrained.state_machine import StateMachine, JsonStates
from src.llm.vocabulary import Vocabulary


class TokenFilter:
    def __init__(self, state_machine: StateMachine, vocab: Vocabulary):
        self.state_machine = state_machine
        self.vocab = vocab

    def get_allowed_token_ids(self) -> list[int]:
        allowed_strings: list[str] = []
        if self.state_machine.current_state == JsonStates.START:
            allowed_strings = ["{"]

        elif self.state_machine.current_state == JsonStates.EXPECTING_KEY:
            allowed_strings = ['"']

        elif self.state_machine.current_state == JsonStates.EXPECTING_COLON:
            allowed_strings = [":"]

        elif self.state_machine.current_state == JsonStates.EXPECTING_VALUE:

            if self.state_machine.current_key in ["name", "prompt"]:
                allowed_strings = ['"']

            elif self.state_machine.current_key == "parameters":
                allowed_strings = ["{"]

        elif self.state_machine.current_state == JsonStates.EXPECTING_COMMA_OR_CLOSE:
            allowed_strings = [",", "}"]

        spaced_strings = []

        for s in allowed_strings:
            spaced_strings.append("Ġ" + s)

        final_allowed_strings = allowed_strings + spaced_strings

        return self._find_matching_token_ids(final_allowed_strings)

    def _find_matching_token_ids(self, allowed_strings: list[str]) -> list[int]:
        valid_ids: list[int] = []
        for token_id, token_str in self.vocab.id_to_token.items():
            for allowed_s in allowed_strings:
                if token_str.startswith(allowed_s):
                    valid_ids.append(token_id)
                    break

        return valid_ids
"{"
