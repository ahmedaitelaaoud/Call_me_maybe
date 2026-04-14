from enum import Enum, auto
from typing import Optional

class JsonStates(Enum):
    START = auto()
    EXPECTING_KEY = auto()
    EXPECTING_COLON = auto()
    EXPECTING_VALUE = auto()
    EXPECTING_COMMA_OR_CLOSE = auto()
    DONE = auto()


class StateMachine:
    def __init__(self):
        self.current_state = JsonStates.START
        self.generated_string = ""
        self.current_key: Optional[str] = None

    def consume(self, text: str) -> None:
        self.generated_string += text
        self._update_state()

    def _update_state(self) -> None:
        # 1. Waiting to open the JSON object
        if self.current_state == JsonStates.START:
            if "{" in self.generated_string:
                self.current_state = JsonStates.EXPECTING_KEY
                self.generated_string = ""
        # 2. Waiting for a string key like "name"
        elif self.current_state == JsonStates.EXPECTING_KEY:
            if self.generated_string.count('"') == 2:
                self.current_key = self.generated_string.replace('"', '').strip()
                self.current_state = JsonStates.EXPECTING_COLON
                self.generated_string = ""
        # 3. Waiting for the colon separator
        elif self.current_state == JsonStates.EXPECTING_COLON:
            if ":" in self.generated_string:
                self.current_state = JsonStates.EXPECTING_VALUE
                self.generated_string = ""
        # 4. Waiting for the value (logic depends on what key we just parsed)
        elif self.current_state == JsonStates.EXPECTING_VALUE:
            if self.current_key in ["name", "prompt"]:
                if self.generated_string.count('"') == 2:
                    self.current_key = self.generated_string.replace('"', '').strip()
                    self.current_state = JsonStates.EXPECTING_COMMA_OR_CLOSE
                    self.generated_string = ""
            elif self.current_key is "parameters":
                if "}" in self.generated_string:
                    self.current_state = JsonStates.EXPECTING_COMMA_OR_CLOSE
                    self.generated_string = ""
        # 5. Waiting to either start a new key or finish the whole object
        elif self.current_state == JsonStates.EXPECTING_COMMA_OR_CLOSE:
            if "," in self.generated_string:
                self.current_state = JsonStates.EXPECTING_KEY
                self.generated_string = ""
            elif "}" in self.generated_string:
                self.current_state = JsonStates.DONE
                self.generated_string = ""
