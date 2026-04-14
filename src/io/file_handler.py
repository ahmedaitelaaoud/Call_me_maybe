import json
import sys
from typing import Any


def load_json(filepath: str) -> Any:
    """
    Safely opens and parses a JSON file.
    Exits gracefully if the file is missing or corrupted.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.", file=sys.stderr)
        sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error: The file '{filepath}' contains invalid JSON.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred reading '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)
