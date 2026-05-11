import argparse
from src.json_loader import load_function_definition, load_prompts
from src.constrained_decoding import build_system_prompt, load_vocabulary
from src.constrained_decoding import build_json_valid_ids, get_best_valid_token, extract_complete_json
from llm_sdk import Small_LLM_Model


def parse_args():
    parse = argparse.ArgumentParser(
        description="Translate natural language into function calls..."
    )

    parse.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )

    parse.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )

    parse.add_argument(
        "--output",
        type=str,
        default="data/output/functions_result.json"
    )

    parse.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B"
    )

    return parse.parse_args()


def main():
    print("ᐅ Starting function and prompts...")
    args = parse_args()
    print("Loading functions and prompts...")
    functions = load_function_definition(args.functions_definition)
    if not functions:
        raise RuntimeError(
            "No function definition found. Please provide at least one."
        )
    prompts = load_prompts(args.input)
    if not prompts:
        raise RuntimeError(
            "No prompt found. Please provide at least one."
        )

    print("Building a system prompt")
    system = build_system_prompt(functions)
    print(system)

    print(f"Loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except OSError:
        raise RuntimeError(
            f"Model {args.model} not found."
        )
    print("Building valid token IDs...")
    vocab = load_vocabulary(model)
    valid_ids = build_json_valid_ids(vocab)

    print("Processing prompts...")
    for p in prompts:
        prompt = p.prompt
        full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
        input_ids = model.encode(full_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated = []
        all_generated.extend(model.encode('{"name": "')[0].tolist())

        for _ in range(50):
            logits = model.get_logits_from_input_ids(generated_ids + all_generated)
            next_id = get_best_valid_token(logits, valid_ids)
            all_generated.append(next_id)
            text = model.decode(all_generated)

            clean_json = extract_complete_json(text)
        print(clean_json)

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}")
