import argparse
from src.json_loader import load_function_definition, load_prompts
from src.constrained_decoding import (
    build_system_prompt,
    load_vocabulary,
    build_json_valid_ids,
    get_best_valid_token,
    extract_complete_json
)
from llm_sdk import Small_LLM_Model
import json
import time
import os


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
        default="data/output/function_calling_results.json"
    )

    parse.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B"
    )

    return parse.parse_args()


def main():
    print("🚀 Starting function calling pipeline...")
    args = parse_args()
    print("📂 Loading functions and prompts...")
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

    print("🧠 Building a system prompt...")
    system = build_system_prompt(functions)

    print(f"🤖 Loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except OSError:
        raise RuntimeError(
            f"Model {args.model} not found."
        )
    print("⚙️  Building valid token IDs...")
    vocab = load_vocabulary(model)
    valid_ids = build_json_valid_ids(vocab)

    all_results = []
    start_time = time.time()

    print("⏱️  Processing prompts...")
    for test_case in prompts:
        prompt = test_case.prompt
        full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
        input_ids = model.encode(full_prompt)
        base_input_ids = input_ids[0].tolist()

        llm_json_response = {"name": "none", "parameters": {}}
        newly_generated_ids = []
        newly_generated_ids.extend(model.encode('{"name": "')[0].tolist())

        for _ in range(50):
            logits = model.get_logits_from_input_ids(
                base_input_ids + newly_generated_ids
            )
            next_id = get_best_valid_token(logits, valid_ids)
            newly_generated_ids.append(next_id)

            decoded_output = model.decode(newly_generated_ids)

            extracted_json_string = extract_complete_json(decoded_output)
            if extracted_json_string:
                try:
                    llm_json_response = json.loads(extracted_json_string)
                    if "args" in llm_json_response:
                        llm_json_response["parameters"] = \
                            llm_json_response.pop("args")
                    break  # Early stopping optimization
                except Exception:
                    pass

        if not extracted_json_string:
            llm_json_response = {"name": "none", "parameters": {}}

        all_results.append({
            "prompt": prompt,
            "name": llm_json_response.get("name", "none"),
            "parameters": llm_json_response.get("parameters", {})
        })

        if llm_json_response.get("name", "none") != "none":
            print(
                f"  ✅ Generated call -> {llm_json_response['name']}"
                f"({llm_json_response.get('parameters', {})})"
            )
        else:
            print("  ❌ [ERROR] Could not generate function call.")

    total_time = time.time() - start_time

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n💾 Saved results to: {args.output}")
    print(f"🎉 Completed in {total_time:.2f} seconds")
    print(
        f"⚡ Average time per prompt: "
        f"{total_time/len(prompts):.2f} seconds"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"Error: {e}")
