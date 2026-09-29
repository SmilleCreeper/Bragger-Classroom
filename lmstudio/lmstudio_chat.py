import argparse
import sys

from lmstudio_client import (
    DEFAULT_BASE_URL,
    LMStudioError,
    chat_stream,
    first_resident_model,
    format_model,
    list_models,
    require_resident_model,
)

TURNS = [
    "What is your name?",
    "What is the capital of France?",
    "That's been a test.",
    "Please summarize this chat.",
]


def run_turn(base_url, model, messages, user_text, max_tokens, temperature):
    print(f"\n[user] {user_text}")
    messages.append({"role": "user", "content": user_text})
    content_parts = []
    reasoning_parts = []
    print("[model] ", end="", flush=True)
    for content, reasoning in chat_stream(
        base_url, model, messages, max_tokens=max_tokens, temperature=temperature
    ):
        if reasoning:
            reasoning_parts.append(reasoning)
            print(reasoning, end="", flush=True)
        if content:
            content_parts.append(content)
            print(content, end="", flush=True)
    print()
    if not content_parts and reasoning_parts:
        print("[model] (no final content - the model spent the whole budget reasoning)")
    reply = "".join(content_parts)
    messages.append({"role": "assistant", "content": reply or "".join(reasoning_parts)})
    return reply


def main():
    parser = argparse.ArgumentParser(description="Hold a short conversation with a local LM Studio model.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=None, help="Model id; must already be loaded on the server.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    try:
        _, models = list_models(args.base_url)
        if args.model:
            model = require_resident_model(models, args.model)
        else:
            model = first_resident_model(models)
    except LMStudioError as exc:
        print(exc)
        return

    print("Server " + args.base_url)
    print("Model  " + format_model(model))

    messages = []
    for turn in TURNS:
        run_turn(args.base_url, model["id"], messages, turn, args.max_tokens, args.temperature)

    print(f"\nTurns held: {len(TURNS)}  |  messages in context: {len(messages)}")


if __name__ == "__main__":
    try:
        main()
    except LMStudioError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted")
