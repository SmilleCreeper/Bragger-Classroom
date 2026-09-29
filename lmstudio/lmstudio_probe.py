import argparse

from lmstudio_client import (
    DEFAULT_BASE_URL,
    LMStudioError,
    first_resident_model,
    format_model,
    list_models,
    resident_chat_models,
)


def main():
    parser = argparse.ArgumentParser(description="Discover models on a local LM Studio server.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    try:
        source, models = list_models(args.base_url)
        print(f"Server      {args.base_url}")
        print(f"List source {source}")
        print(f"Models      {len(models)}")
        print()
        for model in models:
            print("  " + format_model(model))
        resident = resident_chat_models(models)
        print()
        if not resident:
            print("No chat model is currently loaded — load one in LM Studio first.")
            return
        chosen = first_resident_model(models)
        print(f"First resident chat model: {chosen['id']}")
        if chosen["loaded_context"]:
            print(f"Resident context length  : {chosen['loaded_context']} tokens")
    except LMStudioError as exc:
        print(exc)


if __name__ == "__main__":
    main()
