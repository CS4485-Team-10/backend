"""Local Bedrock smoke test: load .env, chat with a fixed Qwen3 text model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT.parent / ".env")

from pipelines.shared.llm_providers import BedrockProvider  # noqa: E402

DEFAULT_BEDROCK_MODEL = "qwen.qwen3-235b-a22b-2507-v1:0"


def main() -> None:
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    model = os.environ.get("BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)
    provider = (
        BedrockProvider(model=model, region=region)
        if region
        else BedrockProvider(model=model)
    )

    print(f"Model: {provider.model}\nRegion: {provider.region}\nEmpty line to exit.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        text = provider.generate_response(
            system="You are a helpful assistant.",
            user_prompt=line,
        )
        print(text)
        print()


if __name__ == "__main__":
    main()
