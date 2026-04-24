"""Concrete LLM provider implementations shared across pipelines."""

from __future__ import annotations

import os
from pathlib import Path
import requests
from .data_models import LLMProvider

# Bedrock `modelId` (including provider prefix, e.g. `qwen.*`). Not an Ollama short name.
DEFAULT_BEDROCK_MODEL_ID = "qwen.qwen3-vl-235b-a22b"


def _load_bedrock_dotenv() -> None:
    """Load ``backend/.env`` then the parent of backend (e.g. repo root) ``.env``.

    Same order as ``test.py`` (backend first; later file does not override by default).
    No-op if ``python-dotenv`` is not installed. Missing files are ignored.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # pipelines/shared/llm_providers.py -> parents[2] == backend (package root for this app)
    backend = Path(__file__).resolve().parents[2]
    load_dotenv(backend / ".env")
    load_dotenv(backend.parent / ".env")


def _resolve_bedrock_model_id(explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    return (
        (os.environ.get("BEDROCK_MODEL") or "").strip()
        or (os.environ.get("LLM_MODEL") or "").strip()
        or DEFAULT_BEDROCK_MODEL_ID
    )


class OllamaProvider(LLMProvider):
    """Call local Ollama using OpenAI-compatible API."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3",
        base_url: str = "http://localhost:11434/v1",
    ):
        super().__init__(provider="ollama", model=model)
        self.base_url = base_url.rstrip("/")

    def generate_response(self, *, system: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        if not resp.ok:
            err_msg = resp.text
            try:
                err = resp.json()
                if isinstance(err.get("error"), dict):
                    err_msg = err["error"].get("message", err_msg)
                elif isinstance(err.get("error"), str):
                    err_msg = err["error"]
            except Exception:
                pass
            hint = ""
            if "not found" in err_msg.lower():
                hint = " Run `ollama pull <model>` (see LLM_MODEL) to download a model first."
            raise RuntimeError(
                f"Ollama API error ({resp.status_code}): {err_msg}.{hint}"
            ) from None
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class BedrockProvider(LLMProvider):
    """Call Amazon Bedrock Converse API.

    On construction, this loads ``.env`` from the backend package and the parent
    directory (if ``python-dotenv`` is available), then reads configuration.

    **Model ID:** pass ``model=``, or set ``BEDROCK_MODEL`` (preferred for Bedrock)
    or ``LLM_MODEL`` in the environment, or rely on
    :data:`DEFAULT_BEDROCK_MODEL_ID` (full Bedrock modelId, e.g. ``qwen.qwen3-vl-...``).

    **Authentication** (boto3 default chain, same as AWS CLI):

    - ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, optional ``AWS_SESSION_TOKEN``
    - ``~/.aws/credentials`` and ``~/.aws/config``
    - Execution role in Lambda/EC2, etc.

    **Region:** constructor ``region``, else ``AWS_REGION`` or ``AWS_DEFAULT_REGION``,
    else ``us-east-1``. Use a region where the model is available.
    """

    name = "bedrock"

    def __init__(self, model: str | None = None, region: str | None = None) -> None:
        _load_bedrock_dotenv()
        resolved = _resolve_bedrock_model_id(model)
        super().__init__(provider="bedrock", model=resolved)
        self.region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self._client = None

    def _get_client(self):
        """Lazy-load boto3 client (credentials resolved by boto3, not hard-coded)."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for BedrockProvider. "
                    "Install it with: pip install boto3"
                ) from None

            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region,
            )
        return self._client

    def generate_response(self, *, system: str, user_prompt: str) -> str:
        """Call Bedrock and return model text."""
        client = self._get_client()

        try:
            response = client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                system=[{"text": system}],
                inferenceConfig={
                    "temperature": 0.3,
                    "maxTokens": 4096,
                },
            )

            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content:
                raise RuntimeError("Bedrock returned empty response")

            return content[0].get("text", "")

        except Exception as e:
            error_msg = str(e)
            if "AccessDeniedException" in error_msg:
                raise RuntimeError(
                    "Bedrock access denied: check IAM allows bedrock:InvokeModel (or "
                    "equivalent) for this model in this region. Locally, ensure "
                    "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and optional "
                    "AWS_SESSION_TOKEN) are set—e.g. via .env loaded before use—or use "
                    "~/.aws/credentials. On Lambda, use the function execution role, not .env."
                ) from e
            elif (
                "ResourceNotFoundException" in error_msg
                or "ValidationException" in error_msg
            ):
                raise RuntimeError(
                    f"Model '{self.model}' not found in region '{self.region}'. "
                    "Verify model ID and ensure you have access to it in Bedrock."
                ) from e
            else:
                raise RuntimeError(f"Bedrock API error: {error_msg}") from e
