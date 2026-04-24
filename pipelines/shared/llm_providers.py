"""Concrete LLM provider implementations shared across pipelines."""

from __future__ import annotations

import os
import requests
from .data_models import LLMProvider


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

    Authentication uses the default boto3 credential chain (same as AWS CLI):

    - Environment: ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, optional
      ``AWS_SESSION_TOKEN`` (e.g. after ``load_dotenv()`` loads a local ``.env``).
    - Shared config: ``~/.aws/credentials``, ``~/.aws/config``.
    - Execution role: Lambda / EC2 / etc. (no access keys in ``.env``).

    Region: constructor ``region``, else ``AWS_REGION`` / ``AWS_DEFAULT_REGION``,
    else ``us-east-1``. Use a region where your model is available.
    """

    name = "bedrock"

    def __init__(
        self,
        model: str = "qwen3-vl-235b-a22b",
        region: str | None = None,
    ):
        super().__init__(provider="bedrock", model=model)
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
