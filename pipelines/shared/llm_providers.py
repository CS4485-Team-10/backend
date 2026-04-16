"""Concrete LLM provider implementations shared across pipelines."""

from __future__ import annotations

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
    """Call Amazon Bedrock Converse API."""

    name = "bedrock"

    def __init__(self, model: str = "qwen3-vl-235b-a22b", region: str = "us-east-1"):
        super().__init__(provider="bedrock", model=model)
        self.region = region
        self._client = None

    def _get_client(self):
        """Lazy-load boto3 client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for BedrockProvider. "
                    "Install it with: pip install boto3"
                ) from None

            self._client = boto3.client(
                service_name="bedrock-runtime", region_name=self.region
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
                    "AWS credentials not configured or invalid. "
                    "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables "
                    "or configure ~/.aws/credentials"
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
