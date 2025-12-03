from google import genai
import os
from google.genai import types
from google.genai.errors import ServerError
from typing import Any, List, Optional, Sequence, Union
import time


class GeminiAPI:
    def __init__(self, model: str = "gemini-2.5-flash", system_prompt: str = "") -> None:
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.model: str = model
        self.system_prompt: str = system_prompt
        self.client: genai.Client = genai.Client(api_key=self.api_key)

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def call(
        self,
        message: Union[str, Sequence[dict[str, Any]]],
        max_retries: int = 5,
        system_prompt: Optional[str] = None,
    ) -> types.GenerateContentResponse:
        base_delay = 2  # Start with 2 seconds
        normalized_contents = list(self._normalize_messages(message))
        active_system_prompt = system_prompt or self.system_prompt
        if active_system_prompt:
            normalized_contents.insert(
                0,
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "System instructions:\n"
                                + active_system_prompt
                                + "\n---"
                            )
                        }
                    ],
                },
            )

        for attempt in range(max_retries):
            try:
                response: types.GenerateContentResponse = self.client.models.generate_content(
                    model=self.model,
                    contents=normalized_contents,
                )
                return response
            except ServerError as e:
                # Check if it's a 503 error by examining the error details
                is_503 = False
                if hasattr(e, 'status_code'):
                    is_503 = getattr(e, 'status_code') == 503  # type: ignore
                elif hasattr(e, 'args') and len(e.args) > 0:
                    is_503 = '503' in str(e.args[0])
                else:
                    is_503 = '503' in str(e)
                
                if is_503 and attempt < max_retries - 1:
                    # Calculate exponential backoff: 2, 4, 8, 16, 32 seconds
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠ Model overloaded (503). Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # If it's not a 503 or we've exhausted retries, re-raise
                    raise
        
        # This should never be reached, but just in case
        raise ServerError(503, {'error': {'message': 'Max retries exceeded'}}, None)

    def embed(self, text: str, model: str = "text-embedding-004") -> List[float]:
        if not text.strip():
            return []

        response = self.client.models.embed_content(
            model=model,
            contents=text,
        )

        if not response.embeddings:
            return []

        values = response.embeddings[0].values or []
        return list(values)

    @staticmethod
    def extract_text(response: types.GenerateContentResponse) -> str:
        if hasattr(response, "text") and response.text is not None:
            return response.text

        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts or []
            return "\n".join(getattr(part, "text", "") for part in parts)

        return ""

    def _normalize_messages(
        self,
        message: Union[str, Sequence[dict[str, Any]]],
    ) -> List[dict[str, Any]]:
        if isinstance(message, str):
            return [
                {
                    "role": "user",
                    "parts": [{"text": message}],
                }
            ]

        normalized: List[dict[str, Any]] = []
        for item in message:
            role = item.get("role", "user")
            if "parts" not in item:
                text_value = item.get("text")
                if text_value is None:
                    text_value = item.get("content", "")
                normalized.append(
                    {
                        "role": role,
                        "parts": [{"text": text_value}],
                    }
                )
                continue

            parts_payload = item.get("parts", [])
            clean_parts = []
            for part in parts_payload:
                if isinstance(part, dict):
                    clean_parts.append({"text": part.get("text", "")})
                else:
                    clean_parts.append({"text": str(part)})

            normalized.append({"role": role, "parts": clean_parts})

        return normalized
