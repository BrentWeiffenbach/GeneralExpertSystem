from google import genai
import os
from google.genai import types
from typing import Optional


class GeminiAPI:
    def __init__(self, model: str = "gemini-2.5-pro", system_prompt: str = "") -> None:
        self.api_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.model: str = model
        self.system_prompt: str = system_prompt
        self.client: genai.Client = genai.Client(api_key=self.api_key)

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def call(self, message: str) -> types.GenerateContentResponse:
        response: types.GenerateContentResponse = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=self.system_prompt),
            contents=message,
        )
        return response
