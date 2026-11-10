import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def _headers(self) -> dict:
        return {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[Message], **kwargs) -> dict:
        # Gemini uses the role 'model' instead of 'assistant'; system messages go to system_instruction
        return {
            "system_instruction": {"parts": [{"text": self._system_prompt}]},
            "contents": [
                {
                    "role": "model" if m.role == Role.ASSISTANT else "user",
                    "parts": [{"text": m.content}],
                }
                for m in messages
                if m.role != Role.SYSTEM
            ],
            "generationConfig": {"maxOutputTokens": kwargs.get("max_tokens", 1024)},
        }

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        url = f"{self._endpoint}/{self._model_name}:generateContent"
        response = requests.post(url, headers=self._headers(), json=self._payload(messages, **kwargs))
        if response.status_code != 200:
            raise Exception(f"Gemini API error {response.status_code}: {response.text}")

        candidates = response.json().get("candidates", [])
        if not candidates:
            raise ValueError("Gemini API returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join(part.get("text", "") for part in parts)
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        url = f"{self._endpoint}/{self._model_name}:streamGenerateContent?alt=sse"
        content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._headers(), json=self._payload(messages, **kwargs)) as response:
                if response.status != 200:
                    raise Exception(f"Gemini API error {response.status}: {await response.text()}")

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue

                    chunk = json.loads(line[len("data: "):])
                    for candidate in chunk.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            text = part.get("text", "")
                            print(text, end="", flush=True)
                            content += text

        return Message(role=Role.ASSISTANT, content=content)
