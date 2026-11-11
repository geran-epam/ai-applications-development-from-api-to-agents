import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Gemini via OpenRouter.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of an SDK.
    OpenRouter has an OpenAI-compatible API: POST /chat/completions with the model in the
    request body (Google's own API puts the model in the URL path) and Server-Sent Events
    for streaming.
    """

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[Message], **kwargs) -> dict:
        history = [Message(role=Role.SYSTEM, content=self._system_prompt)]
        history.extend(m for m in messages if m.role != Role.SYSTEM)
        return {
            "model": self._model_name,
            "messages": [m.to_dict() for m in history],
            "max_tokens": kwargs.get("max_tokens", 1024),
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
            ValueError: If the API response contains no choices.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        response = requests.post(self._endpoint, headers=self._headers(), json=self._payload(messages, **kwargs))
        if response.status_code != 200:
            raise Exception(f"Gemini API error {response.status_code}: {response.text}")

        choices = response.json().get("choices", [])
        if not choices:
            raise ValueError("Gemini API returned no choices")

        content = choices[0]["message"].get("content") or ""
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each text chunk is printed to stdout as it arrives.
        """
        payload = {**self._payload(messages, **kwargs), "stream": True}
        content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=self._headers(), json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Gemini API error {response.status}: {await response.text()}")

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        print(delta, end="", flush=True)
                        content += delta

        print()
        return Message(role=Role.ASSISTANT, content=content)
