import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Chat Completions API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, providing more control over the HTTP layer and demonstrating
    how to interact with the API directly.
    """

    def _headers(self) -> dict:
        # BaseOpenAIClient stores the key already formatted as "Bearer <key>"
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        history = [Message(role=Role.SYSTEM, content=self._system_prompt)]
        history.extend(m for m in messages if m.role != Role.SYSTEM)
        return [m.to_dict() for m in history]

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no choices.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        response = requests.post(
            self._endpoint,
            headers=self._headers(),
            json={
                "model": self._model_name,
                "messages": self._prepare_messages(messages),
            },
        )
        if response.status_code != 200:
            raise Exception(f"OpenAI API error {response.status_code}: {response.text}")

        choices = response.json().get("choices", [])
        if not choices:
            raise ValueError("OpenAI API returned no choices")

        content = choices[0]["message"].get("content") or ""
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed token-by-token using OpenAI's SSE format,
        with each chunk printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
        """
        payload = {
            "model": self._model_name,
            "messages": self._prepare_messages(messages),
            "stream": True,
        }
        content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=self._headers(), json=payload) as response:
                if response.status != 200:
                    raise Exception(f"OpenAI API error {response.status}: {await response.text()}")

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
