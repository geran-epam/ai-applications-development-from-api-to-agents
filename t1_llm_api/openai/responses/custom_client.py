import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIResponsesClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Responses API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with the Responses API directly
    and handle its unique event-based streaming format.
    """

    def _headers(self) -> dict:
        # BaseOpenAIClient stores the key already formatted as "Bearer <key>"
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _prepare_input(messages: list[Message]) -> list[dict]:
        # The system prompt goes into 'instructions', not into 'input'
        return [m.to_dict() for m in messages if m.role != Role.SYSTEM]

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no output text.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Uses the Responses API format with 'instructions' and 'input' parameters.
            The response is printed to stdout before being returned.
        """
        response = requests.post(
            self._endpoint,
            headers=self._headers(),
            json={
                "model": self._model_name,
                "instructions": self._system_prompt,
                "input": self._prepare_input(messages),
            },
        )
        if response.status_code != 200:
            raise Exception(f"OpenAI API error {response.status_code}: {response.text}")

        content = "".join(
            part.get("text", "")
            for item in response.json().get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        if not content:
            raise ValueError("OpenAI API returned no output text")

        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with event-based streaming.

        The Responses API uses a different SSE format than Chat Completions,
        with explicit event types and data fields.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses event-based Server-Sent Events (SSE) format.
            Listens for 'response.output_text.delta' events to build the response.
            Each line with "event: " specifies the event type, followed by "data: " with the payload.
        """
        payload = {
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": self._prepare_input(messages),
            "stream": True,
        }
        content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=self._headers(), json=payload) as response:
                if response.status != 200:
                    raise Exception(f"OpenAI API error {response.status}: {await response.text()}")

                # SSE frames look like "event: <type>" followed by "data: <json>"
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break

                    event = json.loads(data)
                    if event.get("type") == "response.output_text.delta":
                        delta = event["delta"]
                        print(delta, end="", flush=True)
                        content += delta

        return Message(role=Role.ASSISTANT, content=content)
