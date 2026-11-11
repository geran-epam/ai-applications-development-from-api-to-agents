from openai import OpenAI, AsyncOpenAI

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class GeminiAIClient(AIClient):
    """
    Client for Gemini via OpenRouter using the official OpenAI SDK.

    OpenRouter exposes Gemini through an OpenAI-compatible API, so the OpenAI SDK is used
    (the Google GenAI SDK can only talk to Google's own endpoint).

    Attributes:
        _client (OpenAI): Synchronous client instance.
        _async_client (AsyncOpenAI): Asynchronous client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Gemini client with the OpenAI SDK pointed at OpenRouter.

        Args:
            endpoint (str): The chat completions endpoint URL; its base URL is passed to the SDK.
            model_name (str): The OpenRouter model ID (e.g., 'google/gemini-3-flash-preview').
            api_key (str): The OpenRouter API key for authentication.
            system_prompt (str): The system instruction to guide the model's behavior.
        """
        super().__init__(endpoint, model_name, api_key, system_prompt)
        # The SDK appends "/chat/completions" itself, so it needs the base URL only
        base_url = endpoint.removesuffix("/chat/completions")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        history = [Message(role=Role.SYSTEM, content=self._system_prompt)]
        history.extend(m for m in messages if m.role != Role.SYSTEM)
        return [m.to_dict() for m in history]

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Gemini.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            The system prompt is automatically prepended to the messages.
            The response is printed to stdout before being returned.
        """
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._prepare_messages(messages),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        content = response.choices[0].message.content or ""
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Gemini.

        The response is streamed chunk-by-chunk, with each text chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.
        """
        stream = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=self._prepare_messages(messages),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True,
        )
        content = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                content += delta
        print()
        return Message(role=Role.ASSISTANT, content=content)
