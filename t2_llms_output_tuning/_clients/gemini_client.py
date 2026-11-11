import json

import requests

from commons.constants import GEMINI_API_KEY, GEMINI_ENDPOINT
from t2_llms_output_tuning._clients._base_client import AIClient
from commons.models.message import Message
from commons.models.role import Role


class GeminiAIClient(AIClient):
    """
    Gemini via OpenRouter. OpenRouter has an OpenAI-compatible API: POST /chat/completions with the model in the body
    (Google's own API puts the model in the URL: .../models/{model}:generateContent).
    The tasks keep using Gemini-style `generationConfig`, it is translated to OpenRouter parameters here.
    """

    def __init__(self, model_name: str):
        super().__init__(
            endpoint=GEMINI_ENDPOINT,
            model_name=model_name,
            api_key=f"Bearer {GEMINI_API_KEY}",
            api_key_header_name="Authorization"
        )

    def response(
            self,
            messages: list[Message],
            print_request: bool,
            print_only_content: bool,
            **kwargs
    ) -> Message:
        headers = {
            "Authorization": self._api_key,
            "Content-Type": "application/json"
        }
        request_data = {
            "model": self._model_name,
            "messages": [message.to_dict() for message in messages],
            **self._from_generation_config(kwargs.get("generationConfig", {}))
        }
        if print_request:
            self._print_request(request_data, headers)

        response = requests.post(url=self._endpoint, headers=headers, json=request_data)

        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content") or ""
                print("" + "=" * 50 + " RESPONSE " + "=" * 50)
                if print_only_content:
                    print(content)
                else:
                    print(json.dumps(data, indent=2, sort_keys=True))
                print("=" * 109)
                return Message(Role.ASSISTANT, content)
            raise ValueError("No Choice has been present in the response")
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    @staticmethod
    def _from_generation_config(config: dict) -> dict:
        """Translate Gemini `generationConfig` fields to OpenRouter request fields."""
        params = {"max_tokens": config.get("maxOutputTokens", 1024)}
        if "temperature" in config:
            params["temperature"] = config["temperature"]
        if "topP" in config:
            params["top_p"] = config["topP"]
        if "topK" in config:
            params["top_k"] = config["topK"]
        if config.get("responseMimeType") == "application/json" and "responseSchema" in config:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": config["responseSchema"]},
            }
        elif config.get("responseMimeType") == "application/json":
            params["response_format"] = {"type": "json_object"}
        thinking_budget = config.get("thinkingConfig", {}).get("thinkingBudget")
        if thinking_budget is not None:
            params["reasoning"] = {"max_tokens": thinking_budget}
        return params
