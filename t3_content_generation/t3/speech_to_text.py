import json
import os

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/docs/guides/speech-to-text

#TODO:
# You need to transcribe 'audio_sample.mp3':
#   - Create Client that will go to transcriptions OpenAI API
#   - Call API and provide file (pay attention that you work with 'multipart/form-data')
#   - Get response with transcription
# ---
# Hints:
#   - Use /v1/audio/transcriptions endpoint
#   - Use whisper-1 or gpt-4o-transcribe model


AUDIO_PATH = os.path.join(os.path.dirname(__file__), "audio_sample.mp3")


def transcribe(model: str) -> str:
    with open(AUDIO_PATH, "rb") as audio_file:
        response = requests.post(
            url=f"{OPENAI_HOST}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data={"model": model},
            files={"file": ("audio_sample.mp3", audio_file, "audio/mpeg")},
        )
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    return response.json()["text"]


# NOTE: OpenAI's own transcription models (`whisper-1`, `gpt-4o-transcribe`) fail via OpenRouter (HTTP 502,
# the OpenAI provider is blocked for our region), so other transcription models are compared instead.
for model_name in ("openai/whisper-large-v3", "mistralai/voxtral-mini-transcribe", "google/gemini-3.5-transcribe"):
    print(f"=== {model_name} ===")
    try:
        print(transcribe(model_name))
    except Exception as e:
        print(f"Failed: {e}")
