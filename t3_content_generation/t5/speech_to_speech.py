import base64
import os
from datetime import datetime

import requests

from commons.constants import GEMINI_FLASH_MODEL, OPENAI_API_KEY, OPENAI_HOST

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "t3out")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# https://developers.openai.com/api/docs/guides/audio#add-audio-to-your-existing-application

#TODO:
# You need to generate answer in audio format based on the audio message:
#   - Create Client that is similar with OpenAIClients but extracts from message audio (instead of content)
#   - Call API
#   - Get response as base64 content, decode and save as .mp3 file
# ---
# Hints:
#   - Use /v1/chat/completions endpoint
#   - Use gpt-4o-audio-preview model
#   - Use modalities=["text", "audio"]
#   - Use audio={"voice": "ballad", "format": "mp3"}
#   - Use similar method to encode audio as you have done for images encoding




# NOTE: `gpt-4o-audio-preview` is not available via OpenRouter (and `openai/gpt-audio` is blocked for our region),
# and no OpenRouter chat model returns speech directly. So speech-to-speech is built as two calls:
#   1. /v1/chat/completions with an audio-input model: question.mp3 (base64) -> text answer
#   2. /v1/audio/speech with a TTS model: text answer -> mp3
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "question.mp3")
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


def encode_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def post(endpoint: str, body: dict) -> requests.Response:
    response = requests.post(url=f"{OPENAI_HOST}{endpoint}", headers=HEADERS, json=body)
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    return response


# 1. audio question -> text answer
chat_response = post("/v1/chat/completions", {
    "model": GEMINI_FLASH_MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Answer the question asked in this audio concisely."},
                {"type": "input_audio", "input_audio": {"data": encode_audio(AUDIO_PATH), "format": "mp3"}},
            ],
        }
    ],
})
answer = chat_response.json()["choices"][0]["message"]["content"]
print("Answer:", answer)

# 2. text answer -> speech
speech_response = post("/v1/audio/speech", {
    "model": "hexgrad/kokoro-82m",
    "input": answer,
    "voice": "af_heart",
    "response_format": "mp3",
})

filename = os.path.join(OUTPUT_DIR, f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
with open(filename, "wb") as f:
    f.write(speech_response.content)
print(f"Saved {os.path.abspath(filename)}")
