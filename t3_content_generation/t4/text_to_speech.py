import json
import os
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "t3out")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Voice:
    alloy: str = 'alloy'
    ash: str = 'ash'
    ballad: str = 'ballad'
    coral: str = 'coral'
    echo: str = 'echo'
    fable: str = 'fable'
    nova: str = 'nova'
    onyx: str = 'onyx'
    sage: str = 'sage'
    shimmer: str = 'shimmer'


# NOTE: `gpt-4o-mini-tts` does not exist on OpenRouter, so `hexgrad/kokoro-82m` is used (it returns mp3).
# It has its own voices and ignores `instructions`, so the OpenAI voices above are not accepted by it.
class KokoroVoice:
    heart: str = 'af_heart'
    bella: str = 'af_bella'
    nova: str = 'af_nova'
    adam: str = 'am_adam'
    michael: str = 'am_michael'


# https://developers.openai.com/api/docs/guides/text-to-speech
# Request:
# curl https://api.openai.com/v1/audio/speech \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "hexgrad/kokoro-82m",
#     "input": "Why can't we say that black is white?",
#     "voice": "coral",
#     "instructions": "Speak in a cheerful and positive tone."
#   }' \
# Response:
#   bytes with audio

#TODO:
# You need to convert text to speech:
#   - Create Client that will go to speech OpenAI API
#   - Call API
#   - Get response and save as .mp3 file
# ---
# Hints:
#   - Use /v1/audio/speech endpoint
#   - Use gpt-4o-mini-tts model



def text_to_speech(text: str, voice: str, instructions: str) -> str:
    response = requests.post(
        url=f"{OPENAI_HOST}/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "hexgrad/kokoro-82m",
            "input": text,
            "voice": voice,
            "instructions": instructions,
            "response_format": "mp3",
        },
    )
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    filename = os.path.join(OUTPUT_DIR, f"speech_{voice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Saved {os.path.abspath(filename)}")
    return filename


text_to_speech(
    "Why can't we say that black is white?",
    KokoroVoice.heart,
    "Speak in a cheerful and positive tone.",
)
text_to_speech(
    "Why can't we say that black is white?",
    KokoroVoice.adam,
    "Speak slowly in a calm, deep and serious tone.",
)
