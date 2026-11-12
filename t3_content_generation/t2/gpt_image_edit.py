import base64
import os
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "t3out")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# https://developers.openai.com/api/reference/resources/images/methods/edit
# ---
# Request (multipart/form-data, NOT json):
# curl -X POST "https://api.openai.com/v1/images/edits" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -F "model=gpt-image-1" \
#     -F "image=@logo.png" \
#     -F "prompt=Add magical sparkles and glowing aura around the logo"
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": "Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO..."
#     }
#   ]
# }

#TODO:
# You need to edit an existing image with `gpt-image-2` model:
#   - Take a local image (e.g. 'logo.png') and a prompt describing the edit
#   - Send it to the OpenAI images edit API
#   - Decode the returned base64 image and save it locally
# ---
# Hints:
#   - Use /v1/images/edits endpoint
#   - The request must be 'multipart/form-data' (NOT json) — pass the image as a file and the prompt/model as form fields
#   - The edited image will be returned in base64 format


# NOTE: OpenRouter has no /v1/images/edits route (and `gpt-image-2` is blocked for our region), so the edit is done
# via /v1/chat/completions: the source image is sent as a base64 data URL together with the prompt and an
# image-capable model returns the edited image in `message.images`.
# (The multipart/form-data variant is used in t3/speech_to_text.py, where OpenRouter does support it.)
EDIT_MODEL = "google/gemini-2.5-flash-image"
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.png")

with open(LOGO_PATH, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode("utf-8")

response = requests.post(
    url=f"{OPENAI_HOST}/v1/chat/completions",
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
    json={
        "model": EDIT_MODEL,
        "modalities": ["image", "text"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Add magical sparkles, a glowing mystical aura and shimmering light "
                                             "around the logo"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_logo}"}},
                ],
            }
        ],
    },
)

if response.status_code != 200:
    raise Exception(f"HTTP {response.status_code}: {response.text}")

data_url = response.json()["choices"][0]["message"]["images"][0]["image_url"]["url"]
filename = os.path.join(OUTPUT_DIR, f"logo_edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
with open(filename, "wb") as f:
    f.write(base64.b64decode(data_url.split(",", 1)[1]))
print(f"Edited image saved to {os.path.abspath(filename)}")
