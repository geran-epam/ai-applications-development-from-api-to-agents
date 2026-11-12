import base64
import os
from datetime import datetime

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "t3out")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# https://developers.openai.com/api/reference/resources/images/methods/generate
# ---
# Request:
# curl -X POST "https://api.openai.com/v1/images/generations" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -H "Content-type: application/json" \
#     -d '{
#         "model": "gpt-image-2",
#         "prompt": "smiling catdog."
#     }'
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO...,
#     }
#   ]
# }

#TODO:
# You need to create some images with `gpt-image-2` model:
#   - Generate an image with 'Smiling catdog'
#   - Decode and save it locally
# ---
# Hints:
#   - Use OpenAIClientT3 to connect to OpenAI API
#   - Use /v1/images/generations endpoint
#   - The image will be returned in base64 format


# NOTE: OpenRouter has no working `gpt-image-2` (the OpenAI provider rejects our region with HTTP 403 and there is
# no /v1/images/generations route to another provider), so images are generated via /v1/chat/completions
# with `modalities` and an image-capable model. The image comes back as a base64 data URL in `message.images`.
IMAGE_MODEL = "google/gemini-2.5-flash-image"

client = OpenAIClientT3(endpoint=f"{OPENAI_HOST}/v1/chat/completions")

# The base64 payload is huge, so don't print the response
data = client.call(
    print_response=False,
    model=IMAGE_MODEL,
    modalities=["image", "text"],
    messages=[{"role": "user", "content": "Smiling catdog"}],
)

data_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
filename = os.path.join(OUTPUT_DIR, f"catdog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
with open(filename, "wb") as f:
    f.write(base64.b64decode(data_url.split(",", 1)[1]))
print(f"Image saved to {os.path.abspath(filename)}")
