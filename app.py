import os
import io
import base64
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from flask import Flask, request, Response
from PIL import Image, ImageOps, ImageEnhance
from google import genai
from google.genai import types

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ---- Post-processing pipeline for ESP32-CAM frames ----
# The device sensor cannot rotate (its only physical corrections are 180-degree
# flips), so orientation/level normalization happens here, once, at the proxy.
def _preprocess(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.rotate(90, expand=True)          # fix the camera's 90-deg CCW mounting
        img = ImageOps.autocontrast(img, cutoff=1) # normalize white/black levels (blowout-safe)
        w, h = img.size
        img = img.crop((int(w*0.04), int(h*0.04), int(w*0.96), int(h*0.96)))  # trim vignette border
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        return buf.getvalue()
    except Exception:
        # Unparseable frame: pass the original bytes through untouched
        return image_bytes

SYSTEM_PROMPT = """
You are a calculator parsing a camera feed. If the image contains several problems, solve every one you can read, one line each, in reading order.
Follow these absolute rules:
1. For each problem output at most two lines: the problem restated briefly, then the final answer (plus one brief line of working if essential).
2. DO NOT use any Markdown formatting (no asterisks, no bold, no LaTeX).
3. Insert manual line breaks (\n) every 18 characters so the text wraps correctly on a tiny screen.
4. If a problem is unreadable, skip it silently - never guess.
"""

# Pool of vision-capable Flash models to try sequentially, verified live against
# this API key via GET /models (2.5-flash, 2.0-flash and 1.5-flash all 404 on
# requests despite 2.5 appearing in the list). gemini-flash-latest is a stable
# alias that always tracks the current flash generation.
MODELS = ["gemini-3.8-flash", "gemini-flash-latest"]

# Per-attempt cap. The google-genai==0.3.0 client exposes no timeout option, so its
# internal 503 retries can run unbounded; wrapping each attempt caps one model's
# worst case. Budget contract: 2 attempts x 28s = ~56s must fit inside the ESP32
# client's hard HTTPClient::setTimeout ceiling (uint16: max 65000 ms). During
# demand-spike flapping, wider windows beat more fallback hops.
PER_MODEL_TIMEOUT_S = 28
_executor = ThreadPoolExecutor(max_workers=2)

def _generate(model_name, image_bytes):
    return client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            "Solve this."
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1
        )
    )

@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

@app.route('/models', methods=['GET'])
def models():
    """Diagnostic: list models this API key can actually use."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={os.environ.get('GEMINI_API_KEY')}&pageSize=100"
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.load(r)
        names = "\n".join(m["name"] for m in payload.get("models", []))
        return Response(names, mimetype='text/plain')
    except Exception as e:
        return str(e), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return "Error: No image provided", 400

        image_bytes = base64.b64decode(data['image'])
        image_bytes = _preprocess(image_bytes)
        last_error = None

        for model_name in MODELS:
            try:
                future = _executor.submit(_generate, model_name, image_bytes)
                response = future.result(timeout=PER_MODEL_TIMEOUT_S)
                if not (response and response.text):
                    # Model ran but produced no text (blocked/empty/unreadable
                    # input). Treat as a model failure so fallback proceeds.
                    raise RuntimeError(f"{model_name}: no text in response")
                clean_text = response.text.replace('**', '').replace('`', '').replace('*', '').strip()
                return Response(clean_text, mimetype='text/plain')
            except Exception as err:
                last_error = err
                print(f"Model {model_name} failed: {err}. Falling back to next model...")
                continue

        return f"{type(last_error).__name__}: {last_error}", 500

    except Exception as e:
        return f"{type(e).__name__}: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)