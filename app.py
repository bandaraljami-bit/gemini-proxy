import os
import base64
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from flask import Flask, request, Response
from google import genai
from google.genai import types

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are a calculator parsing a camera feed. Solve the equation or answer the question in the image.
Follow these absolute rules:
1. Output ONLY the final answer and maximum one brief line of working.
2. DO NOT use any Markdown formatting (no asterisks, no bold, no LaTeX).
3. Insert manual line breaks (\n) every 18 characters so the text wraps correctly on a tiny screen.
"""

# Pool of vision-capable Flash models to try sequentially, verified live against
# this API key via GET /models (2.5-flash, 2.0-flash and 1.5-flash all 404 on
# requests despite 2.5 appearing in the list). gemini-flash-latest is a stable
# alias that always tracks the current flash generation.
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"]

# Per-attempt cap. The google-genai==0.3.0 client exposes no timeout option, so its
# internal 503 retries can run unbounded; wrapping each attempt caps one model's
# worst case. Budget contract: 3 attempts x 20s = ~60s must fit inside the ESP32
# client's hard HTTPClient::setTimeout ceiling (uint16: max 65000 ms).
PER_MODEL_TIMEOUT_S = 20
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
        last_error = None

        for model_name in MODELS:
            try:
                future = _executor.submit(_generate, model_name, image_bytes)
                response = future.result(timeout=PER_MODEL_TIMEOUT_S)
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