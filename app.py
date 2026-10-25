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

# Pool per calibration-dashboard directive. Order: flash-latest (tracking alias)
# -> 2.5-flash -> 3.8-flash (request-verified solver). 2.5-flash is
# request-verified to 404 on this key — a deterministic fast-fail hop (~1 s);
# included in case Google re-enables it. Each hop is capped at
# PER_MODEL_TIMEOUT_S so a stalled endpoint is abandoned within ~12 s instead of
# letting the SDK's internal 503 retries run 28 s+.
MODELS = ["gemini-4-argon", "gemini-3.7-flash", "gemini-3.6-flash"]
# 2.5-flash-lite now returns request-404 (removed); 3.5/3.6/3.8 flash generations
# in a sustained model-wide "high demand" incident. 4-argon is the newest
# generation on a different serving stack — tried first.
# Generation-diverse chain: 3.8-flash has been in sustained "high demand" 503
# since yesterday evening (paid tier 1 affected too — model-level overload, NOT
# key/QoS). 3.6/3.5 generations are separate serving silos; 2.5-flash-lite as
# the stable last resort. All four verified present for this key via GET /models.

# Per-attempt cap: fast-failover directive (10-12 s per model; max 3 models =>
# ~36 s worst-case chain, still inside the ESP32 client's 65 s HTTP ceiling).
# The device retransmits its payload up to 3 times 15 s apart, so slow-but-real
# solves beyond a single 12 s window are retried by the client rather than held
# by a long server block.
PER_MODEL_TIMEOUT_S = 12
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