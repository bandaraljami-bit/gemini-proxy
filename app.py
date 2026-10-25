import os
import io
import base64
import json
import urllib.request
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from flask import Flask, request, Response
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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
        # NO fixed 90-degree rotation anymore: the on-device hmirror + the
        # sheet's final positioning produce UPRIGHT frames; the old rotation
        # (correct for the original 90-degree-rotated mount) was turning
        # readable text sideways -> "No problem visible".
        # NO grayscale/3x-contrast either: that combination clipped the faint
        # pencil strokes to white (model-view proved: sheet rendered solid
        # white, zero strokes visible = "no text" from every model). The
        # verified x=2 / expansion solves all went through THIS mild version:
        img = ImageOps.autocontrast(img, cutoff=1) # gentle level normalization only
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
# PER_MODEL_TIMEOUT_S (=25 s): peaks need the larger inference window, and
# the ESP32 retry stack re-fires the payload outside this window.
# letting the SDK's internal 503 retries run 28 s+.
MODELS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3.6-flash"]
# Dashboard (ai.dev/rate-limit) verified 3.8-flash at 6/1000 RPM, 62/10,000 RPD on
# tier 1 — quota is NOT the blocker for this endpoint. 4-argon dropped: its own
# tiny tier-1 quota bucket 429s immediately (new-flagship caps). flash-latest
# tracks the same family; 3.6-flash behind on a different serving silo.
# Generation-diverse chain: 3.8-flash has been in sustained "high demand" 503
# since yesterday evening (paid tier 1 affected too — model-level overload, NOT
# key/QoS). 3.6/3.5 generations are separate serving silos; 2.5-flash-lite as
# the stable last resort. All four verified present for this key via GET /models.

# Per-attempt cap: fast-failover directive (10-12 s per model; max 3 models =>
# ~36 s worst-case chain, still inside the ESP32 client's 65 s HTTP ceiling).
# The device retransmits its payload up to 3 times 15 s apart, so slow-but-real
# solves beyond a single 12 s window are retried by the client rather than held
# by a long server block.
PER_MODEL_TIMEOUT_S = 25
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
        hop_errors = []

        # ---- Parallel model race ----
        # All pool models are submitted CONCURRENTLY; the first one that returns
        # usable text wins the response. Failover latency = min(model latencies)
        # instead of the serial sum — a stalled/stormed generation never blocks
        # the others. Losers are cancelled; their threads idle out harmlessly.
        futures = {m: _executor.submit(_generate, m, image_bytes) for m in MODELS}

        try:
            for done_future in concurrent.futures.as_completed(
                    list(futures.values()), timeout=PER_MODEL_TIMEOUT_S):
                try:
                    response = done_future.result()
                    if response and response.text:
                        clean_text = response.text.replace('**', '').replace('`', '').replace('*', '').strip()
                        return Response(clean_text, mimetype='text/plain')
                    hop_errors.append("no text in response")
                except Exception as err:
                    hop_errors.append(f"{type(err).__name__}: {err}")
                    print(f"Model failed: {err}. Race continues...")
        except FutureTimeoutError:
            for fut in futures.values():
                fut.cancel()
            hop_errors.append(f"all {len(futures)} models exceeded {PER_MODEL_TIMEOUT_S}s")
            # Cancelled futures' threads may still be running; they idle out
            # harmlessly and must NOT block this response.

        return " | ".join(hop_errors) if hop_errors else "no model responded", 500

    except Exception as e:
        return f"{type(e).__name__}: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)