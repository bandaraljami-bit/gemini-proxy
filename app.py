import os
import base64
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

# Pool of vision-capable Flash models to try sequentially on capacity/model errors.
# Ordered newest-first; only models available to this API key (2.5-flash is blocked
# for new accounts; 2.0-flash and 1.5-flash are retired - all return immediate 404s).
MODELS = ["gemini-3.8-flash", "gemini-3.8-flash-lite"]

@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

@app.route('/models', methods=['GET'])
def models():
    """Diagnostic: list models this API key can actually use."""
    try:
        return Response(
            "\n".join(m.name for m in client.models.list()),
            mimetype='text/plain'
        )
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
                response = client.models.generate_content(
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
                clean_text = response.text.replace('**', '').replace('`', '').replace('*', '').strip()
                return Response(clean_text, mimetype='text/plain')
            except Exception as err:
                last_error = err
                print(f"Model {model_name} failed: {err}. Falling back to next model...")
                continue

        return str(last_error), 500

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)