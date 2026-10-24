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

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return "Error: No image provided", 400

        image_bytes = base64.b64decode(data['image'])

        response = client.models.generate_content(
            model='gemini-3.8-flash',
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

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)