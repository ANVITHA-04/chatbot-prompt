import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Use Groq-supported model
MODEL = "llama-3.1-8b-instant"

app = Flask(__name__)

def load_prompts(kind):
    with open(f"prompts/{kind}.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/api/ask")
def ask():
    data = request.get_json()
    function = data["function"]
    variant = data["variant"]
    user_input = data["input"]
    extra = data.get("extra", {})

    # Load prompt template
    prompts = load_prompts(function)
    if variant not in prompts:
        return jsonify({"answer": f"Error: Variant '{variant}' not found in prompts."})

    template = prompts[variant]

    filled_prompt = (template
        .replace("{q}", user_input)
        .replace("{text}", user_input)
        .replace("{genre}", extra.get("genre", ""))
        .replace("{theme}", extra.get("theme", ""))
        .replace("{character}", extra.get("character", ""))
        .replace("{topic}", extra.get("topic", ""))
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": filled_prompt}
            ]
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"Error: {str(e)}"

    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Bind to Render's PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
