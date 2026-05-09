import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# -------------------------------
# Load API key
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

# -------------------------------
# Initialize Groq client
# -------------------------------
client = Groq(api_key=GROQ_API_KEY)

# Model
MODEL = "llama-3.1-8b-instant"

# Flask app
app = Flask(__name__)

# -------------------------------
# Load prompt templates
# -------------------------------
def load_prompts(kind):
    try:
        with open(f"prompts/{kind}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print("Error loading prompts:", e)
        return {}

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Main API
# -------------------------------
@app.post("/api/ask")
def ask():
    try:
        data = request.get_json()

        # -------------------------------
        # Extract inputs safely
        # -------------------------------
        function = data.get("function")
        variant = data.get("variant")
        user_input = data.get("input", "").strip()
        extra = data.get("extra", {})

        # -------------------------------
        # Validate input
        # -------------------------------
        if not user_input:
            return jsonify({"answer": "⚠️ Please enter a valid prompt."})

        if not function or not variant:
            return jsonify({"answer": "⚠️ Missing function or variant."})

        # -------------------------------
        # Load prompt template
        # -------------------------------
        prompts = load_prompts(function)

        if not prompts:
            return jsonify({"answer": f"⚠️ No prompts found for function '{function}'."})

        if variant not in prompts:
            return jsonify({"answer": f"⚠️ Variant '{variant}' not found."})

        template = prompts[variant]

        # -------------------------------
        # Default values (IMPORTANT FIX)
        # -------------------------------
        genre = extra.get("genre") or "science fiction"
        theme = extra.get("theme") or "technology and human life"
        character = extra.get("character") or "a brilliant scientist"
        topic = extra.get("topic") or user_input

        # -------------------------------
        # Build final prompt
        # -------------------------------
        filled_prompt = (template
            .replace("{q}", user_input)
            .replace("{text}", user_input)
            .replace("{genre}", genre)
            .replace("{theme}", theme)
            .replace("{character}", character)
            .replace("{topic}", topic)
        )

        # Optional: guide model for missing details
        if not extra:
            filled_prompt += "\n\nMake reasonable assumptions if details are missing."

        # -------------------------------
        # Call Groq API
        # -------------------------------
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": filled_prompt}
            ]
        )

        answer = resp.choices[0].message.content

    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    return jsonify({"answer": answer})


# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)