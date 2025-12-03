# app.py
from flask import Flask, render_template, request, jsonify
import os
import csv
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

# Torch / Transformers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hugging Face hub
from huggingface_hub import login as hf_login

# Groq (optional)
try:
    from groq import Groq
except Exception:
    Groq = None

app = Flask(__name__)

# -------------------------------
# ENV / CLIENTS
# -------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Manasa-Raghavendra07/mental_stress_model")
HF_TOKEN = os.getenv("HF_TOKEN")  # must be set in Render if model is private

# Initialize groq client if available/key present (non-fatal)
groq_client = None
if GROQ_API_KEY and Groq is not None:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized.")
    except Exception as e:
        print("Warning: failed to initialize Groq client:", e)
else:
    if GROQ_API_KEY:
        print("Warning: groq package not installed; GROQ_API_KEY ignored.")
    else:
        print("No GROQ_API_KEY set; Groq features disabled.")

# Login to HF if token present (makes downloads for private repos possible)
if HF_TOKEN:
    try:
        hf_login(token=HF_TOKEN)
        print("Logged in to Hugging Face hub with HF_TOKEN.")
    except Exception as e:
        print("Warning: huggingface_hub.login() failed:", e)
else:
    print("HF_TOKEN not set — attempting anonymous access to Hugging Face model.")

# -------------------------------
# PATHS / CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_FOLDER, "predictions.csv")
MAX_HISTORY = 20

os.makedirs(LOG_FOLDER, exist_ok=True)

# -------------------------------
# Lazy model/tokenizer loader
# -------------------------------
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_model_loaded():
    global _tokenizer, _model, _device

    if _tokenizer is not None and _model is not None:
        return

    print("Loading model/tokenizer from Hugging Face:", HF_MODEL_ID)

    try:
        print("STEP 1: Loading tokenizer…")
        _tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID,
            token=HF_TOKEN if HF_TOKEN else None
        )
        print("Tokenizer loaded successfully.")

        print("STEP 2: Loading model…")
        _model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_ID,
            subfolder="model",    # <-- DEBUG: this might be wrong
            token=HF_TOKEN if HF_TOKEN else None
        )
        print("Model weights loaded successfully.")

        _model.to(_device)
        _model.eval()

        print("MODEL LOADED SUCCESSFULLY.")

    except Exception as e:
        print("\n❌ MODEL LOAD ERROR ❌")
        print("DETAILS:", e)
        print("HF_MODEL_ID =", HF_MODEL_ID)
        print("subfolder='model' probably wrong!")
        raise



# -------------------------------
# Logging helpers (CSV)
# -------------------------------
def log_prediction(text, result, confidence):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "result", "confidence"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text,
            result,
            f"{confidence:.4f}"
        ])

# -------------------------------
# Chat history memory
# -------------------------------
chat_histories = {}

def get_history(session_id=None):
    key = session_id or "global"
    if key not in chat_histories:
        chat_histories[key] = deque(maxlen=MAX_HISTORY)
    return chat_histories[key]

# -------------------------------
# Stress detection
# -------------------------------
def detect_stress(text):
    """
    Returns (label_str, confidence_float)
    """
    ensure_model_loaded()
    global _tokenizer, _model, _device

    if not _tokenizer or not _model:
        raise RuntimeError("Model/tokenizer not loaded.")

    # Tokenize and run model
    enc = _tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(_device)
    with torch.no_grad():
        out = _model(**enc)
        probs = torch.softmax(out.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label_idx = int(pred.item())
    label_str = "Stress" if label_idx == 1 else "No Stress"
    return label_str, float(confidence.item())

# -------------------------------
# Groq Chatbot reply generator (robust extraction)
# -------------------------------
conversation_history = {}

def _extract_groq_text(resp):
    """
    Groq SDK can return different shapes; attempt to extract assistant text robustly.
    """
    try:
        # Try common shape: response.choices[0].message.content
        choice = resp.choices[0]
        # some SDKs return a mapping-like object
        if hasattr(choice, "message") and getattr(choice.message, "content", None) is not None:
            return choice.message.content
        # fallback: maybe choice has 'text' attribute
        if getattr(choice, "text", None):
            return choice.text
        # fallback to stringifying
        return str(resp)
    except Exception:
        return str(resp)

def generate_chatbot_reply(user_message, session_id="default"):
    """Generate a ChatGPT-style conversational reply using Groq (if configured)."""
    # Initialize session history
    if session_id not in conversation_history:
        conversation_history[session_id] = [
            {"role": "system", "content": "You are a helpful, friendly, conversational AI assistant similar to ChatGPT. Respond naturally and clearly."}
        ]

    # Add user's message
    conversation_history[session_id].append({"role": "user", "content": user_message})

    if groq_client is None:
        # Groq not configured — fallback reply
        return "I can't reach the Groq API right now. I'm here to listen though — tell me more."

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation_history[session_id],
            max_tokens=200
        )
        ai_response = _extract_groq_text(response)
        conversation_history[session_id].append({"role": "assistant", "content": ai_response})
        return ai_response
    except Exception as e:
        print("Groq API Error:", e)
        return "Something went wrong with the chatbot. Try again later."

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json() or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "Empty input!"}), 400

    try:
        label, conf = detect_stress(text)
        # Log only predictions (not chatbot messages)
        log_prediction(text, label, round(conf, 4))
        return jsonify({"result": label, "confidence": round(conf, 4)})
    except Exception as e:
        # Return a 500 with the message (also printed in logs)
        print("Predict error:", e)
        return jsonify({"error": "Model error", "message": str(e)}), 500

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        logs = []
        if os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                dict_reader = csv.DictReader(f)
                for row in dict_reader:
                    logs.append({
                        "timestamp": row.get("timestamp", ""),
                        "text": row.get("text", ""),
                        "result": row.get("result", ""),
                        "confidence": row.get("confidence", "")
                    })
        return jsonify({"status": "success", "logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/chatbot_message", methods=["POST"])
def chatbot_response():
    payload = request.get_json() or {}
    user_message = payload.get("message", "")
    session_id = payload.get("session_id", "default")

    if not isinstance(user_message, str) or user_message.strip() == "":
        return jsonify({"reply": "Please type something."}), 400

    # PURE ChatGPT-like conversation — no stress check, no logging
    reply_text = generate_chatbot_reply(user_message, session_id=session_id)
    return jsonify({"reply": reply_text})

# -------------------------------
# Run (for local dev)
# -------------------------------
if __name__ == "__main__":
    # For local dev you can preload model to test (optional)
    if os.getenv("PRELOAD_MODEL", "0") == "1":
        ensure_model_loaded()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
