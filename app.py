from flask import Flask, render_template, request, jsonify
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import csv
import os
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from groq import Groq   # NEW: Groq API
from huggingface_hub import login

app = Flask(__name__)

# -------------------------------
# LOAD ENV + GROQ CLIENT
# -------------------------------
load_dotenv()

print("LOADED KEY:", os.getenv("GROQ_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# PATHS / CONFIG
# -------------------------------
HF_MODEL_ID = "Manasa-Raghavendra07/mental_stress_model"
HF_TOKEN = os.getenv("HF_TOKEN")

login(token=HF_TOKEN)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_FOLDER, "predictions.csv")
MAX_HISTORY = 20

# -------------------------------
# LOAD STRESS DETECTION MODEL
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    HF_MODEL_ID,
    token=HF_TOKEN,
    subfolder="model"
)

classifier = DistilBertForSequenceClassification.from_pretrained(
    HF_MODEL_ID,
    token=HF_TOKEN,
    subfolder="model"
)

classifier.to(device)
classifier.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

# -------------------------------
# Logging helpers (CSV)
# -------------------------------
os.makedirs(LOG_FOLDER, exist_ok=True)

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
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = classifier(**enc)
        probs = torch.softmax(out.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label_idx = int(pred.item())
    label_str = "Stress" if label_idx == 1 else "No Stress"
    return label_str, float(confidence.item())


# -------------------------------
# GROQ Chatbot reply generator
# -------------------------------
# Global store for conversation history per session
conversation_history = {}

def generate_chatbot_reply(user_message, session_id="default"):
    """Generate a ChatGPT-style conversational reply using Groq."""

    # Initialize session history
    if session_id not in conversation_history:
        conversation_history[session_id] = [
            {"role": "system", "content": "You are a helpful, friendly, conversational AI assistant similar to ChatGPT. Respond naturally and clearly."}
        ]

    # Add user's message
    conversation_history[session_id].append(
        {"role": "user", "content": user_message}
    )

    try:
        # Generate AI reply
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation_history[session_id],
            max_tokens=200
        )

        ai_response = response.choices[0].message.content

        # Save reply to history
        conversation_history[session_id].append(
            {"role": "assistant", "content": ai_response}
        )

        return ai_response

    except Exception as e:
        print("Groq API Error:", e)
        return "Something went wrong, but I'm here. Try saying that again!"






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

    label, conf = detect_stress(text)
    log_prediction(text, label, round(conf, 4))

    return jsonify({"result": label, "confidence": round(conf, 4)})


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

    return jsonify({
        "reply": reply_text
    })




# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
